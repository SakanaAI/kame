import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kame import server_oracle
from kame.deferred_logging import DeferredSessionLogger


class DummyServerState:
    def __init__(self, pending_user_text: str = "") -> None:
        self.pending_user_text = pending_user_text
        self.llm_event_queue = asyncio.Queue()

    def get_pending_user_text(self) -> str:
        return self.pending_user_text


def test_importing_server_oracle_does_not_create_logs_dir(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("MOSHI_LOG_DIR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    subprocess.run(
        [sys.executable, "-c", "import kame.server_oracle; print('ok')"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not (tmp_path / "logs").exists()


def test_plaintext_logs_are_written_only_when_configured(tmp_path: Path, monkeypatch) -> None:
    original_save_dir = server_oracle.SAVE_DIR
    try:
        monkeypatch.setattr(server_oracle, "SAVE_DIR", None)
        server_oracle._append_session_log("conversation.txt", "hello\n")
        assert not (tmp_path / "conversation.txt").exists()

        log_dir = tmp_path / "session-logs"
        server_oracle.configure_save_dir(str(log_dir))
        server_oracle._append_session_log("conversation.txt", "hello\n")
        assert (log_dir / "conversation.txt").read_text() == "hello\n"
    finally:
        server_oracle.SAVE_DIR = original_save_dir


def test_add_to_conversation_reuses_speaker_prefix_for_contiguous_chunks(tmp_path: Path) -> None:
    original_save_dir = server_oracle.SAVE_DIR
    original_conversation_text = server_oracle.conversation_text
    original_current_speaker = server_oracle.current_speaker
    try:
        server_oracle.configure_save_dir(str(tmp_path))
        server_oracle.conversation_text = ""
        server_oracle.current_speaker = None

        server_oracle.add_to_conversation("moshi", "hello", flush_file=True)
        server_oracle.add_to_conversation("moshi", "world", flush_file=True)
        server_oracle.add_to_conversation("user", "hi", flush_file=True)

        expected = "moshi: hello world \nuser: hi "
        assert server_oracle.get_conversation_snapshot() == expected
        assert (tmp_path / "conversation.txt").read_text() == expected
    finally:
        server_oracle.SAVE_DIR = original_save_dir
        server_oracle.conversation_text = original_conversation_text
        server_oracle.current_speaker = original_current_speaker


def test_active_deferred_logger_writes_session_logs(tmp_path: Path, monkeypatch) -> None:
    logger = DeferredSessionLogger(tmp_path, console_enabled=False)
    monkeypatch.setattr(server_oracle, "SAVE_DIR", tmp_path)
    monkeypatch.setattr(server_oracle, "SESSION_LOGGER", logger)
    monkeypatch.setattr(server_oracle, "conversation_text", "")
    monkeypatch.setattr(server_oracle, "current_speaker", None)

    logger.start_session()
    try:
        server_oracle._append_session_log("oracle_stream.txt", "first\n")
        server_oracle.add_to_conversation("user", "hello", flush_file=True)
    finally:
        summary = logger.finish_session()

    assert (tmp_path / "oracle_stream.txt").read_text() == "first\n"
    assert (tmp_path / "conversation.txt").read_text() == "user: hello "
    assert summary["deferred_log_error_count"] == 0


def test_llm_mux_prompt_includes_pending_user_text(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    original_conversation_text = server_oracle.conversation_text
    original_current_speaker = server_oracle.current_speaker
    try:
        server_oracle.conversation_text = "moshi: hello "
        server_oracle.current_speaker = "moshi"
        mux = server_oracle.LLMStreamMultiplexer(
            DummyServerState("I need help"),
            system_prompt="system",
            max_prompt_chars=1000,
        )

        messages, has_user_input = mux._build_messages_from_state()

        assert has_user_input is True
        assert messages == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "moshi: hello\nuser: I need help "},
        ]
    finally:
        server_oracle.conversation_text = original_conversation_text
        server_oracle.current_speaker = original_current_speaker


@pytest.mark.parametrize("max_prompt_chars", [0, -1])
def test_llm_mux_rejects_nonpositive_max_prompt_chars(max_prompt_chars: int) -> None:
    with pytest.raises(ValueError, match="max_prompt_chars must be positive"):
        server_oracle.LLMStreamMultiplexer(
            DummyServerState(),
            max_prompt_chars=max_prompt_chars,
        )


def test_llm_mux_strips_streamed_chunks_before_enqueue(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = DummyServerState()
    mux = server_oracle.LLMStreamMultiplexer(state, system_prompt="system")

    async def fake_stream():
        for text in (" \n ", "\n Hello \n"):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
            )

    async def fake_create(**_kwargs):
        return fake_stream()

    monkeypatch.setattr(mux.client.chat.completions, "create", fake_create)

    async def run_stream() -> None:
        session_id = mux.start_session(asyncio.get_running_loop())
        await mux._stream_single([], gen_id=1, session_id=session_id)

    asyncio.run(run_stream())

    assert state.llm_event_queue.get_nowait() == ("append", 1, "Hello")
    assert state.llm_event_queue.empty()


def test_llm_mux_adoption_does_not_roll_back_to_older_generation(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mux = server_oracle.LLMStreamMultiplexer(DummyServerState(), system_prompt="system")
    mux.adopted_gen = 4

    asyncio.run(mux._adopt_generation(3))

    assert mux.adopted_gen == 4


def test_llm_mux_ignores_stale_session_start_tasks(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mux = server_oracle.LLMStreamMultiplexer(DummyServerState("hello"), system_prompt="system")
    started = False

    async def fake_start_stream(*args, **kwargs) -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(mux, "_start_stream", fake_start_stream)

    async def run_stale_start() -> None:
        mux.start_session(asyncio.get_running_loop())
        stale_session_id = mux._session_id
        await mux.stop()
        await mux._maybe_start_new_stream(force=True, session_id=stale_session_id)

    asyncio.run(run_stale_start())

    assert started is False
