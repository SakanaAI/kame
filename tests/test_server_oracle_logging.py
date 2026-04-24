import os
import subprocess
import sys
from pathlib import Path

from kame import server_oracle


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
