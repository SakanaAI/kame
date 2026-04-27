import asyncio

import numpy as np
import pytest

from kame.asr_processors import (
    DEFAULT_GOOGLE_ASR_LANGUAGE,
    DEFAULT_OPENAI_LOCAL_VAD_ENERGY_THRESHOLD,
    DEFAULT_OPENAI_ASR_LANGUAGE,
    DEFAULT_OPENAI_ASR_MODEL,
    DEFAULT_OPENAI_MANUAL_COMMIT_INTERVAL_MS,
    DEFAULT_OPENAI_MANUAL_MIN_COMMIT_MS,
    DEFAULT_OPENAI_MANUAL_PREFIX_PADDING_MS,
    DEFAULT_OPENAI_MANUAL_SILENCE_DURATION_MS,
    DEFAULT_OPENAI_MANUAL_START_DEBOUNCE_MS,
    GoogleASRProcessor,
    OpenAIRealtimeASRProcessor,
    create_asr_processor,
    require_initialized_asr,
)


class DisabledASRProcessor:
    asr_enabled: bool = False
    init_error: str | None = "missing credentials"
    running: bool = False

    def register_callbacks(self, on_partial, on_final):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass

    def process_audio(self, pcm_data):
        pass


def test_create_openai_asr_processor_uses_openai_provider(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    processor = create_asr_processor("openai", sample_rate=24000)

    assert isinstance(processor, OpenAIRealtimeASRProcessor)
    assert processor.asr_enabled
    assert processor.model == DEFAULT_OPENAI_ASR_MODEL
    assert processor.language == DEFAULT_OPENAI_ASR_LANGUAGE
    assert processor.manual_commit
    assert processor.manual_commit_interval_ms == DEFAULT_OPENAI_MANUAL_COMMIT_INTERVAL_MS
    assert processor.manual_min_commit_ms == DEFAULT_OPENAI_MANUAL_MIN_COMMIT_MS
    assert processor.manual_silence_duration_ms == DEFAULT_OPENAI_MANUAL_SILENCE_DURATION_MS
    assert processor.manual_prefix_padding_ms == DEFAULT_OPENAI_MANUAL_PREFIX_PADDING_MS
    assert processor.manual_start_debounce_ms == DEFAULT_OPENAI_MANUAL_START_DEBOUNCE_MS
    assert processor.local_vad_energy_threshold == DEFAULT_OPENAI_LOCAL_VAD_ENERGY_THRESHOLD


def test_create_openai_manual_commit_processor_disables_server_vad(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    processor = create_asr_processor("openai", sample_rate=24000, manual_commit=True)

    assert isinstance(processor, OpenAIRealtimeASRProcessor)
    assert processor.manual_commit
    assert processor._session_update_event()["session"]["audio"]["input"]["turn_detection"] is None


def test_openai_manual_commit_local_vad_queues_periodic_and_final_commits(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    processor = create_asr_processor("openai", sample_rate=24000, manual_commit=True)
    assert isinstance(processor, OpenAIRealtimeASRProcessor)
    processor.running = True

    frame = 1920  # 80 ms at 24 kHz.
    silence = np.zeros(frame, dtype=np.float32)
    speech = np.full(frame, 0.02, dtype=np.float32)

    for _ in range(3):
        processor.process_audio(silence)
    for _ in range(14):
        processor.process_audio(speech)
    for _ in range(6):
        processor.process_audio(silence)

    commands = []
    while True:
        try:
            command = processor.audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if command is not None:
            commands.append(command[0])

    assert commands.count("commit") == 2
    assert commands[0] == "append"
    assert processor._manual_pending_commit_acks == 2
    assert processor.stats["manual_speech_starts"] == 1
    assert processor.stats["manual_speech_ends"] == 1


def test_openai_manual_commit_start_debounce_ignores_short_candidate(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    processor = create_asr_processor(
        "openai",
        sample_rate=24000,
        manual_commit=True,
        manual_start_debounce_ms=160,
        local_vad_energy_threshold=0.01,
    )
    assert isinstance(processor, OpenAIRealtimeASRProcessor)
    processor.running = True

    frame = 1920  # 80 ms at 24 kHz.
    silence = np.zeros(frame, dtype=np.float32)
    speech = np.full(frame, 0.02, dtype=np.float32)

    processor.process_audio(speech)
    processor.process_audio(silence)

    assert processor.stats["manual_speech_starts"] == 0
    assert processor.audio_queue.empty()


def test_openai_manual_commit_start_debounce_retains_candidate_audio(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    processor = create_asr_processor(
        "openai",
        sample_rate=24000,
        manual_commit=True,
        manual_prefix_padding_ms=240,
        manual_start_debounce_ms=160,
        local_vad_energy_threshold=0.01,
    )
    assert isinstance(processor, OpenAIRealtimeASRProcessor)
    processor.running = True

    frame = 1920  # 80 ms at 24 kHz.
    silence = np.zeros(frame, dtype=np.float32)
    speech = np.full(frame, 0.02, dtype=np.float32)

    for _ in range(3):
        processor.process_audio(silence)
    processor.process_audio(speech)
    assert processor.stats["manual_speech_starts"] == 0
    processor.process_audio(speech)

    command = processor.audio_queue.get_nowait()
    assert command is not None
    action, payload = command
    assert action == "append"
    assert isinstance(payload, bytes)
    assert len(payload) == frame * 5 * 2
    assert processor.stats["manual_speech_starts"] == 1
    assert processor._manual_buffer_ms == pytest.approx(400.0)


def test_openai_manual_commit_aggregates_chunks_before_final(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    processor = create_asr_processor("openai", sample_rate=24000, manual_commit=True)
    assert isinstance(processor, OpenAIRealtimeASRProcessor)

    partials = []
    finals = []
    processor.register_callbacks(partials.append, finals.append)
    utterance = processor._start_manual_utterance()
    utterance.pending_commit_acks = 2
    utterance.finalize_requested = True
    processor._manual_pending_commit_acks = 2
    processor._manual_pending_commit_utterance_ids.extend([utterance.utterance_id, utterance.utterance_id])

    processor._handle_realtime_event({"type": "input_audio_buffer.committed", "item_id": "item_1"})
    processor._handle_realtime_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "hello",
        }
    )
    processor._handle_realtime_event({"type": "input_audio_buffer.committed", "item_id": "item_2"})

    assert partials == ["hello"]
    assert finals == []

    processor._handle_realtime_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_2",
            "transcript": "world",
        }
    )

    assert partials == ["hello", "hello world"]
    assert finals == ["hello world"]


def test_openai_manual_commit_accepts_late_transcript_after_next_utterance_starts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    processor = create_asr_processor("openai", sample_rate=24000, manual_commit=True)
    assert isinstance(processor, OpenAIRealtimeASRProcessor)

    finals = []
    processor.register_callbacks(lambda text: None, finals.append)

    first = processor._start_manual_utterance()
    first.pending_commit_acks = 1
    first.finalize_requested = True
    processor._manual_pending_commit_acks = 1
    processor._manual_pending_commit_utterance_ids.append(first.utterance_id)
    processor._handle_realtime_event({"type": "input_audio_buffer.committed", "item_id": "item_1"})

    second = processor._start_manual_utterance()

    processor._handle_realtime_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "late hello",
        }
    )

    assert finals == ["late hello"]
    assert processor._manual_item_to_utterance_id["item_1"] == first.utterance_id
    assert processor._manual_current_utterance_id == second.utterance_id


def test_openai_manual_commit_handles_transcript_before_committed_ack(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    processor = create_asr_processor("openai", sample_rate=24000, manual_commit=True)
    assert isinstance(processor, OpenAIRealtimeASRProcessor)

    finals = []
    processor.register_callbacks(lambda text: None, finals.append)
    utterance = processor._start_manual_utterance()
    utterance.pending_commit_acks = 1
    utterance.finalize_requested = True
    processor._manual_pending_commit_acks = 1
    processor._manual_pending_commit_utterance_ids.append(utterance.utterance_id)

    processor._handle_realtime_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_early",
            "transcript": "early hello",
        }
    )

    assert finals == []

    processor._handle_realtime_event({"type": "input_audio_buffer.committed", "item_id": "item_early"})

    assert finals == ["early hello"]
    assert "item_early" not in processor._manual_orphan_completed_items


def test_openai_asr_processor_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    processor = create_asr_processor("openai", sample_rate=24000)

    assert isinstance(processor, OpenAIRealtimeASRProcessor)
    assert not processor.asr_enabled
    assert processor.init_error == "OPENAI_API_KEY environment variable is not set."


def test_create_google_asr_processor_uses_google_provider_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    processor = create_asr_processor("google", sample_rate=24000)

    assert isinstance(processor, GoogleASRProcessor)
    assert not processor.asr_enabled
    assert processor.language_code == DEFAULT_GOOGLE_ASR_LANGUAGE
    assert processor.init_error == "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."


def test_require_initialized_asr_allows_disabled_asr() -> None:
    require_initialized_asr(enable_asr=False, asr_processor=None)


def test_require_initialized_asr_raises_with_processor_init_error() -> None:
    with pytest.raises(RuntimeError, match="missing credentials"):
        require_initialized_asr(enable_asr=True, asr_processor=DisabledASRProcessor())
