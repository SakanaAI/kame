import pytest

from kame.asr_processors import (
    DEFAULT_GOOGLE_ASR_LANGUAGE,
    DEFAULT_OPENAI_ASR_LANGUAGE,
    DEFAULT_OPENAI_ASR_MODEL,
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
