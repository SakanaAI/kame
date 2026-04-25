from kame import server_oracle


class DummyTokenizer:
    def __init__(self, underscore_id: int = 7):
        self.underscore_id = underscore_id

    def piece_to_id(self, piece: str) -> int:
        if piece == "▁":
            return self.underscore_id
        raise KeyError(piece)


def test_count_text_units_uses_words_for_english_and_characters_for_japanese() -> None:
    assert server_oracle.count_text_units("hello there", "en") == 2
    assert server_oracle.count_text_units("こんにちは 世界", "ja") == 7
    assert server_oracle.count_text_units("こんにちは　世界", "ja") == 7


def test_resolve_asr_language_uses_language_and_provider_defaults() -> None:
    assert server_oracle.resolve_asr_language("en", "openai", None) == "en"
    assert server_oracle.resolve_asr_language("en", "google", None) == "en-US"
    assert server_oracle.resolve_asr_language("ja", "openai", None) == "ja"
    assert server_oracle.resolve_asr_language("ja", "google", None) == "ja-JP"
    assert server_oracle.resolve_asr_language("ja", "google", "en-US") == "en-US"


def test_filter_oracle_token_ids_removes_japanese_leading_space_after_first_chunk() -> None:
    tokenizer = DummyTokenizer(underscore_id=7)

    token_ids, is_first_chunk = server_oracle.filter_oracle_token_ids_for_language(
        [7, 10],
        tokenizer,  # type: ignore[arg-type]
        "ja",
        is_first_chunk=True,
    )
    assert token_ids == [7, 10]
    assert is_first_chunk is False

    token_ids, is_first_chunk = server_oracle.filter_oracle_token_ids_for_language(
        [7, 11],
        tokenizer,  # type: ignore[arg-type]
        "ja",
        is_first_chunk=False,
    )
    assert token_ids == [11]
    assert is_first_chunk is False


def test_filter_oracle_token_ids_leaves_english_unchanged() -> None:
    token_ids, is_first_chunk = server_oracle.filter_oracle_token_ids_for_language(
        [7, 10],
        DummyTokenizer(),  # type: ignore[arg-type]
        "en",
        is_first_chunk=True,
    )
    assert token_ids == [7, 10]
    assert is_first_chunk is True
