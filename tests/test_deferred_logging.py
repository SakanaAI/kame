from pathlib import Path

from kame.deferred_logging import DeferredSessionLogger


def test_deferred_logger_preserves_session_log_formats(tmp_path: Path) -> None:
    logger = DeferredSessionLogger(tmp_path, console_enabled=False)
    logger.start_session()

    logger.append_text("words.txt", "first\n")
    logger.append_text("words.txt", "second\n")
    logger.replace_text("conversation.txt", "user: hello ")
    summary = logger.finish_session()

    assert (tmp_path / "words.txt").read_text() == "first\nsecond\n"
    assert (tmp_path / "conversation.txt").read_text() == "user: hello "
    assert summary["deferred_log_dropped"] == {}
    assert summary["deferred_log_error_count"] == 0


def test_disabled_deferred_logger_is_a_noop(tmp_path: Path) -> None:
    logger = DeferredSessionLogger(None, console_enabled=False)
    logger.start_session()
    logger.append_text("words.txt", "ignored\n")

    summary = logger.finish_session()

    assert list(tmp_path.iterdir()) == []
    assert summary["deferred_log_processed"] == {}
