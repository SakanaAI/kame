from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
import threading
from typing import Any

from .client_utils import log


@dataclass(slots=True, frozen=True)
class _LogRecord:
    kind: str
    filename: str | None = None
    text: str | None = None
    level: str | None = None


class DeferredSessionLogger:
    """Write high-frequency session logs outside the asyncio event-loop thread."""

    _STOP = object()

    def __init__(
        self,
        output_dir: str | Path | None,
        *,
        console_enabled: bool = True,
        max_queue_size: int = 8192,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.console_enabled = bool(console_enabled)
        self.max_queue_size = int(max_queue_size)
        self._queue: Queue[_LogRecord | object] = Queue(maxsize=self.max_queue_size)
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._drop_counts: Counter[str] = Counter()
        self._processed_counts: Counter[str] = Counter()
        self._error_count = 0

    @property
    def active(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start_session(self) -> None:
        with self._state_lock:
            if self.active:
                raise RuntimeError("deferred logger session is already active")
            self._queue = Queue(maxsize=self.max_queue_size)
            self._drop_counts.clear()
            self._processed_counts.clear()
            self._error_count = 0
            if self.output_dir is None and not self.console_enabled:
                self._thread = None
                return
            self._thread = threading.Thread(
                target=self._run,
                name="kame-deferred-session-logger",
                daemon=True,
            )
            self._thread.start()

    def append_text(self, filename: str, text: str) -> None:
        if self.output_dir is None:
            return
        self._try_put(_LogRecord(kind="text", filename=filename, text=text))

    def replace_text(self, filename: str, text: str) -> None:
        if self.output_dir is None:
            return
        self._try_put(_LogRecord(kind="replace_text", filename=filename, text=text))

    def console(self, level: str, message: str) -> None:
        if not self.console_enabled:
            return
        self._try_put(_LogRecord(kind="console", level=level, text=message))

    def finish_session(self) -> dict[str, Any]:
        thread = self._thread
        if thread is not None:
            # Streaming has stopped, so blocking here cannot stall audio.
            self._queue.put(self._STOP)
            thread.join()
        with self._state_lock:
            self._thread = None
            return {
                "deferred_log_processed": dict(self._processed_counts),
                "deferred_log_dropped": dict(self._drop_counts),
                "deferred_log_error_count": self._error_count,
                "deferred_log_queue_size": self.max_queue_size,
            }

    def _try_put(self, record: _LogRecord) -> None:
        if not self.active:
            return
        try:
            self._queue.put_nowait(record)
        except Full:
            with self._state_lock:
                self._drop_counts[record.kind] += 1

    def _run(self) -> None:
        handles: dict[str, Any] = {}
        try:
            while True:
                try:
                    record = self._queue.get(timeout=0.5)
                except Empty:
                    continue
                if record is self._STOP:
                    break
                assert isinstance(record, _LogRecord)
                try:
                    self._write_record(record, handles)
                    with self._state_lock:
                        self._processed_counts[record.kind] += 1
                except Exception:
                    # Logging must never terminate or delay inference.
                    with self._state_lock:
                        self._error_count += 1
        finally:
            for handle in handles.values():
                handle.close()

    def _write_record(self, record: _LogRecord, handles: dict[str, Any]) -> None:
        if record.kind == "console":
            assert record.level is not None and record.text is not None
            log(record.level, record.text)
            return

        assert self.output_dir is not None and record.filename is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / record.filename
        if record.kind == "replace_text":
            assert record.text is not None
            path.write_text(record.text, encoding="utf-8")
            return

        handle = handles.get(record.filename)
        if handle is None:
            handle = path.open("a", encoding="utf-8")
            handles[record.filename] = handle

        if record.kind == "text":
            assert record.text is not None
            handle.write(record.text)
        else:
            raise ValueError(f"unknown deferred log record kind: {record.kind}")
