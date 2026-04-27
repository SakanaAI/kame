import asyncio
import base64
from collections import deque
from dataclasses import dataclass, field
import json
import os
import queue
import time
from typing import Any, Literal, Protocol

import aiohttp
from google.cloud import speech
import numpy as np

from .client_utils import log

ASRProvider = Literal["openai", "google"]
DEFAULT_ASR_PROVIDER: ASRProvider = "openai"
DEFAULT_GOOGLE_ASR_LANGUAGE = "en-US"
DEFAULT_OPENAI_ASR_MODEL = "gpt-4o-transcribe"
DEFAULT_OPENAI_ASR_LANGUAGE = "en"
DEFAULT_OPENAI_MANUAL_COMMIT = True
DEFAULT_OPENAI_MANUAL_COMMIT_INTERVAL_MS = 1000
DEFAULT_OPENAI_MANUAL_MIN_COMMIT_MS = 480
DEFAULT_OPENAI_MANUAL_SILENCE_DURATION_MS = 480
DEFAULT_OPENAI_MANUAL_PREFIX_PADDING_MS = 320
DEFAULT_OPENAI_MANUAL_START_DEBOUNCE_MS = 160
DEFAULT_OPENAI_LOCAL_VAD_ENERGY_THRESHOLD = 0.010
OPENAI_REALTIME_TRANSCRIPTION_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


@dataclass
class _ManualASRUtterance:
    utterance_id: int
    pending_commit_acks: int = 0
    item_order: list[str] = field(default_factory=list)
    partial_transcripts: dict[str, str] = field(default_factory=dict)
    completed_items: set[str] = field(default_factory=set)
    completed_transcripts: dict[str, str] = field(default_factory=dict)
    failed_items: set[str] = field(default_factory=set)
    finalize_requested: bool = False
    final_emitted: bool = False
    created_at: float = field(default_factory=time.monotonic)
    final_emitted_at: float | None = None


class ASRProcessor(Protocol):
    asr_enabled: bool
    init_error: str | None
    running: bool

    def register_callbacks(self, on_partial, on_final): ...

    async def start(self): ...

    async def stop(self): ...

    def process_audio(self, pcm_data): ...


def _linear_resample_int16(x_int16: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """Very lightweight linear resample to reduce aliasing vs index stepping."""
    if src_hz == dst_hz:
        return x_int16
    n_src = len(x_int16)
    n_dst = int(n_src * dst_hz / src_hz)
    if n_dst <= 0:
        return np.zeros(0, dtype=np.int16)
    src_idx = np.arange(n_src, dtype=np.float64)
    dst_pos = np.linspace(0, n_src - 1, n_dst, endpoint=True)
    y = np.interp(dst_pos, src_idx, x_int16.astype(np.float64))
    y = np.clip(y, -32768, 32767).astype(np.int16)
    return y


class GoogleASRProcessor:
    """Async ASR with Google Speech-to-Text. Produces partial (pending) and final commits via callbacks.
    Audio is pushed via process_audio(...) from the main audio loop thread.
    """

    def __init__(self, sample_rate=24000, language_code: str = DEFAULT_GOOGLE_ASR_LANGUAGE):
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000  # Google Speech API requirement
        self.language_code = language_code

        self.audio_buffer: queue.Queue[bytes | None] = queue.Queue(maxsize=100)  # thread-safe
        self.running = False
        self.asr_task = None

        # Stats
        self.stats = {"words_detected": 0, "final_transcripts": 0, "buffer_drops": 0, "reconnections": 0}

        # Google Speech
        self.asr_enabled = False
        self.init_error: str | None = None
        self.speech_client = None
        self.config = None
        self.streaming_config = None

        # Callbacks (set by ServerState)
        self._on_partial = None
        self._on_final = None

        # Internals
        self.stream_start_time = None
        self.last_partial_text = ""

        self._initialize_speech_client()

    def register_callbacks(self, on_partial, on_final):
        """Both are plain callables; they will schedule async work in the server loop."""
        self._on_partial = on_partial
        self._on_final = on_final

    def _initialize_speech_client(self):
        try:
            if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
                self.init_error = "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."
                return

            self.speech_client = speech.SpeechClient()
            self.config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.target_sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=False,
                enable_word_time_offsets=False,
                enable_word_confidence=False,
                use_enhanced=True,
                metadata=speech.RecognitionMetadata(
                    interaction_type=speech.RecognitionMetadata.InteractionType.VOICE_SEARCH,
                    microphone_distance=speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD,
                    recording_device_type=speech.RecognitionMetadata.RecordingDeviceType.PC,
                ),
            )
            self.streaming_config = speech.StreamingRecognitionConfig(
                config=self.config,
                interim_results=True,
                single_utterance=False,
            )
            self.asr_enabled = True
            self.init_error = None
            log("info", f"Async ASR processor initialized (language: {self.language_code})")
        except Exception as e:
            self.init_error = str(e)
            log("warning", f"ASR initialization failed: {e}")

    async def start(self):
        if self.asr_enabled and not self.running:
            self.running = True
            self.asr_task = asyncio.create_task(self._run_asr_streaming())
            log("info", "Async ASR streaming started")

    async def stop(self):
        if self.running:
            self.running = False
            try:
                self.audio_buffer.put(None, block=False)  # signal end
            except queue.Full:
                # If the buffer is already full, we can rely on task cancellation below
                # to stop the ASR loop; the explicit sentinel is not strictly required.
                pass

            if self.asr_task:
                self.asr_task.cancel()
                try:
                    await self.asr_task
                except asyncio.CancelledError:
                    # Task cancellation is expected during cleanup; safe to ignore.
                    pass
            log("info", f"Async ASR streaming stopped. Stats: {self.stats}")

    def process_audio(self, pcm_data):
        """Accept float32 mono [-1,1] or int16 numpy array; pushes 16k int16 bytes into a thread-safe buffer."""
        if not self.asr_enabled or not self.running:
            return
        try:
            # Convert to int16
            if isinstance(pcm_data, np.ndarray):
                if pcm_data.dtype == np.float32:
                    pcm_data = np.clip(pcm_data, -1.0, 1.0)
                    pcm_16bit = (pcm_data * 32767).astype(np.int16)
                elif pcm_data.dtype == np.int16:
                    pcm_16bit = pcm_data
                else:
                    pcm_16bit = pcm_data.astype(np.int16)
            else:
                pcm_float = pcm_data.numpy() if hasattr(pcm_data, "numpy") else np.asarray(pcm_data, dtype=np.float32)
                pcm_float = np.clip(pcm_float, -1.0, 1.0)
                pcm_16bit = (pcm_float * 32767).astype(np.int16)

            # Resample to 16 kHz linearly
            pcm_16k = _linear_resample_int16(pcm_16bit, self.sample_rate, self.target_sample_rate)

            try:
                self.audio_buffer.put(pcm_16k.tobytes(), block=False)
            except queue.Full:
                self.stats["buffer_drops"] += 1
                try:
                    _ = self.audio_buffer.get_nowait()
                    self.audio_buffer.put(pcm_16k.tobytes(), block=False)
                except Exception:
                    # Best-effort buffer swap failed; drop this chunk silently.
                    # This is rare and losing one audio chunk is acceptable.
                    pass
        except Exception as e:
            log("error", f"Error processing audio: {e}")

    async def _run_asr_streaming(self):
        retry_count = 0
        max_retries = 5

        while self.running:
            try:
                self.stream_start_time = time.time()
                # Run Google streaming in a worker thread (blocking)
                await asyncio.to_thread(self._run_speech_streaming)
                retry_count = 0
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                log("info", "ASR streaming cancelled")
                raise
            except Exception as e:
                if self.running:
                    retry_count += 1
                    self.stats["reconnections"] += 1
                    log("error", f"ASR streaming error (retry {retry_count}/{max_retries}): {e}")
                    if retry_count >= max_retries:
                        log("warning", "Max retries reached, waiting before reset...")
                        await asyncio.sleep(30)
                        retry_count = 0
                    else:
                        await asyncio.sleep(min(retry_count * 0.5, 2.0))

    def _run_speech_streaming(self):
        try:

            def audio_generator():
                # 10ms at 16kHz, 2 bytes per sample -> 160 samples -> 320 bytes
                min_chunk_size_bytes = 320
                last_data_time = time.time()
                while self.running:
                    chunks = []
                    total_size = 0
                    try:
                        chunk = self.audio_buffer.get(timeout=0.02)
                        if chunk is None:
                            return
                        chunks.append(chunk)
                        total_size += len(chunk)
                        last_data_time = time.time()
                    except queue.Empty:
                        if time.time() - last_data_time > 5.0:
                            log("warning", "No audio data for 5s, sending short silence")
                            yield b"\x00" * 320  # ~10ms silence
                            last_data_time = time.time()
                        continue

                    # Coalesce up to ~10ms
                    deadline = time.time() + 0.01
                    while total_size < min_chunk_size_bytes and time.time() < deadline:
                        try:
                            chunk = self.audio_buffer.get(timeout=0.005)
                            if chunk is None:
                                return
                            chunks.append(chunk)
                            total_size += len(chunk)
                        except queue.Empty:
                            break

                    # Drain any residual without blocking
                    while True:
                        try:
                            chunk = self.audio_buffer.get_nowait()
                            if chunk is None:
                                return
                            chunks.append(chunk)
                        except queue.Empty:
                            break

                    if chunks:
                        yield b"".join(chunks)

            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator())
            if self.speech_client is None:
                return
            responses = self.speech_client.streaming_recognize(self.streaming_config, requests)
            self._process_responses(responses)
        except Exception:
            if self.running:
                raise

    def _process_responses(self, responses):
        for response in responses:
            if not self.running:
                break
            if not response.results:
                continue

            for result in response.results:
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                if not transcript:
                    continue

                # Emit partial (debounced: only if changed)
                if not result.is_final:
                    if transcript != self.last_partial_text:
                        self.last_partial_text = transcript
                        if self._on_partial:
                            try:
                                self._on_partial(transcript)
                            except Exception:
                                # Callback errors should not stop ASR streaming; ignore.
                                pass
                    continue

                # Final result
                self.last_partial_text = ""
                self.stats["final_transcripts"] += 1
                if self._on_final:
                    try:
                        self._on_final(transcript)
                    except Exception:
                        # Callback errors should not stop ASR streaming; ignore.
                        pass


class OpenAIRealtimeASRProcessor:
    """Async ASR with OpenAI Realtime transcription.

    The public surface intentionally mirrors the ASR processor used by
    server_oracle.py, so ServerState can keep feeding PCM through process_audio().
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        model: str = DEFAULT_OPENAI_ASR_MODEL,
        language: str = DEFAULT_OPENAI_ASR_LANGUAGE,
        max_queue_size: int = 100,
        vad_threshold: float = 0.5,
        vad_prefix_padding_ms: int = 240,
        vad_silence_duration_ms: int = 240,
        manual_commit: bool = DEFAULT_OPENAI_MANUAL_COMMIT,
        manual_commit_interval_ms: int = DEFAULT_OPENAI_MANUAL_COMMIT_INTERVAL_MS,
        manual_min_commit_ms: int = DEFAULT_OPENAI_MANUAL_MIN_COMMIT_MS,
        manual_silence_duration_ms: int = DEFAULT_OPENAI_MANUAL_SILENCE_DURATION_MS,
        manual_prefix_padding_ms: int = DEFAULT_OPENAI_MANUAL_PREFIX_PADDING_MS,
        manual_start_debounce_ms: int = DEFAULT_OPENAI_MANUAL_START_DEBOUNCE_MS,
        local_vad_energy_threshold: float = DEFAULT_OPENAI_LOCAL_VAD_ENERGY_THRESHOLD,
    ):
        self.sample_rate = sample_rate
        self.target_sample_rate = 24000  # Realtime transcription supports 24 kHz mono PCM input.
        self.model = model
        self.language = language
        self.vad_threshold = vad_threshold
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.manual_commit = manual_commit
        self.manual_commit_interval_ms = manual_commit_interval_ms
        self.manual_min_commit_ms = manual_min_commit_ms
        self.manual_silence_duration_ms = manual_silence_duration_ms
        self.manual_prefix_padding_ms = manual_prefix_padding_ms
        self.manual_start_debounce_ms = max(0, manual_start_debounce_ms)
        self.local_vad_energy_threshold = local_vad_energy_threshold

        self.audio_queue: asyncio.Queue[tuple[str, bytes | str | None] | None] = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
        self.asr_task: asyncio.Task | None = None

        self.stats = {
            "partial_events": 0,
            "final_transcripts": 0,
            "buffer_drops": 0,
            "reconnections": 0,
            "sent_audio_chunks": 0,
            "sent_commit_events": 0,
            "sent_clear_events": 0,
            "manual_speech_starts": 0,
            "manual_speech_ends": 0,
        }

        self.asr_enabled = False
        self.init_error: str | None = None
        self.api_key = os.environ.get("OPENAI_API_KEY")

        self._on_partial = None
        self._on_final = None

        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._partial_transcripts: dict[str, str] = {}
        self.last_partial_text = ""

        self._manual_prefix_chunks: list[tuple[bytes, float]] = []
        self._manual_in_speech = False
        self._manual_start_candidate_ms = 0.0
        self._manual_silence_ms = 0.0
        self._manual_buffer_ms = 0.0
        self._manual_buffer_has_speech = False
        self._manual_pending_commit_acks = 0
        self._manual_commit_seq = 0
        self._manual_utterance_seq = 0
        self._manual_current_utterance_id: int | None = None
        self._manual_utterances: dict[int, _ManualASRUtterance] = {}
        self._manual_pending_commit_utterance_ids: deque[int] = deque()
        self._manual_item_to_utterance_id: dict[str, int] = {}
        self._manual_orphan_partials: dict[str, str] = {}
        self._manual_orphan_completed_items: set[str] = set()
        self._manual_orphan_completed_transcripts: dict[str, str] = {}
        self._manual_orphan_failed_items: set[str] = set()
        self._manual_orphan_item_seen_at: dict[str, float] = {}
        self._manual_late_event_retention_ms = 5000

        self._initialize_openai_client()

    def register_callbacks(self, on_partial, on_final):
        """Both are plain callables; they will schedule async work in the server loop."""
        self._on_partial = on_partial
        self._on_final = on_final

    def _initialize_openai_client(self):
        if not self.api_key:
            self.init_error = "OPENAI_API_KEY environment variable is not set."
            return

        self.asr_enabled = True
        self.init_error = None
        mode = "manual_commit" if self.manual_commit else "server_vad"
        log(
            "info",
            f"OpenAI Realtime ASR initialized (model: {self.model}, language: {self.language}, mode: {mode})",
        )

    async def start(self):
        if self.asr_enabled and not self.running:
            self.running = True
            self._reset_transcription_state(clear_prefix=True)
            self.asr_task = asyncio.create_task(self._run_realtime_transcription())
            log("info", "OpenAI Realtime ASR streaming started")

    async def stop(self):
        if self.running:
            self.running = False
            try:
                self.audio_queue.put_nowait(None)
            except asyncio.QueueFull:
                self._drop_oldest_audio_chunk()
                try:
                    self.audio_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

            if self.ws is not None:
                await self.ws.close()

            if self.asr_task:
                self.asr_task.cancel()
                try:
                    await self.asr_task
                except asyncio.CancelledError:
                    # Task cancellation is expected during cleanup; safe to ignore.
                    pass
                self.asr_task = None
            log("info", f"OpenAI Realtime ASR streaming stopped. Stats: {self.stats}")

    def process_audio(self, pcm_data):
        """Accept float32 mono [-1,1] or int16 numpy array; queue 24 kHz PCM16 bytes for ASR."""
        if not self.asr_enabled or not self.running:
            return
        try:
            # Convert to int16
            if isinstance(pcm_data, np.ndarray):
                if pcm_data.dtype == np.float32:
                    pcm_data = np.clip(pcm_data, -1.0, 1.0)
                    pcm_16bit = (pcm_data * 32767).astype(np.int16)
                elif pcm_data.dtype == np.int16:
                    pcm_16bit = pcm_data
                else:
                    pcm_16bit = pcm_data.astype(np.int16)
            else:
                pcm_float = pcm_data.numpy() if hasattr(pcm_data, "numpy") else np.asarray(pcm_data, dtype=np.float32)
                pcm_float = np.clip(pcm_float, -1.0, 1.0)
                pcm_16bit = (pcm_float * 32767).astype(np.int16)

            pcm_24k = _linear_resample_int16(pcm_16bit, self.sample_rate, self.target_sample_rate)
            pcm_bytes = pcm_24k.tobytes()
            if not self.manual_commit:
                self._queue_append_audio(pcm_bytes)
                return

            self._process_audio_manual(pcm_24k, pcm_bytes)
        except Exception as e:
            log("error", f"Error processing audio: {e}")

    def _queue_append_audio(self, pcm_bytes: bytes) -> bool:
        try:
            self.audio_queue.put_nowait(("append", pcm_bytes))
            return True
        except asyncio.QueueFull:
            self.stats["buffer_drops"] += 1
            self._drop_oldest_audio_chunk()
            try:
                self.audio_queue.put_nowait(("append", pcm_bytes))
                return True
            except asyncio.QueueFull:
                # If the sender is still behind, drop the new chunk too and keep
                # the main audio loop moving.
                return False

    def _queue_commit(self, reason: str, utterance_id: int, buffered_ms: float) -> bool:
        self._manual_commit_seq += 1
        event_id = f"kame_manual_commit_{self._manual_commit_seq}"
        try:
            self.audio_queue.put_nowait(("commit", event_id))
        except asyncio.QueueFull:
            self._drop_oldest_audio_chunk()
            try:
                self.audio_queue.put_nowait(("commit", event_id))
            except asyncio.QueueFull:
                self.stats["buffer_drops"] += 1
                log(
                    "warning",
                    f"OpenAI manual ASR commit dropped ({reason}, utterance_id={utterance_id}); audio queue is full.",
                )
                return False
        log(
            "info",
            f"OpenAI manual ASR commit queued "
            f"({reason}, utterance_id={utterance_id}, buffered_ms={buffered_ms:.0f}, event_id={event_id})",
        )
        return True

    def _queue_clear(self) -> None:
        try:
            self.audio_queue.put_nowait(("clear", None))
        except asyncio.QueueFull:
            self._drop_oldest_audio_chunk()
            try:
                self.audio_queue.put_nowait(("clear", None))
            except asyncio.QueueFull:
                self.stats["buffer_drops"] += 1

    def _drop_oldest_audio_chunk(self) -> None:
        retained: list[tuple[str, bytes | str | None] | None] = []
        dropped = False
        while True:
            try:
                command = self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if command is not None and command[0] == "append" and not dropped:
                dropped = True
                continue
            retained.append(command)

        for command in retained:
            try:
                self.audio_queue.put_nowait(command)
            except asyncio.QueueFull:
                break

    def _clear_audio_queue(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _reset_transcription_state(self, clear_prefix: bool) -> None:
        self._partial_transcripts.clear()
        self.last_partial_text = ""
        self._manual_in_speech = False
        self._manual_start_candidate_ms = 0.0
        self._manual_silence_ms = 0.0
        self._manual_buffer_ms = 0.0
        self._manual_buffer_has_speech = False
        self._manual_pending_commit_acks = 0
        self._manual_current_utterance_id = None
        self._manual_utterances.clear()
        self._manual_pending_commit_utterance_ids.clear()
        self._manual_item_to_utterance_id.clear()
        self._manual_orphan_partials.clear()
        self._manual_orphan_completed_items.clear()
        self._manual_orphan_completed_transcripts.clear()
        self._manual_orphan_failed_items.clear()
        self._manual_orphan_item_seen_at.clear()
        if clear_prefix:
            self._manual_prefix_chunks.clear()

    def _start_manual_utterance(self) -> _ManualASRUtterance:
        self._manual_utterance_seq += 1
        utterance = _ManualASRUtterance(utterance_id=self._manual_utterance_seq)
        self._manual_utterances[utterance.utterance_id] = utterance
        self._manual_current_utterance_id = utterance.utterance_id
        self._prune_manual_utterances()
        return utterance

    def _current_manual_utterance(self) -> _ManualASRUtterance:
        if self._manual_current_utterance_id is not None:
            utterance = self._manual_utterances.get(self._manual_current_utterance_id)
            if utterance is not None and not utterance.finalize_requested:
                return utterance
        return self._start_manual_utterance()

    def _prune_manual_utterances(self) -> None:
        now = time.monotonic()
        retention_seconds = self._manual_late_event_retention_ms / 1000.0
        stale_utterance_ids = [
            utterance_id
            for utterance_id, utterance in self._manual_utterances.items()
            if utterance.final_emitted
            and utterance.final_emitted_at is not None
            and now - utterance.final_emitted_at > retention_seconds
        ]
        for utterance_id in stale_utterance_ids:
            utterance = self._manual_utterances.pop(utterance_id)
            for item_id in utterance.item_order:
                if self._manual_item_to_utterance_id.get(item_id) == utterance_id:
                    self._manual_item_to_utterance_id.pop(item_id, None)

        stale_orphan_ids = [
            item_id
            for item_id, seen_at in self._manual_orphan_item_seen_at.items()
            if now - seen_at > retention_seconds
        ]
        for item_id in stale_orphan_ids:
            self._clear_manual_orphan_item(item_id)

    def _process_audio_manual(self, pcm_24k: np.ndarray, pcm_bytes: bytes) -> None:
        duration_ms = self._pcm_duration_ms(pcm_24k)
        if duration_ms <= 0:
            return

        is_speech = self._pcm_energy(pcm_24k) >= self.local_vad_energy_threshold

        if not self._manual_in_speech:
            if not is_speech:
                self._manual_start_candidate_ms = 0.0
                self._add_manual_prefix_chunk(pcm_bytes, duration_ms)
                return

            if self.manual_start_debounce_ms > 0:
                self._manual_start_candidate_ms += duration_ms
                self._add_manual_prefix_chunk(pcm_bytes, duration_ms)
                if self._manual_start_candidate_ms < self.manual_start_debounce_ms:
                    return
                self._manual_start_candidate_ms = 0.0
                utterance = self._start_manual_utterance()
                self._manual_in_speech = True
                self._manual_silence_ms = 0.0
                self.stats["manual_speech_starts"] += 1
                log("info", f"OpenAI manual ASR local speech started (utterance_id={utterance.utterance_id})")

                prefix_bytes, prefix_ms = self._consume_manual_prefix()
                if prefix_bytes:
                    self._append_manual_audio(prefix_bytes, prefix_ms, has_speech=True)
                self._maybe_commit_manual_interval()
                return

            utterance = self._start_manual_utterance()
            self._manual_in_speech = True
            self._manual_start_candidate_ms = 0.0
            self._manual_silence_ms = 0.0
            self.stats["manual_speech_starts"] += 1
            log("info", f"OpenAI manual ASR local speech started (utterance_id={utterance.utterance_id})")

            prefix_bytes, prefix_ms = self._consume_manual_prefix()
            if prefix_bytes:
                self._append_manual_audio(prefix_bytes, prefix_ms, has_speech=False)
            self._append_manual_audio(pcm_bytes, duration_ms, has_speech=True)
            self._maybe_commit_manual_interval()
            return

        self._append_manual_audio(pcm_bytes, duration_ms, has_speech=is_speech)
        if is_speech:
            self._manual_silence_ms = 0.0
            self._maybe_commit_manual_interval()
            return

        self._manual_silence_ms += duration_ms
        if self._manual_silence_ms >= self.manual_silence_duration_ms:
            self._finish_manual_utterance()
            self._add_manual_prefix_chunk(pcm_bytes, duration_ms)

    def _append_manual_audio(self, pcm_bytes: bytes, duration_ms: float, has_speech: bool) -> None:
        if not self._queue_append_audio(pcm_bytes):
            return
        self._manual_buffer_ms += duration_ms
        self._manual_buffer_has_speech = self._manual_buffer_has_speech or has_speech

    def _maybe_commit_manual_interval(self) -> None:
        if self._manual_buffer_ms < self.manual_commit_interval_ms:
            return
        if self._manual_buffer_ms < self.manual_min_commit_ms or not self._manual_buffer_has_speech:
            return
        self._commit_manual_buffer("interval")

    def _finish_manual_utterance(self) -> None:
        if self._manual_current_utterance_id is None:
            return
        utterance = self._manual_utterances.get(self._manual_current_utterance_id)
        if utterance is None:
            return

        self._manual_in_speech = False
        self._manual_start_candidate_ms = 0.0
        self._manual_silence_ms = 0.0
        utterance.finalize_requested = True
        self.stats["manual_speech_ends"] += 1
        log("info", f"OpenAI manual ASR local speech ended (utterance_id={utterance.utterance_id})")

        if self._manual_buffer_ms > 0 and self._manual_buffer_has_speech:
            self._commit_manual_buffer("speech_end")
        elif self._manual_buffer_ms > 0:
            self._queue_clear()
            self._manual_buffer_ms = 0.0
            self._manual_buffer_has_speech = False
        self._manual_current_utterance_id = None
        self._try_emit_manual_final(utterance)

    def _commit_manual_buffer(self, reason: str) -> None:
        if self._manual_buffer_ms <= 0:
            return
        current_utterance_id = self._manual_current_utterance_id
        if current_utterance_id is None:
            utterance = self._current_manual_utterance()
        else:
            existing_utterance = self._manual_utterances.get(current_utterance_id)
            if existing_utterance is None:
                utterance = self._current_manual_utterance()
            else:
                utterance = existing_utterance
        buffered_ms = self._manual_buffer_ms
        if self._queue_commit(reason, utterance.utterance_id, buffered_ms):
            utterance.pending_commit_acks += 1
            self._manual_pending_commit_utterance_ids.append(utterance.utterance_id)
            self._manual_pending_commit_acks += 1
        self._manual_buffer_ms = 0.0
        self._manual_buffer_has_speech = False

    def _add_manual_prefix_chunk(self, pcm_bytes: bytes, duration_ms: float) -> None:
        retention_ms = self._manual_prefix_retention_ms()
        if retention_ms <= 0:
            return
        self._manual_prefix_chunks.append((pcm_bytes, duration_ms))
        total_ms = sum(ms for _, ms in self._manual_prefix_chunks)
        while self._manual_prefix_chunks and total_ms > retention_ms:
            _, removed_ms = self._manual_prefix_chunks.pop(0)
            total_ms -= removed_ms

    def _manual_prefix_retention_ms(self) -> int:
        return max(0, self.manual_prefix_padding_ms) + max(0, self.manual_start_debounce_ms)

    def _consume_manual_prefix(self) -> tuple[bytes, float]:
        if not self._manual_prefix_chunks:
            return b"", 0.0
        prefix_bytes = b"".join(chunk for chunk, _ in self._manual_prefix_chunks)
        prefix_ms = sum(ms for _, ms in self._manual_prefix_chunks)
        self._manual_prefix_chunks.clear()
        return prefix_bytes, prefix_ms

    def _pcm_duration_ms(self, pcm_24k: np.ndarray) -> float:
        return float(len(pcm_24k)) * 1000.0 / float(self.target_sample_rate)

    def _pcm_energy(self, pcm_24k: np.ndarray) -> float:
        if len(pcm_24k) == 0:
            return 0.0
        pcm_float = pcm_24k.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(np.square(pcm_float))))

    def _session_update_event(self) -> dict[str, Any]:
        if self.manual_commit:
            turn_detection: dict[str, Any] | None = None
        else:
            turn_detection = {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_padding_ms,
                "silence_duration_ms": self.vad_silence_duration_ms,
            }

        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.target_sample_rate,
                        },
                        "noise_reduction": {
                            "type": "near_field",
                        },
                        "transcription": {
                            "model": self.model,
                            "language": self.language,
                        },
                        "turn_detection": turn_detection,
                    },
                },
            },
        }

    async def _run_realtime_transcription(self):
        retry_count = 0
        max_retries = 5

        while self.running:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.ws_connect(OPENAI_REALTIME_TRANSCRIPTION_URL, heartbeat=20) as ws:
                        self.ws = ws
                        await ws.send_str(json.dumps(self._session_update_event()))
                        log("info", "OpenAI Realtime transcription WebSocket connected")

                        send_task = asyncio.create_task(self._send_audio_loop(ws))
                        receive_task = asyncio.create_task(self._receive_events_loop(ws))
                        done, pending = await asyncio.wait(
                            {send_task, receive_task},
                            return_when=asyncio.FIRST_EXCEPTION,
                        )

                        for task in pending:
                            task.cancel()
                        for task in pending:
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        for task in done:
                            exc = task.exception()
                            if exc is not None:
                                raise exc
                        if self.running:
                            raise RuntimeError("OpenAI Realtime WebSocket closed")

                retry_count = 0
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                log("info", "OpenAI Realtime ASR streaming cancelled")
                raise
            except Exception as e:
                if self.running:
                    retry_count += 1
                    self.stats["reconnections"] += 1
                    log("error", f"OpenAI Realtime ASR error (retry {retry_count}/{max_retries}): {e}")
                    self._clear_audio_queue()
                    self._reset_transcription_state(clear_prefix=True)
                    if retry_count >= max_retries:
                        log("warning", "Max retries reached, waiting before reset...")
                        await asyncio.sleep(30)
                        retry_count = 0
                    else:
                        await asyncio.sleep(min(retry_count * 0.5, 2.0))
            finally:
                self.ws = None

    async def _send_audio_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while self.running:
            command = await self.audio_queue.get()
            if command is None:
                return

            action, payload = command
            if action == "append":
                assert isinstance(payload, bytes)
                audio_b64 = base64.b64encode(payload).decode("ascii")
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        }
                    )
                )
                self.stats["sent_audio_chunks"] += 1
                continue

            if action == "commit":
                event: dict[str, str] = {"type": "input_audio_buffer.commit"}
                if isinstance(payload, str):
                    event["event_id"] = payload
                await ws.send_str(json.dumps(event))
                self.stats["sent_commit_events"] += 1
                continue

            if action == "clear":
                await ws.send_str(json.dumps({"type": "input_audio_buffer.clear"}))
                self.stats["sent_clear_events"] += 1

    async def _receive_events_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for message in ws:
            if message.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(message.data)
                except json.JSONDecodeError:
                    log("warning", f"Received non-JSON OpenAI Realtime event: {message.data[:200]}")
                    continue
                self._handle_realtime_event(event)
                continue

            if message.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"OpenAI Realtime WebSocket error: {ws.exception()}")

            if message.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                return

    def _handle_realtime_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "input_audio_buffer.committed":
            if self.manual_commit:
                self._handle_manual_committed(event)
            return

        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = str(event.get("delta", ""))
            if not delta:
                return
            item_id = str(event.get("item_id", "default"))
            self.stats["partial_events"] += 1
            if self.manual_commit:
                utterance = self._manual_utterance_for_item(item_id)
                if utterance is None:
                    self._remember_manual_orphan_item(item_id)
                    self._manual_orphan_partials[item_id] = self._manual_orphan_partials.get(item_id, "") + delta
                    return
                if utterance.final_emitted:
                    return
                transcript = utterance.partial_transcripts.get(item_id, "") + delta
                utterance.partial_transcripts[item_id] = transcript
                self._emit_partial(self._combined_manual_transcript(utterance))
            else:
                transcript = self._partial_transcripts.get(item_id, "") + delta
                self._partial_transcripts[item_id] = transcript
                self._emit_partial(transcript)
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            item_id = str(event.get("item_id", "default"))
            transcript = str(event.get("transcript", "")).strip()
            if self.manual_commit:
                utterance = self._manual_utterance_for_item(item_id)
                if utterance is None:
                    self._remember_manual_orphan_item(item_id)
                    self._manual_orphan_completed_items.add(item_id)
                    if transcript:
                        self._manual_orphan_completed_transcripts[item_id] = transcript
                    return
                utterance.partial_transcripts.pop(item_id, None)
                if utterance.final_emitted:
                    return
                utterance.completed_items.add(item_id)
                if transcript:
                    utterance.completed_transcripts[item_id] = transcript
                    self._emit_partial(self._combined_manual_transcript(utterance))
                self._try_emit_manual_final(utterance)
            else:
                self._partial_transcripts.pop(item_id, None)
                if transcript:
                    self._emit_final(transcript)
            return

        if event_type == "error":
            error = event.get("error", event)
            raise RuntimeError(f"OpenAI Realtime error event: {error}")

        if event_type == "conversation.item.input_audio_transcription.failed":
            raw_item_id = event.get("item_id")
            failed_utterance: _ManualASRUtterance | None = None
            if raw_item_id is not None:
                item_id = str(raw_item_id)
                if self.manual_commit:
                    failed_utterance = self._manual_utterance_for_item(item_id)
                    if failed_utterance is None:
                        self._remember_manual_orphan_item(item_id)
                        self._manual_orphan_failed_items.add(item_id)
                    else:
                        failed_utterance.partial_transcripts.pop(item_id, None)
                        failed_utterance.failed_items.add(item_id)
                else:
                    self._partial_transcripts.pop(item_id, None)
            error = event.get("error", event)
            log("warning", f"OpenAI Realtime transcription failed: {error}")
            if self.manual_commit and failed_utterance is not None:
                self._try_emit_manual_final(failed_utterance)

    def _handle_manual_committed(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        utterance_id = (
            self._manual_pending_commit_utterance_ids.popleft() if self._manual_pending_commit_utterance_ids else None
        )
        if utterance_id is not None:
            self._manual_pending_commit_acks = max(0, self._manual_pending_commit_acks - 1)

        utterance = self._manual_utterances.get(utterance_id) if utterance_id is not None else None
        if utterance is None and item_id is not None:
            utterance = self._manual_utterance_for_item(str(item_id), log_unknown=False)
        if utterance is None:
            log("warning", f"OpenAI manual ASR committed event has no matching utterance (item_id={item_id})")
            return

        utterance.pending_commit_acks = max(0, utterance.pending_commit_acks - 1)
        if item_id is None:
            self._try_emit_manual_final(utterance)
            return

        item_id = str(item_id)
        self._register_manual_item(utterance, item_id)
        log("info", f"OpenAI manual ASR buffer committed (utterance_id={utterance.utterance_id}, item_id={item_id})")
        self._try_emit_manual_final(utterance)

    def _register_manual_item(self, utterance: _ManualASRUtterance, item_id: str) -> None:
        existing_utterance_id = self._manual_item_to_utterance_id.get(item_id)
        if existing_utterance_id is not None and existing_utterance_id != utterance.utterance_id:
            log(
                "warning",
                f"OpenAI manual ASR item_id remapped "
                f"(item_id={item_id}, from={existing_utterance_id}, to={utterance.utterance_id})",
            )
        self._manual_item_to_utterance_id[item_id] = utterance.utterance_id
        if item_id not in utterance.item_order:
            utterance.item_order.append(item_id)

        orphan_partial = self._manual_orphan_partials.pop(item_id, None)
        if orphan_partial:
            utterance.partial_transcripts[item_id] = orphan_partial
        if item_id in self._manual_orphan_completed_items:
            utterance.completed_items.add(item_id)
            self._manual_orphan_completed_items.discard(item_id)
            orphan_completed = self._manual_orphan_completed_transcripts.pop(item_id, None)
            if orphan_completed:
                utterance.completed_transcripts[item_id] = orphan_completed
                utterance.partial_transcripts.pop(item_id, None)
                if not utterance.final_emitted:
                    self._emit_partial(self._combined_manual_transcript(utterance))
        if item_id in self._manual_orphan_failed_items:
            utterance.failed_items.add(item_id)
            self._manual_orphan_failed_items.discard(item_id)
            utterance.partial_transcripts.pop(item_id, None)
        self._manual_orphan_item_seen_at.pop(item_id, None)

    def _manual_utterance_for_item(self, item_id: str, log_unknown: bool = True) -> _ManualASRUtterance | None:
        utterance_id = self._manual_item_to_utterance_id.get(item_id)
        if utterance_id is None:
            if log_unknown:
                log("info", f"Deferring OpenAI manual ASR transcript for unmapped item_id={item_id}")
            return None

        utterance = self._manual_utterances.get(utterance_id)
        if utterance is None:
            self._manual_item_to_utterance_id.pop(item_id, None)
            if log_unknown:
                log("warning", f"Ignoring OpenAI manual ASR transcript for expired item_id={item_id}")
            return None
        return utterance

    def _remember_manual_orphan_item(self, item_id: str) -> None:
        self._manual_orphan_item_seen_at[item_id] = time.monotonic()
        self._prune_manual_utterances()

    def _clear_manual_orphan_item(self, item_id: str) -> None:
        self._manual_orphan_partials.pop(item_id, None)
        self._manual_orphan_completed_items.discard(item_id)
        self._manual_orphan_completed_transcripts.pop(item_id, None)
        self._manual_orphan_failed_items.discard(item_id)
        self._manual_orphan_item_seen_at.pop(item_id, None)

    def _combined_manual_transcript(self, utterance: _ManualASRUtterance) -> str:
        parts = []
        for item_id in utterance.item_order:
            transcript = utterance.completed_transcripts.get(item_id)
            if transcript is None:
                transcript = utterance.partial_transcripts.get(item_id)
            if transcript:
                parts.append(transcript.strip())
        return " ".join(parts).strip()

    def _manual_has_open_items(self, utterance: _ManualASRUtterance) -> bool:
        if utterance.pending_commit_acks > 0:
            return True
        for item_id in utterance.item_order:
            if item_id in utterance.completed_items or item_id in utterance.failed_items:
                continue
            return True
        return False

    def _try_emit_manual_final(self, utterance: _ManualASRUtterance) -> None:
        if utterance.final_emitted or not utterance.finalize_requested:
            return
        if self._manual_has_open_items(utterance):
            return

        transcript = self._combined_manual_transcript(utterance)
        if transcript:
            self._emit_final(transcript)
        utterance.final_emitted = True
        utterance.final_emitted_at = time.monotonic()
        utterance.partial_transcripts.clear()
        log(
            "info",
            f"OpenAI manual ASR final emitted "
            f"(utterance_id={utterance.utterance_id}, items={len(utterance.item_order)})",
        )
        self._prune_manual_utterances()

    def _emit_partial(self, transcript: str) -> None:
        if transcript == self.last_partial_text:
            return
        self.last_partial_text = transcript
        if self._on_partial:
            try:
                self._on_partial(transcript)
            except Exception:
                # Callback errors should not stop ASR streaming; ignore.
                pass

    def _emit_final(self, transcript: str) -> None:
        self.last_partial_text = ""
        self.stats["final_transcripts"] += 1
        if self._on_final:
            try:
                self._on_final(transcript)
            except Exception:
                # Callback errors should not stop ASR streaming; ignore.
                pass


def create_asr_processor(
    provider: ASRProvider,
    sample_rate: int,
    model: str = DEFAULT_OPENAI_ASR_MODEL,
    language: str | None = None,
    vad_threshold: float = 0.5,
    vad_prefix_padding_ms: int = 240,
    vad_silence_duration_ms: int = 240,
    manual_commit: bool = DEFAULT_OPENAI_MANUAL_COMMIT,
    manual_commit_interval_ms: int = DEFAULT_OPENAI_MANUAL_COMMIT_INTERVAL_MS,
    manual_min_commit_ms: int = DEFAULT_OPENAI_MANUAL_MIN_COMMIT_MS,
    manual_silence_duration_ms: int = DEFAULT_OPENAI_MANUAL_SILENCE_DURATION_MS,
    manual_prefix_padding_ms: int = DEFAULT_OPENAI_MANUAL_PREFIX_PADDING_MS,
    manual_start_debounce_ms: int = DEFAULT_OPENAI_MANUAL_START_DEBOUNCE_MS,
    local_vad_energy_threshold: float = DEFAULT_OPENAI_LOCAL_VAD_ENERGY_THRESHOLD,
) -> ASRProcessor:
    if provider == "openai":
        return OpenAIRealtimeASRProcessor(
            sample_rate=sample_rate,
            model=model,
            language=language or DEFAULT_OPENAI_ASR_LANGUAGE,
            vad_threshold=vad_threshold,
            vad_prefix_padding_ms=vad_prefix_padding_ms,
            vad_silence_duration_ms=vad_silence_duration_ms,
            manual_commit=manual_commit,
            manual_commit_interval_ms=manual_commit_interval_ms,
            manual_min_commit_ms=manual_min_commit_ms,
            manual_silence_duration_ms=manual_silence_duration_ms,
            manual_prefix_padding_ms=manual_prefix_padding_ms,
            manual_start_debounce_ms=manual_start_debounce_ms,
            local_vad_energy_threshold=local_vad_energy_threshold,
        )
    if provider == "google":
        return GoogleASRProcessor(
            sample_rate=sample_rate,
            language_code=language or DEFAULT_GOOGLE_ASR_LANGUAGE,
        )
    raise ValueError(f"Unsupported ASR provider: {provider}")


def require_initialized_asr(enable_asr: bool, asr_processor: ASRProcessor | None) -> None:
    if not enable_asr:
        return

    if asr_processor is not None and asr_processor.asr_enabled:
        return

    reason = "unknown error"
    if asr_processor is not None and asr_processor.init_error:
        reason = asr_processor.init_error
    raise RuntimeError(
        "ASR is enabled but the selected ASR provider could not be initialized. "
        f"{reason} "
        "Set the required credentials for the selected provider or rerun with --no-enable-asr."
    )
