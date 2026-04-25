import asyncio
import base64
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
DEFAULT_OPENAI_ASR_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_OPENAI_ASR_LANGUAGE = "en"
OPENAI_REALTIME_TRANSCRIPTION_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


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

        self.audio_buffer = queue.Queue(maxsize=100)  # thread-safe
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
        vad_prefix_padding_ms: int = 300,
        vad_silence_duration_ms: int = 500,
    ):
        self.sample_rate = sample_rate
        self.target_sample_rate = 24000  # Realtime transcription supports 24 kHz mono PCM input.
        self.model = model
        self.language = language
        self.vad_threshold = vad_threshold
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.vad_silence_duration_ms = vad_silence_duration_ms

        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
        self.asr_task: asyncio.Task | None = None

        self.stats = {
            "partial_events": 0,
            "final_transcripts": 0,
            "buffer_drops": 0,
            "reconnections": 0,
            "sent_audio_chunks": 0,
        }

        self.asr_enabled = False
        self.init_error: str | None = None
        self.api_key = os.environ.get("OPENAI_API_KEY")

        self._on_partial = None
        self._on_final = None

        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._partial_transcripts: dict[str, str] = {}
        self.last_partial_text = ""

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
        log("info", f"OpenAI Realtime ASR initialized (model: {self.model}, language: {self.language})")

    async def start(self):
        if self.asr_enabled and not self.running:
            self.running = True
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

            try:
                self.audio_queue.put_nowait(pcm_24k.tobytes())
            except asyncio.QueueFull:
                self.stats["buffer_drops"] += 1
                self._drop_oldest_audio_chunk()
                try:
                    self.audio_queue.put_nowait(pcm_24k.tobytes())
                except asyncio.QueueFull:
                    # If the sender is still behind, drop the new chunk too and keep
                    # the main audio loop moving.
                    pass
        except Exception as e:
            log("error", f"Error processing audio: {e}")

    def _drop_oldest_audio_chunk(self) -> None:
        try:
            self.audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    def _clear_audio_queue(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _session_update_event(self) -> dict[str, Any]:
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
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": self.vad_threshold,
                            "prefix_padding_ms": self.vad_prefix_padding_ms,
                            "silence_duration_ms": self.vad_silence_duration_ms,
                        },
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
            chunk = await self.audio_queue.get()
            if chunk is None:
                return

            audio_b64 = base64.b64encode(chunk).decode("ascii")
            await ws.send_str(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                )
            )
            self.stats["sent_audio_chunks"] += 1

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

        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = str(event.get("delta", ""))
            if not delta:
                return
            item_id = str(event.get("item_id", "default"))
            transcript = self._partial_transcripts.get(item_id, "") + delta
            self._partial_transcripts[item_id] = transcript
            self.stats["partial_events"] += 1
            self._emit_partial(transcript)
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            item_id = str(event.get("item_id", "default"))
            transcript = str(event.get("transcript", "")).strip()
            self._partial_transcripts.pop(item_id, None)
            if transcript:
                self._emit_final(transcript)
            return

        if event_type == "error":
            error = event.get("error", event)
            raise RuntimeError(f"OpenAI Realtime error event: {error}")

        if event_type == "conversation.item.input_audio_transcription.failed":
            item_id = event.get("item_id")
            if item_id is not None:
                self._partial_transcripts.pop(str(item_id), None)
            error = event.get("error", event)
            log("warning", f"OpenAI Realtime transcription failed: {error}")

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
) -> ASRProcessor:
    if provider == "openai":
        return OpenAIRealtimeASRProcessor(
            sample_rate=sample_rate,
            model=model,
            language=language or DEFAULT_OPENAI_ASR_LANGUAGE,
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
