import argparse
import asyncio
from dataclasses import dataclass
import inspect
import random
import os
from pathlib import Path
from typing import Any, Optional
import tarfile
import time
import secrets
import sys
import threading
import queue
import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
from openai import AsyncOpenAI
from google.cloud import speech
from ._tar_utils import extract_data_archive
from .client_utils import log
from .deferred_logging import DeferredSessionLogger
from .models import loaders, MimiModel, LMModel, LMGen
from .run_inference import get_condition_tensors

# -----------------------
# English-only inference configuration
# -----------------------
SYSTEM_PROMPT = """
You are Moshi, talking with the User. The User is currently mid-conversation.
Predict the flow of the User's dialogue and generate a suitable next response accordingly.
Generate only the dialogue directly, without any additional commentary.
Speak confidently on the predicted topic—there is no need to ask for confirmation.
Your answer must be short and concise in maximum 30 words. Do not include moshi: at the top.
Sometimes you as Moshi say incorrect things. Pay attention to the User's statements and provide correct information.
Since the output words will be spoken, do not include any symbols unrelated to pronunciation (e.g., " ー ;). Avoid anything not relevant to pronunciation.
""".strip()

ASR_LANGUAGE_CODE = "en-US"
ORACLE_EVENT_APPLY_BUDGET_NS = 2_000_000

# -----------------------
# Global conversation state (thread-safe)
# -----------------------
# NOTE: These globals are safe under the current single-session design.
# The ServerState.lock ensures only one WebSocket session is active at a time.
# If multi-session support is needed in the future, encapsulate these into
# a per-session ConversationState class (see server_oracle_former.py).
conversation_text = ""
current_speaker = None
conversation_lock = threading.Lock()
SAVE_DIR: Optional[Path] = None
SESSION_LOGGER: DeferredSessionLogger | None = None


def configure_save_dir(log_dir: str | None) -> None:
    global SAVE_DIR
    if not log_dir:
        SAVE_DIR = None
        return

    SAVE_DIR = Path(log_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log("info", f"Plaintext session logging enabled at {SAVE_DIR}")


def _append_session_log(filename: str, text: str) -> None:
    if SAVE_DIR is None:
        return
    if SESSION_LOGGER is not None and SESSION_LOGGER.active:
        SESSION_LOGGER.append_text(filename, text)
        return
    with (SAVE_DIR / filename).open("a", encoding="utf-8") as f:
        f.write(text)


def _clear_session_logs() -> None:
    if SAVE_DIR is None:
        return

    log_files = [
        "llm_stream_words.txt",
        "user_words.txt",
        "moshi_words.txt",
        "asr_partial.txt",
        "oracle_stream.txt",
        "conversation.txt",
    ]
    for filename in log_files:
        fpath = SAVE_DIR / filename
        if fpath.exists():
            fpath.unlink()
    log("info", f"Cleared session log files in {SAVE_DIR}")


def add_to_conversation(speaker: str, text: str, flush_file: bool = True):
    """Append a *committed* utterance chunk.
    This function is thread-safe. Use it only for committed text (final ASR or Moshi tokens).
    """
    global conversation_text, current_speaker
    text = text.strip()
    if not text:
        return
    snapshot = None
    save_dir = SAVE_DIR
    with conversation_lock:
        if speaker != current_speaker:
            if conversation_text and not conversation_text.endswith("\n"):
                conversation_text += "\n"
            conversation_text += f"{speaker}: "
            current_speaker = speaker
        conversation_text += f"{text} "
        if flush_file and save_dir is not None:
            snapshot = conversation_text
    if snapshot is not None:
        if SESSION_LOGGER is not None and SESSION_LOGGER.active:
            SESSION_LOGGER.replace_text("conversation.txt", snapshot)
        else:
            assert save_dir is not None
            (save_dir / "conversation.txt").write_text(snapshot, encoding="utf-8")


def get_conversation_snapshot() -> str:
    with conversation_lock:
        return conversation_text


def get_last_speaker(conversation_snapshot: str) -> str | None:
    lines = conversation_snapshot.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and ":" in line:
            return line.split(":", 1)[0].strip()
    return None


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class LLMStreamMultiplexer:
    """Starts overlapping LLM streams and adopts the first stream that emits.

    The audio loop is still the only writer to lm_gen. This class only enqueues
    generation-tagged text events for opus_loop to apply.
    """

    def __init__(
        self,
        server_state,
        system_prompt: str = "",
        *,
        min_restart_interval: float = 0.50,
        max_prompt_chars: int = 6000,
        max_concurrent_streams: int = 7,
    ):
        self.server_state = server_state
        self.system_prompt = system_prompt
        if max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before starting the server to enable LLM streaming."
            )
        self.client = AsyncOpenAI()

        self.min_restart_interval = float(min_restart_interval)
        self.max_prompt_chars = max_prompt_chars
        self.max_concurrent_streams = max(1, int(max_concurrent_streams))

        self.loop: asyncio.AbstractEventLoop | None = None
        self._last_start_ts = 0.0
        self._trailing_start_handle: asyncio.TimerHandle | None = None
        self._utterance_id = 0
        self._latest_partial: tuple[int, str] | None = None
        self._last_requested_partial: tuple[int, str] | None = None

        self._gen_counter = 0
        self._tasks: dict[int, asyncio.Task] = {}

        self.adopted_gen = 0
        self._first_emit_ts: dict[int, float] = {}
        self._start_ts: dict[int, float] = {}
        self._latest_gen = 0

        self._adopt_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._running = False
        self._session_id = 0

    def _hot_path_log(self, level: str, message: str) -> None:
        session_logger = getattr(self.server_state, "session_logger", None)
        if session_logger is not None and session_logger.active:
            session_logger.console(level, message)
        else:
            log(level, message)

    def start_session(self, loop: asyncio.AbstractEventLoop) -> int:
        self._cancel_trailing_start()
        self.loop = loop
        self._session_id += 1
        self._running = True
        self._utterance_id = 0
        self._latest_partial = None
        self._last_requested_partial = None
        return self._session_id

    async def warmup_generation(self) -> None:
        started_at = time.monotonic()
        stream = None
        try:
            stream = await self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "Reply OK."}],
                max_completion_tokens=1,
                stream=True,
            )
            async for _ in stream:
                pass
        except Exception as error:
            self._hot_path_log("warning", f"OpenAI generation warm-up failed: {error}")
            return
        finally:
            if stream is not None:
                try:
                    await stream.close()
                except Exception as error:
                    self._hot_path_log("warning", f"OpenAI warm-up stream close failed: {error}")

        elapsed = time.monotonic() - started_at
        self._hot_path_log("info", f"OpenAI generation warmed in {elapsed:.3f}s")

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.start_session(loop)

    @staticmethod
    def _normalize_asr_text(text: str) -> str:
        return " ".join(text.split())

    def on_interim_pending(self, full_text: str) -> None:
        """Try to start a new stream at the fixed restart cadence."""
        loop = self.loop
        if not loop or not self._running:
            return
        session_id = self._session_id
        normalized_text = self._normalize_asr_text(full_text)

        def schedule() -> None:
            if not self._running or session_id != self._session_id:
                return
            utterance_id = self._utterance_id
            self._latest_partial = (utterance_id, normalized_text)
            loop.create_task(
                self._maybe_start_new_stream(
                    bypass_restart_interval=False,
                    session_id=session_id,
                    partial_utterance_id=utterance_id,
                )
            )

        loop.call_soon_threadsafe(schedule)

    def on_final_committed(self, full_text: str) -> None:
        """Start for new final text; otherwise keep the latest partial request."""
        loop = self.loop
        if not loop or not self._running:
            return
        session_id = self._session_id
        normalized_text = self._normalize_asr_text(full_text)

        def schedule() -> None:
            if not self._running or session_id != self._session_id:
                return
            utterance_id = self._utterance_id
            self._utterance_id += 1
            self._cancel_trailing_start()

            already_requested = self._last_requested_partial == (utterance_id, normalized_text)
            self._latest_partial = None
            self._last_requested_partial = None
            if already_requested:
                return

            loop.create_task(
                self._maybe_start_new_stream(
                    bypass_restart_interval=True,
                    session_id=session_id,
                    partial_utterance_id=None,
                )
            )

        loop.call_soon_threadsafe(schedule)

    def _trim_prompt(self, text: str) -> str:
        if len(text) <= self.max_prompt_chars:
            return text
        return text[-self.max_prompt_chars :]

    def _cancel_trailing_start(self) -> None:
        handle = self._trailing_start_handle
        if handle is not None:
            handle.cancel()
            self._trailing_start_handle = None

    def _schedule_trailing_start(self, *, delay: float, session_id: int, utterance_id: int) -> None:
        loop = self.loop
        if not loop or not self._running or session_id != self._session_id:
            return

        handle = self._trailing_start_handle
        if handle is not None and not handle.cancelled():
            return

        def run_trailing_start() -> None:
            self._trailing_start_handle = None
            if not self._running or session_id != self._session_id or utterance_id != self._utterance_id:
                return
            loop.create_task(
                self._maybe_start_new_stream(
                    bypass_restart_interval=False,
                    session_id=session_id,
                    partial_utterance_id=utterance_id,
                )
            )

        self._trailing_start_handle = loop.call_later(max(0.0, delay), run_trailing_start)

    def _build_messages_from_state(self, pending_text: str | None = None) -> tuple[list[dict[str, Any]], bool]:
        committed = get_conversation_snapshot().rstrip()
        if pending_text is None:
            pending_text = self.server_state.get_pending_user_text()
        assert pending_text is not None
        pending = pending_text.strip()

        has_user_input = False
        if pending:
            if committed and not committed.endswith("\n"):
                committed += "\n"
            committed += f"user: {pending} "
            has_user_input = True
        else:
            has_user_input = get_last_speaker(committed) == "user"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._trim_prompt(committed)},
        ]
        return messages, has_user_input

    async def _maybe_start_new_stream(
        self,
        *,
        bypass_restart_interval: bool,
        session_id: int,
        partial_utterance_id: int | None,
    ) -> None:
        if not self._running or session_id != self._session_id:
            return
        if partial_utterance_id is not None and partial_utterance_id != self._utterance_id:
            return

        async with self._start_lock:
            if not self._running or session_id != self._session_id:
                return
            if partial_utterance_id is not None and partial_utterance_id != self._utterance_id:
                return

            requested_partial: tuple[int, str] | None = None
            if partial_utterance_id is not None:
                latest_partial = self._latest_partial
                if latest_partial is None or latest_partial[0] != partial_utterance_id:
                    return
                requested_partial = latest_partial

            messages, has_user_input = self._build_messages_from_state(
                pending_text=requested_partial[1] if requested_partial is not None else None
            )
            if not has_user_input:
                return

            now = time.monotonic()
            elapsed = now - self._last_start_ts
            if not bypass_restart_interval and elapsed < self.min_restart_interval:
                assert partial_utterance_id is not None
                self._schedule_trailing_start(
                    delay=self.min_restart_interval - elapsed,
                    session_id=session_id,
                    utterance_id=partial_utterance_id,
                )
                return

            self._cancel_trailing_start()
            if requested_partial is not None:
                self._last_requested_partial = requested_partial
            self._last_start_ts = now
            await self._start_stream(messages, session_id=session_id)

    async def _start_stream(self, messages: list[dict[str, Any]], *, session_id: int):
        if not self._running or session_id != self._session_id:
            return
        assert self.loop is not None

        self._gen_counter += 1
        gen_id = self._gen_counter
        self._start_ts[gen_id] = time.monotonic()
        self._first_emit_ts[gen_id] = 0.0

        task = self.loop.create_task(self._stream_single(messages, gen_id, session_id))
        self._tasks[gen_id] = task
        self._hot_path_log("info", f"LLM started (gen {gen_id})")

        await self._enforce_stream_limit()

    async def _adopt_generation(self, gen_id: int):
        """Adopt gen_id as the active stream when it emits its first token."""
        async with self._adopt_lock:
            if gen_id <= self.adopted_gen:
                return

            try:
                while True:
                    self.server_state.llm_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            for gid, task in list(self._tasks.items()):
                if gid < gen_id and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    self._tasks.pop(gid, None)

            self.adopted_gen = gen_id
            self._latest_gen = gen_id

            start_ts = self._start_ts.get(gen_id)
            if start_ts:
                ttft = time.monotonic() - start_ts
                self._hot_path_log("info", f"LLM adopted (gen {gen_id}) TTFT={ttft:.3f}s")

            await self._enforce_stream_limit()

    async def _enforce_stream_limit(self):
        live = [(gid, task) for gid, task in self._tasks.items() if not task.done()]
        if len(live) <= self.max_concurrent_streams:
            return

        live_gids = {gid for gid, _ in live}
        keep: set[int] = set()
        if self.adopted_gen in live_gids:
            keep.add(self.adopted_gen)

        not_emitting_newest = [
            gid
            for gid, _ in sorted(live, key=lambda item: item[0], reverse=True)
            if self._first_emit_ts.get(gid, 0.0) == 0.0 and gid != self.adopted_gen
        ]
        budget = self.max_concurrent_streams - len(keep)
        if budget > 0:
            keep.update(not_emitting_newest[:budget])

        for gid, task in live:
            if gid in keep or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._tasks.pop(gid, None)

    async def _stream_single(self, messages: list[dict[str, Any]], gen_id: int, session_id: int):
        try:
            if not self._running or session_id != self._session_id:
                return

            stream = await self.client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            )

            async for chunk in stream:
                if not self._running or session_id != self._session_id:
                    return

                if not (chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content):
                    continue

                text = (chunk.choices[0].delta.content or "").strip()
                if not text:
                    continue

                if not self._first_emit_ts.get(gen_id, 0.0):
                    self._first_emit_ts[gen_id] = time.monotonic()
                    if gen_id > self.adopted_gen:
                        await self._adopt_generation(gen_id)
                    else:
                        continue

                if gen_id != self.adopted_gen:
                    continue

                if not self._running or session_id != self._session_id:
                    return

                await self.server_state.llm_event_queue.put(("append", gen_id, text))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log("error", f"LLM gen {gen_id} streaming error: {e}")
        finally:
            self._tasks.pop(gen_id, None)

    async def stop(self):
        self._running = False
        self._session_id += 1
        self._cancel_trailing_start()

        for _, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()
        self._gen_counter = 0
        self.adopted_gen = 0
        self._latest_gen = 0
        self._first_emit_ts.clear()
        self._start_ts.clear()
        self._last_start_ts = 0.0
        self._utterance_id = 0
        self._latest_partial = None
        self._last_requested_partial = None
        self.loop = None


class AsyncASRProcessor:
    """Async ASR with Google Speech-to-Text. Produces partial (pending) and final commits via callbacks.
    Audio is pushed via process_audio(...) from the main audio loop thread.
    """

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000  # Google Speech API requirement

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
            language_code = ASR_LANGUAGE_CODE
            self.config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.target_sample_rate,
                language_code=language_code,
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
            log("info", f"Async ASR processor initialized (language: {language_code})")
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

    @staticmethod
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
            pcm_16k = self._linear_resample_int16(pcm_16bit, self.sample_rate, self.target_sample_rate)

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
        except Exception as e:
            if self.running:
                raise e

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


def _require_initialized_asr(enable_asr: bool, asr_processor: Optional[AsyncASRProcessor]) -> None:
    if not enable_asr:
        return

    if asr_processor is not None and asr_processor.asr_enabled:
        return

    reason = "unknown error"
    if asr_processor is not None and asr_processor.init_error:
        reason = asr_processor.init_error
    raise RuntimeError(
        "ASR is enabled but Google Speech-to-Text could not be initialized. "
        f"{reason} "
        "Set GOOGLE_APPLICATION_CREDENTIALS to a valid Google Cloud service account credential file "
        "or rerun with --no-enable-asr."
    )


@dataclass
class ServerState:
    model_type: str
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock
    asr_processor: Optional[AsyncASRProcessor] = None

    def __init__(
        self,
        model_type: str,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        cfg_coef: float,
        device: str | torch.device,
        enable_asr: bool = True,
        min_restart_interval: float = 0.50,
        max_prompt_chars: int = 6000,
        max_concurrent_streams: int = 7,
        **kwargs,
    ):
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=cfg_coef)
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs)

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()
        self.session_logger = DeferredSessionLogger(SAVE_DIR)
        global SESSION_LOGGER
        SESSION_LOGGER = self.session_logger

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # LLM text events from the multiplexer. opus_loop is the only lm_gen writer.
        self.llm_event_queue: asyncio.Queue[tuple[str, int, str]] = asyncio.Queue(maxsize=256)

        # Pending ASR text (not yet committed to conversation)
        self._pending_user_text = ""
        self._pending_lock = asyncio.Lock()

        # Cumulative word count for ASR partial logging (only log when words increase)
        self._last_logged_total_units = 0

        # ASR word count state for partial logging and diagnostics.
        # Counts user words only and excludes moshi output.
        self._max_pending_units = 0
        self._committed_units_asr = 0

        # Event loop handle (set in handle_chat)
        self.loop: asyncio.AbstractEventLoop | None = None

        # ASR processor
        self.asr_processor = AsyncASRProcessor(sample_rate=int(self.mimi.sample_rate)) if enable_asr else None
        _require_initialized_asr(enable_asr, self.asr_processor)

        # Parallel LLM stream multiplexer. It never touches lm_gen directly.
        self.llm_mux = LLMStreamMultiplexer(
            server_state=self,
            system_prompt=SYSTEM_PROMPT,
            min_restart_interval=min_restart_interval,
            max_prompt_chars=max_prompt_chars,
            max_concurrent_streams=max_concurrent_streams,
        )

    # ----- Pending user text API -----
    def get_pending_user_text(self) -> str:
        return self._pending_user_text

    def _asr_on_partial(self, text: str):
        # Called from a worker thread; schedule into event loop
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._asr_on_partial_async(text), self.loop)

    def _count_units(self, text: str) -> int:
        """Count whitespace-delimited words."""
        if not text or not text.strip():
            return 0
        return len(text.split())

    async def _asr_on_partial_async(self, text: str):
        # Minimal locking; do not write to conversation here
        async with self._pending_lock:
            self._pending_user_text = text

        pending_units = self._count_units(text)
        if pending_units > self._max_pending_units:
            self._max_pending_units = pending_units

        # For ASR logging: use user word count only
        current_total_units = self._committed_units_asr + pending_units

        # Only log when cumulative word count increases (new words added)
        if current_total_units > self._last_logged_total_units:
            units_added = current_total_units - self._last_logged_total_units
            self._last_logged_total_units = current_total_units
            self._hot_path_log("info", f"[ASR Partial +{units_added}] {text}")
            # Log ASR partial for visualization
            timestamp_ms = int(time.time() * 1000)
            _append_session_log("asr_partial.txt", f"{timestamp_ms}: {text}\n")

        self.llm_mux.on_interim_pending(text)

    def _asr_on_final(self, text: str):
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._asr_on_final_async(text), self.loop)

    async def _asr_on_final_async(self, text: str):
        text = text.strip()
        if text:
            add_to_conversation("user", text, flush_file=True)
            # Increment ASR-only word count (excludes moshi output)
            self._committed_units_asr += self._count_units(text)
            # Log user words with timestamp for analysis
            timestamp_ms = int(time.time() * 1000)
            _append_session_log("user_words.txt", f"{timestamp_ms}: {text}\n")
        async with self._pending_lock:
            self._pending_user_text = ""
        self._max_pending_units = 0
        if text:
            self.llm_mux.on_final_committed(text)

    async def _cleanup_llm_stream(self):
        """Stop LLM streams and drain queued text to avoid leakage between sessions."""
        await self.llm_mux.stop()
        try:
            while True:
                self.llm_event_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Queue is empty; draining complete.
            pass

    # ----------------------------------

    def _hot_path_log(self, level: str, message: str) -> None:
        if self.session_logger.active:
            self.session_logger.console(level, message)
        else:
            log(level, message)

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        resolved_device = torch.device(self.device)
        if torch.cuda.is_available() and resolved_device.type == "cuda":
            torch.cuda.synchronize(device=resolved_device)

    def __del__(self):
        if hasattr(self, "asr_processor") and self.asr_processor:
            self.asr_processor.running = False

    async def handle_chat(self, request):
        global conversation_text, current_speaker

        # Reject if another session is active (early return for better UX)
        if self.lock.locked():
            return web.Response(status=503, text="Server busy - another session is active")

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    data = message.data
                    if not isinstance(data, bytes):
                        log("error", f"unsupported message type {type(data)}")
                        continue
                    if len(data) == 0:
                        log("warning", "empty message")
                        continue
                    kind = data[0]
                    if kind == 1:  # audio
                        payload = data[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                log("info", "connection closed (recv_loop)")

        async def opus_loop():
            """Single owner of lm_gen operations. It drains LLM events and updates oracle tokens here."""
            all_pcm_data = None
            skip_frames = 1
            active_gen = 0

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)

                # Drain LLM events and update oracle tokens BEFORE reading pcm.
                drain_start_ns = time.perf_counter_ns()
                drained_events = 0
                try:
                    while True:
                        if drained_events and time.perf_counter_ns() - drain_start_ns >= ORACLE_EVENT_APPLY_BUDGET_NS:
                            break
                        event_type, gen_id, text = self.llm_event_queue.get_nowait()
                        drained_events += 1
                        if event_type != "append" or not text:
                            continue
                        if gen_id < active_gen:
                            continue

                        timestamp_ms = int(time.time() * 1000)
                        if gen_id > active_gen:
                            self.lm_gen.update_oracle_tokens_streaming(None, reset=True)
                            active_gen = gen_id
                            _append_session_log("oracle_stream.txt", f"{timestamp_ms}: [RESET]\n")
                            _append_session_log("llm_stream_words.txt", f"{timestamp_ms}: [RESET gen={gen_id}]\n")

                        token_ids = list(self.text_tokenizer.encode(text))  # type: ignore[attr-defined]
                        self.lm_gen.update_oracle_tokens_streaming(token_ids, reset=False)
                        self._hot_path_log("info", f"[LLM] {text}")
                        _append_session_log("oracle_stream.txt", f"{timestamp_ms}: {text}\n")
                        _append_session_log("llm_stream_words.txt", f"{timestamp_ms}: {text}\n")
                except asyncio.QueueEmpty:
                    # Queue is empty; nothing to process, continue to next iteration.
                    pass

                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue

                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))

                while all_pcm_data.shape[-1] >= self.frame_size:
                    # Encode a frame
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size :]

                    # Feed ASR
                    if self.asr_processor:
                        self.asr_processor.process_audio(chunk.copy())

                    # Decode audio with moshi
                    chunk_t = torch.from_numpy(chunk).to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk_t)
                    if skip_frames:
                        self.mimi.reset_streaming()
                        skip_frames -= 1

                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore[attr-defined]
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            self._hot_path_log("info", f"text token '{_text}'")
                            add_to_conversation("moshi", _text.strip(), flush_file=False)
                            timestamp_ms = int(time.time() * 1000)
                            _append_session_log("moshi_words.txt", f"{timestamp_ms}: {_text.strip()}\n")
                            await ws.send_bytes(msg)

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        log("info", "accepted connection")
        close = False
        tasks: list[asyncio.Task] = []
        async with self.lock:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            try:
                # All initialization and cleanup stay inside the session lock.
                with conversation_lock:
                    conversation_text = ""
                    current_speaker = None

                # Reset ASR state for new session
                async with self._pending_lock:
                    self._pending_user_text = ""
                self._committed_units_asr = 0
                self._last_logged_total_units = 0
                self._max_pending_units = 0

                _clear_session_logs()

                # Stop any old LLM stream and drain old generation events.
                await self._cleanup_llm_stream()

                self.session_logger.start_session()

                self.loop = asyncio.get_running_loop()
                self.llm_mux.start_session(self.loop)
                await self.llm_mux.warmup_generation()

                # Register ASR callbacks (must be before start)
                if self.asr_processor:
                    self.asr_processor.register_callbacks(self._asr_on_partial, self._asr_on_final)
                    await self.asr_processor.start()

                # Initialize streaming components
                opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
                opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
                self.mimi.reset_streaming()
                self.lm_gen.reset_streaming()
                await ws.send_bytes(b"\x00")  # handshake
                tasks = [
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(send_loop()),
                ]
                await asyncio.gather(*tasks)
            finally:
                close = True
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                if self.asr_processor:
                    await self.asr_processor.stop()

                await self._cleanup_llm_stream()

                self.loop = None
                self.session_logger.finish_session()

        log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action="store_true", help="Activate a gradio tunnel.")
    parser.add_argument(
        "--gradio-tunnel-token", help="Provide a custom (secret) token here to keep getting the same URL."
    )

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument(
        "--moshi-weight",
        type=str,
        help="Path to a local checkpoint file for KAME or Moshi-compatible weights.",
    )
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into. Defaults to the upstream Moshi checkpoint repo.",
    )
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1.0, help="CFG coefficient.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument(
        "--no_fuse_lora",
        action="store_false",
        dest="fuse_lora",
        default=True,
        help="Do not fuse LoRA layers into Linear layers.",
    )
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Run inference with float16, not bfloat16, better for old GPUs.",
    )
    parser.add_argument(
        "--enable-asr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ASR processing for transcription (default: True)",
    )
    parser.add_argument(
        "--min-restart-interval",
        type=float,
        default=0.50,
        help="Minimum interval in seconds between background LLM stream starts.",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=6000,
        help="Maximum prompt characters sent to the backend LLM.",
    )
    parser.add_argument(
        "--max-concurrent-streams",
        type=int,
        default=7,
        help="Maximum number of concurrent background LLM streams.",
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for plaintext local session logs. If omitted, no local "
            "conversation or token logs are written. Can also be set via MOSHI_LOG_DIR."
        ),
    )

    args = parser.parse_args()
    seed_all(42424242)
    configure_save_dir(args.log_dir or os.environ.get("MOSHI_LOG_DIR"))

    setup_tunnel = None
    tunnel_token = ""
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log(
                "error",
                "Cannot find gradio which is required to activate a tunnel. Please install the optional tunnel support with `pip install 'kame-model[tunnel]'`.",
            )
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        args.moshi_weight,
        args.mimi_weight,
        args.tokenizer,
        lora_weights=args.lora_weight,
        config_path=args.config_path,
    )
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading language model")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora)
    log("info", "language model loaded")

    state = ServerState(
        checkpoint_info.model_type,
        mimi,
        text_tokenizer,
        lm,
        args.cfg_coef,
        args.device,
        enable_asr=args.enable_asr,
        min_restart_interval=args.min_restart_interval,
        max_prompt_chars=args.max_prompt_chars,
        max_concurrent_streams=args.max_concurrent_streams,
        **checkpoint_info.lm_gen_config,
    )
    log("info", "warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    static_path: None | str = None
    if args.static is None:
        log("info", "retrieving the static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                extract_data_archive(tar, dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        static_path = args.static

    if static_path is not None:

        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        log("info", f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=False, name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if args.enable_asr:
        log(
            "info",
            "ASR processing enabled (English) - partials nudge parallel LLM streams; finals commit to transcript",
        )
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None, **tunnel_kwargs)
        log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
        log("info", "Note that this tunnel goes through the US and you might experience high latency in Europe.")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


def cli():
    with torch.no_grad():
        main()


if __name__ == "__main__":
    cli()
