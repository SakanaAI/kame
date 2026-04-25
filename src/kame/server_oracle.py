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
import collections
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
    with conversation_lock:
        if speaker != current_speaker:
            if conversation_text and not conversation_text.endswith("\n"):
                conversation_text += "\n"
            conversation_text += f"{speaker}: "
            current_speaker = speaker
        conversation_text += f"{text} "
        if flush_file:
            if SAVE_DIR is not None:
                (SAVE_DIR / "conversation.txt").write_text(conversation_text, encoding="utf-8")


def get_conversation_snapshot() -> str:
    with conversation_lock:
        return conversation_text


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class LLMStreamManager:
    """Continuously (re)streams the LLM with debouncing and pushes text to the audio LM via an oracle queue.

    Important: we NEVER call lm_gen directly from this class. We only put ('reset'|'append', payload) into oracle_queue.
    The audio loop is the single writer to lm_gen to avoid race conditions.
    """

    def __init__(self, server_state, interval=0.25, system_prompt=""):
        self.server_state = server_state
        self.interval = interval
        self.system_prompt = system_prompt
        self.current_stream = None
        self.running = False

        # Validate that the OpenAI API key is available at initialization time
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before starting the server to enable LLM streaming."
            )
        self.client = AsyncOpenAI()  # API key taken from environment

        # Debounce / restart control
        self.last_start_time = 0.0
        self.restart_history = collections.deque(maxlen=10)  # timestamps

        # Cumulative word count tracking (handles partial text sliding window)
        self.last_start_total_units = 0

        self.min_units_delta = 2  # Require 2+ new words before restart
        self.max_restarts_per_2s = 5

    def _count_units(self, text: str) -> int:
        """Count whitespace-delimited words."""
        if not text or not text.strip():
            return 0
        return len(text.split())

    def _restart_allowed(self) -> bool:
        now = time.time()
        # keep only the last 2 seconds
        while self.restart_history and now - self.restart_history[0] > 2.0:
            self.restart_history.popleft()
        return len(self.restart_history) < self.max_restarts_per_2s

    async def run_periodic_streaming(self):
        """Polls pending user text and (re)starts the LLM stream when it grows meaningfully or finalizes."""
        while self.running:
            try:
                await asyncio.sleep(self.interval)

                # Snapshot state
                pending = self.server_state.get_pending_user_text()
                finalized_bump = self.server_state.consume_and_clear_final_bump()
                committed_conv = get_conversation_snapshot()  # Full conversation (user + moshi)

                # Calculate cumulative USER word count (ASR only, excludes moshi output)
                # This prevents LLM restarts when moshi speaks
                committed_units_asr = self.server_state._committed_units_asr
                pending_units = self._count_units(pending) if pending else 0
                max_pending = max(pending_units, self.server_state._max_pending_units)
                current_total_units = committed_units_asr + max_pending

                # Decide whether to (re)start
                need_restart = False
                if self.current_stream is None:
                    # Start as soon as there's anything pending or after any finalization.
                    need_restart = (len(pending.strip()) > 0) or finalized_bump
                else:
                    # Restart only if cumulative USER word count grew significantly or finalization
                    units_added = current_total_units - self.last_start_total_units
                    if units_added >= self.min_units_delta or finalized_bump:
                        need_restart = True

                if need_restart and self._restart_allowed():
                    if self.current_stream:
                        self.current_stream.cancel()
                        self.current_stream = None
                    self.restart_history.append(time.time())
                    self.current_stream = asyncio.create_task(self._stream_llm_response(committed_conv, pending))
                    self.last_start_total_units = current_total_units
                    self.last_start_time = time.time()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log("error", f"LLM periodic loop error: {e}")

    def _build_messages(self, committed_conversation: str, pending_user_text: str) -> list[dict[str, Any]]:
        # Provide the LLM with the whole conversation and the *current* pending user text.
        convo = committed_conversation.rstrip()
        if pending_user_text.strip():
            if not convo.endswith("\n"):
                convo += "\n"
            convo += f"user: {pending_user_text.strip()} "
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": convo}]
        return messages

    async def _stream_llm_response(self, committed_conversation, pending_user_text):
        """Stream LLM tokens and enqueue oracle updates. Audio loop is the single writer to lm_gen."""
        stream_start_ms = int(time.time() * 1000)  # Timestamp for this entire stream session
        stream_tokens = []  # Collect all tokens in this stream session for logging
        try:
            messages: list[dict[str, Any]] = self._build_messages(committed_conversation, pending_user_text)
            stream = await self.client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            )
            first_chunk = True
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                    stripped_text = chunk_text.strip()
                    if not stripped_text:
                        continue

                    if first_chunk:
                        # Issue a reset to the oracle at the very beginning of a new run
                        try:
                            self.server_state.oracle_queue.put_nowait(("reset", ""))
                        except asyncio.QueueFull:
                            # Reset is important but not strictly critical - the next LLM restart
                            # will issue another reset. Log for monitoring purposes.
                            log("warning", "Oracle queue full; reset command dropped (will retry on next restart)")
                            pass
                        first_chunk = False

                    # Log all LLM tokens for debugging (regardless of queue status)
                    stream_tokens.append(stripped_text)
                    log("info", f"[LLM] {stripped_text}")

                    # Append chunk to oracle via queue
                    try:
                        self.server_state.oracle_queue.put_nowait(("append", stripped_text))
                    except asyncio.QueueFull:
                        # If congested, we drop tiny chunks. Oracle will catch up next restart.
                        log("warning", "Oracle queue full; dropping small LLM chunk.")

            # Write all tokens from this stream session with the same timestamp
            if stream_tokens:
                _append_session_log("llm_stream_words.txt", f"{stream_start_ms}: {' '.join(stream_tokens)}\n")
        except asyncio.CancelledError:
            log("info", "LLM stream cancelled")
            # Log cancelled stream tokens with [CANCELLED] marker
            if stream_tokens:
                _append_session_log(
                    "llm_stream_words.txt",
                    f"{stream_start_ms}: [CANCELLED] {' '.join(stream_tokens)}\n",
                )
            raise
        except Exception as e:
            log("error", f"LLM streaming error: {e}")
            # Log error stream tokens with [ERROR] marker
            if stream_tokens:
                _append_session_log(
                    "llm_stream_words.txt",
                    f"{stream_start_ms}: [ERROR] {' '.join(stream_tokens)}\n",
                )

    async def stop(self):
        """Stop the LLM stream manager and cancel any running stream.

        This ensures both the periodic loop and any active LLM stream are properly
        cancelled to prevent token leakage between sessions.
        """
        self.running = False

        # Cancel the current LLM stream if running
        if self.current_stream is not None:
            self.current_stream.cancel()
            try:
                await self.current_stream
            except asyncio.CancelledError:
                # Task cancellation is expected during cleanup; safe to ignore.
                pass
            self.current_stream = None


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

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Oracle queue: single writer to lm_gen lives in opus_loop
        self.oracle_queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        # Pending ASR text (not yet committed to conversation)
        self._pending_user_text = ""
        self._pending_lock = asyncio.Lock()
        self._final_bump_flag = False  # Set when final transcript lands

        # Cumulative word count for ASR partial logging (only log when words increase)
        self._last_logged_total_units = 0

        # Cumulative ASR word count (user words only, excludes moshi output)
        # Used for LLM restart decisions to avoid restarting when moshi speaks
        self._max_pending_units = 0
        self._committed_units_asr = 0

        # Event loop handle (set in handle_chat)
        self.loop: asyncio.AbstractEventLoop | None = None

        # ASR processor
        self.asr_processor = AsyncASRProcessor(sample_rate=int(self.mimi.sample_rate)) if enable_asr else None
        _require_initialized_asr(enable_asr, self.asr_processor)

        # LLM stream manager (uses oracle_queue; never touches lm_gen directly)
        self.llm_stream_manager = LLMStreamManager(
            server_state=self,
            interval=0.5,
            system_prompt=SYSTEM_PROMPT,
        )
        self.llm_stream_task = None

    # ----- Pending user text API -----
    def get_pending_user_text(self) -> str:
        return self._pending_user_text

    def consume_and_clear_final_bump(self) -> bool:
        if self._final_bump_flag:
            self._final_bump_flag = False
            return True
        return False

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
            log("info", f"[ASR Partial +{units_added}] {text}")
            # Log ASR partial for visualization
            timestamp_ms = int(time.time() * 1000)
            _append_session_log("asr_partial.txt", f"{timestamp_ms}: {text}\n")

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
        self._final_bump_flag = True
        self._max_pending_units = 0

    async def _cleanup_llm_stream(self):
        """Stop LLM stream manager and cancel the periodic task."""
        await self.llm_stream_manager.stop()
        if self.llm_stream_task:
            self.llm_stream_task.cancel()
            try:
                await self.llm_stream_task
            except asyncio.CancelledError:
                # Task cancellation is expected during cleanup; safe to ignore.
                pass
            self.llm_stream_task = None

        # Drain oracle queue to prevent token leakage between sessions
        try:
            while True:
                self.oracle_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Queue is empty; draining complete.
            pass

    # ----------------------------------

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
            """Single owner of lm_gen operations. It drains oracle_queue and calls update_oracle_tokens_streaming here."""
            all_pcm_data = None
            skip_frames = 1

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)

                # Drain oracle_queue and update oracle tokens BEFORE reading pcm
                try:
                    while True:
                        action, payload = self.oracle_queue.get_nowait()
                        timestamp_ms = int(time.time() * 1000)
                        if action == "reset":
                            self.lm_gen.update_oracle_tokens_streaming(None, reset=True)
                            _append_session_log("oracle_stream.txt", f"{timestamp_ms}: [RESET]\n")
                        elif action == "append" and payload:
                            token_ids = list(self.text_tokenizer.encode(payload))  # type: ignore[attr-defined]
                            self.lm_gen.update_oracle_tokens_streaming(token_ids, reset=False)
                            _append_session_log("oracle_stream.txt", f"{timestamp_ms}: {payload}\n")
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
                            log("info", f"text token '{_text}'")
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
        async with self.lock:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            # All initialization inside lock to prevent race conditions
            # Reset session transcript
            with conversation_lock:
                conversation_text = ""
                current_speaker = None

            # Reset ASR-only word counter for new session
            self._committed_units_asr = 0
            self._last_logged_total_units = 0
            self._max_pending_units = 0

            self.llm_stream_manager.last_start_total_units = 0
            self.llm_stream_manager.current_stream = None
            self.llm_stream_manager.restart_history.clear()

            _clear_session_logs()

            # Stop any old LLM stream (including current_stream)
            await self._cleanup_llm_stream()

            self.loop = asyncio.get_running_loop()

            # Register ASR callbacks (must be before start)
            if self.asr_processor:
                self.asr_processor.register_callbacks(self._asr_on_partial, self._asr_on_final)
                await self.asr_processor.start()

            # Start LLM stream manager
            self.llm_stream_manager.running = True
            self.llm_stream_task = asyncio.create_task(self.llm_stream_manager.run_periodic_streaming())

            # Initialize streaming components
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            await ws.send_bytes(b"\x00")  # handshake
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())

        # Cleanup
        await self._cleanup_llm_stream()

        if self.asr_processor:
            await self.asr_processor.stop()

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
        log("info", "ASR processing enabled (English) - partials stream to LLM; finals commit to transcript")
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
