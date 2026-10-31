"""
Claude Voice Commander — hands-free dictation for the Claude Desktop app.

Flow:
    (Claude window active) say "start mic"  ->  beep, dictation starts
    speak your message                       ->  text collects across pauses
    say "enter"                              ->  pasted into Claude and sent
    app goes back to waiting for "start mic"

    "cancel" / "clear" at the end throws the message away.
    "edit X to Y" fixes a word before sending.
    F4 (configurable) starts/stops dictation without the wake phrase.

Speech recognition runs locally with faster-whisper; no audio leaves your computer.
"""
import os
import platform
import queue
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

from audio_utils import SAMPLING_RATE, UtteranceSegmenter, make_beep, resample_to_16k
from config_loader import load_and_validate_config
from focus import app_matches, get_foreground_app
from text_processing import (
    check_clear_trigger,
    check_trailing_trigger,
    detect_wake_phrase,
    is_probable_hallucination,
    process_edit_commands,
)

__version__ = "3.0.9"

BLOCK_DURATION = 0.1          # 100 ms audio chunks
MAX_QUEUE_BLOCKS = 300        # at most 30 s of audio buffered
FOCUS_POLL_SECONDS = 0.4
BEEP_SAMPLE_RATE = 44100
MUTE_AFTER_BEEP = 0.35        # ignore mic briefly so our own beep isn't transcribed

def resolve_model_path(name: str) -> str:
    """
    whisper_model may be a model name ("base.en") or a folder with a
    downloaded model. A relative folder is looked up next to main.py, so it
    works no matter which directory the app is started from.
    """
    expanded = os.path.expanduser(name)
    if os.path.isabs(expanded):
        return expanded
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), expanded)
    return local if os.path.isdir(local) else name


class PastedNotSubmitted(Exception):
    """The message reached Claude's text box, but pressing Enter failed."""


class ShuttingDown(Exception):
    """The app is quitting; nothing was copied or pasted."""


WAITING = "waiting"           # listening only for the wake phrase
DICTATING = "dictating"       # collecting a message until "enter"


class VoiceCommander:
    def __init__(self, config: dict):
        self.cfg = config
        self.submit = config["triggers"]["submit"]
        self.edit = config["triggers"]["edit"]
        self.clear = config["triggers"]["clear"]

        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.focused = threading.Event()   # True while audio should be captured
        self._segmenter_reset = threading.Event()
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=MAX_QUEUE_BLOCKS)

        self.state = WAITING if config["wake_word_enabled"] else DICTATING
        self.generation = 0                # bumped on every state change (stale-result guard)
        self.parts: list[str] = []
        self.last_activity = time.monotonic()
        self.mute_until = 0.0

        self.model = None
        self.stream = None
        self.stream_sr = SAMPLING_RATE
        self.stream_running = False
        self.segmenter: Optional[UtteranceSegmenter] = None
        self._kb = None
        self._warned_focus_unknown = False
        self._warned_resume = False
        self.retry_pending = False         # a failed send's text is waiting for "enter"
        self._restore_timer: Optional[threading.Timer] = None
        self._clip_lock = threading.Lock()  # serializes our clipboard writes vs. delayed restore

    # ------------------------------------------------------------------ setup

    def load_model(self):
        # Windows without Developer Mode can't make symlinks; Hugging Face then
        # prints a long, harmless warning on every first run. Hide it.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from faster_whisper import WhisperModel
        name = resolve_model_path(self.cfg["whisper_model"])
        if os.path.isdir(name):
            print(f"Loading speech model from folder '{name}'...")
        else:
            print(f"Loading speech model '{name}'... "
                  f"(first run downloads ~150 MB; Ctrl+C to abort)")
        try:
            self.model = WhisperModel(name, device="cpu", compute_type="int8")
        except Exception as e:
            print(f"\nFatal: could not load whisper_model '{name}': {e}", file=sys.stderr)
            if not os.path.isdir(name):
                print("\nThe model download failed. See \"Speech model download fails\" in "
                      "README.md: you can download the model in your browser instead.",
                      file=sys.stderr)
            sys.exit(1)
        print("Speech model ready.")

    def open_stream(self):
        """
        Try 16 kHz (Whisper's native rate). If the device refuses, retry at its
        native rate and resample later. The stream is started here (not via a
        `with` block) so a device that only fails on .start() still falls back.
        """
        import sounddevice as sd
        device = self.cfg["input_device"]
        errors = []
        rates = [SAMPLING_RATE]
        try:
            native = int(sd.query_devices(device, "input")["default_samplerate"])
            if native != SAMPLING_RATE:
                rates.append(native)
        except Exception as e:
            errors.append(f"query device: {e}")
        for sr in rates:
            stream = None
            try:
                stream = sd.InputStream(device=device, samplerate=sr, channels=1, dtype="float32",
                                        blocksize=max(1, int(sr * BLOCK_DURATION)),
                                        callback=self._audio_callback)
                stream.start()
                self.stream, self.stream_sr, self.stream_running = stream, sr, True
                if sr != SAMPLING_RATE:
                    print(f"Note: microphone runs at {sr} Hz; resampling to 16 kHz.", file=sys.stderr)
                break
            except Exception as e:
                errors.append(f"{sr} Hz: {e}")
                # A stream that was created but failed to start may still hold
                # the device; close it before trying the next rate.
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
        if self.stream is None:
            print("Fatal: could not open the microphone.\n  " + "\n  ".join(errors) +
                  "\nCheck microphone permissions, or set \"input_device\" in config.json.",
                  file=sys.stderr)
            sys.exit(1)
        self.segmenter = UtteranceSegmenter(
            self.stream_sr, BLOCK_DURATION, self.cfg["silence_threshold"],
            self.cfg["pause_seconds"], self.cfg["min_speech_seconds"])

    # ----------------------------------------------------------------- audio

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio thread: keep it fast. Drop-oldest when the queue is full."""
        if not self.focused.is_set() or self.shutdown.is_set():
            return
        if time.monotonic() < self.mute_until:
            return
        chunk = indata[:, 0].copy()
        try:
            self.audio_q.put_nowait(chunk)
        except queue.Full:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_q.put_nowait(chunk)
            except queue.Full:
                pass

    def _drain_queue(self):
        while True:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                return

    def beep(self, kind: str):
        """
        Fire-and-forget. sd.play() opens an output stream, which can take tens
        of ms; beep() is often called while the state lock is held, so the
        playback runs on its own short-lived thread and never delays the
        hotkey or the worker.
        """
        if not self.cfg["beep"]:
            return
        tone = make_beep(kind, BEEP_SAMPLE_RATE)
        # Set the mute window right away (not on the thread) so the mic
        # ignores our own beep even if playback starts a little late.
        self.mute_until = time.monotonic() + len(tone) / BEEP_SAMPLE_RATE + MUTE_AFTER_BEEP

        def play():
            try:
                import sounddevice as sd
                sd.play(tone, BEEP_SAMPLE_RATE)
            except Exception:
                pass  # a missing beep should never break dictation

        threading.Thread(target=play, daemon=True).start()

    # ----------------------------------------------------------------- focus

    def _claude_is_active(self) -> bool:
        if not self.cfg["only_when_claude_focused"]:
            return True
        app = get_foreground_app()
        if app is None:
            if not self._warned_focus_unknown:
                self._warned_focus_unknown = True
                print("Warning: can't detect the active window on this system, so listening is "
                      "always on. (Linux: install xdotool. Or set only_when_claude_focused to false.)",
                      file=sys.stderr)
            return True
        return app_matches(app, self.cfg["target_app_names"])

    def _focus_loop(self):
        while not self.shutdown.is_set():
            self._focus_tick()
            self.shutdown.wait(FOCUS_POLL_SECONDS)

    def _focus_tick(self):
        """One focus poll: start/stop the mic when Claude gains/loses focus."""
        active = self._claude_is_active()
        if active:
            if not self.focused.is_set() and self._resume_stream():
                # Only mark as listening once the mic really restarted;
                # otherwise stay unfocused so the next poll retries.
                self.focused.set()
                print("● Claude is active — " + ("say \"start mic\"" if self.state == WAITING
                                                 else "still dictating") + ".")
            return
        was_focused = self.focused.is_set()
        self.focused.clear()
        if was_focused:
            self._drain_queue()
            # The segmenter belongs to the worker thread; ask it to reset
            # rather than touching its buffers from here.
            self._segmenter_reset.set()
        # Also covers startup: open_stream() starts the mic, and if Claude
        # isn't in front yet it must be switched off right away.
        if self.stream_running:
            self._pause_stream()
            if was_focused:
                print("○ Claude not active — microphone paused.")

    def _pause_stream(self):
        # Stopping the stream (not just ignoring audio) turns off the OS
        # microphone indicator while you're in other apps.
        if self.stream is not None and self.stream_running:
            try:
                self.stream.stop()
                self.stream_running = False
            except Exception:
                pass

    def _resume_stream(self) -> bool:
        """Start the mic if stopped. Returns True if it is running afterwards."""
        if self.stream is None:
            return False
        if self.stream_running:
            return True
        try:
            self.stream.start()
            self.stream_running = True
            self._warned_resume = False
            return True
        except Exception as e:
            if not self._warned_resume:  # retried every poll; don't spam the console
                self._warned_resume = True
                print(f"Warning: could not restart microphone ({e}); retrying...", file=sys.stderr)
            return False

    # ----------------------------------------------------------------- state

    def _set_state(self, new_state: str, reason: str = ""):
        with self.lock:
            self.state = new_state
            self.generation += 1
            self.parts = []
            self.retry_pending = False  # any state change ends a pending retry
            self.last_activity = time.monotonic()
        if reason:
            print(reason)

    def toggle_hotkey(self):
        with self.lock:
            if self.state == WAITING:
                self._set_state(DICTATING, "🎙  Dictating (hotkey) — say \"enter\" to send.")
                self.beep("start")
            else:
                had_text = " ".join(self.parts).strip()
                if had_text:
                    print(f"(discarded: \"{had_text}\")")
                if self.cfg["wake_word_enabled"]:
                    self._set_state(WAITING, "⏸  Stopped. Say \"start mic\" to dictate again.")
                else:
                    self._set_state(DICTATING, "↺  Cleared.")
                self.beep("cancel")

    def _check_timeout(self):
        timeout = self.cfg["dictation_timeout_seconds"]
        if not timeout or not self.cfg["wake_word_enabled"]:
            return
        with self.lock:
            if self.state != DICTATING:
                return
            if self.retry_pending:
                # A kept message (failed send) waits until the user says
                # "enter"/"cancel" or presses the hotkey. The failure usually
                # happened because they left Claude, so a 60 s clock running
                # while they're away would quietly throw it away.
                return
            if self.segmenter and self.segmenter.in_speech:
                return
            if time.monotonic() - self.last_activity < timeout:
                return
            leftover = " ".join(self.parts).strip()
        if leftover:
            print(f"(timed out, not sent: \"{leftover}\")")
        self._set_state(WAITING, f"⏸  No speech for {int(timeout)}s — back to waiting for \"start mic\".")
        self.beep("cancel")

    # ------------------------------------------------------------ processing

    def _transcribe(self, audio: np.ndarray, quick: bool) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg["language"],
            beam_size=1 if quick else 5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def _worker_loop(self):
        while not self.shutdown.is_set():
            # Checked every iteration: while Claude is focused, silent chunks keep
            # arriving, so the queue is rarely empty long enough to rely on that.
            self._check_timeout()
            if self._segmenter_reset.is_set():
                self._segmenter_reset.clear()
                self.segmenter.reset()
            try:
                chunk = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.focused.is_set():
                # A chunk can slip in after the focus thread drained the queue
                # (the audio callback checked `focused` just before it was
                # cleared). Drop it so it can't leak into the next utterance.
                self._drain_queue()
                continue
            result = self.segmenter.feed(chunk)
            if result is None:
                continue
            audio, peak = result
            audio = resample_to_16k(audio, self.stream_sr)
            with self.lock:
                gen, state = self.generation, self.state
            try:
                text = self._transcribe(audio, quick=(state == WAITING))
            except Exception as e:
                print(f"Warning: transcription failed: {e}", file=sys.stderr)
                continue
            if not text or is_probable_hallucination(text, peak, self.cfg["silence_threshold"]):
                continue
            # No focus check here on purpose: this audio was recorded while
            # Claude was active. If the user switched away during the decode,
            # _deliver notices and keeps the text for a retry; dropping it here
            # would silently lose the message.
            if self.shutdown.is_set():
                continue
            try:
                self.handle_text(text, gen)
            except Exception as e:
                # Last-resort barrier: nothing in text handling may kill the
                # worker thread, or the app would silently stop listening.
                print(f"Warning: unexpected error handling speech: {e}", file=sys.stderr)

    def handle_text(self, text: str, gen: Optional[int] = None):
        """
        Route one transcribed utterance. State changes happen under the lock;
        the slow part (focus check, paste, Enter) runs after releasing it so
        the hotkey stays responsive.
        """
        with self.lock:
            # State changed (hotkey, timeout) during the slow decode: drop this
            # result instead of acting on a stale utterance.
            if gen is not None and gen != self.generation:
                return
            message = self._route(text)
            gen_after = self.generation
        if message:
            self._deliver(message, gen_after)

    def _route(self, text: str) -> Optional[str]:
        """Update state for one utterance. Returns a message to send, if any. Caller holds lock."""
        if self.state == WAITING:
            if not self.cfg["wake_word_enabled"]:
                return None
            found, rest = detect_wake_phrase(text, self.cfg["wake_phrases"])
            if not found:
                # Shown so users can spot how Whisper mishears their wake
                # phrase and add that spelling to config.json.
                print(f"   (heard: \"{text}\")")
                return None
            self._set_state(DICTATING, "🎙  Listening — speak, then say \"enter\" to send.")
            self.beep("start")
            return self._handle_dictation(rest) if rest else None
        if not self.parts and self.cfg["wake_word_enabled"]:
            # Already listening, nothing said yet, and the user says "start
            # mic" again (e.g. they missed the beep). Treat it as the wake
            # phrase, not as the first words of the message.
            found, rest = detect_wake_phrase(text, self.cfg["wake_phrases"])
            if found:
                if not rest:
                    print("   (already listening)")
                    self.last_activity = time.monotonic()
                    return None
                text = rest
        return self._handle_dictation(text)

    def _deliver(self, message: str, gen_after: Optional[int] = None):
        """
        Send a finished message to Claude. Called WITHOUT the state lock held
        (the paste is slow); _keep_for_retry briefly re-acquires it.
        """
        if self.shutdown.is_set():
            # Ctrl+C after the message was recognized: don't start typing into
            # Claude while the app is quitting. (Checked here, before the
            # clipboard transaction; an already-started send is allowed to
            # finish so the clipboard/keyboard are never left half-done.)
            print(f"Not sent (shutting down). Your text:\n  {message}", file=sys.stderr)
            return
        if not self._claude_is_active():
            self._keep_for_retry(message, gen_after, "Claude isn't the active window")
            return
        try:
            self.send_to_claude(message)
        except ShuttingDown:
            print(f"Not sent (shutting down). Your text:\n  {message}", file=sys.stderr)
            return
        except PastedNotSubmitted as e:
            # The text is already in Claude's message box. Retrying would
            # paste it a second time, so tell the user instead.
            print(f"Warning: your message was pasted but not submitted ({e.__cause__}). "
                  f"It's in Claude's message box — switch to Claude and press Enter to send it.",
                  file=sys.stderr)
            self.beep("cancel")
            return
        except Exception as e:
            # Failed before anything was pasted (e.g. clipboard busy, macOS
            # Accessibility not granted): safe to retry.
            self._keep_for_retry(message, gen_after, f"could not send the message ({e})")
            return
        print(f"➤  Sent: {message}")
        self.beep("sent")

    def _keep_for_retry(self, message: str, gen_after: Optional[int], why: str):
        """
        State was reset optimistically when "enter" was heard. If sending then
        fails, put the text back into dictation so the user can simply say
        "enter" again (or "cancel"), instead of re-dictating everything.
        Skipped if the user already did something else (hotkey, new message).
        """
        with self.lock:
            keep = gen_after is None or gen_after == self.generation
            if keep:
                self._set_state(DICTATING)
                # _set_state just cleared parts; put the message back so the
                # next "enter" sends it (extra words spoken now are appended).
                self.parts = [message]
                self.retry_pending = True
        # One message either way, so the console never says "kept" and
        # then contradicts itself. The text is always shown so it isn't lost.
        if keep:
            print(f"Warning: {why}; nothing was sent yet.\n  ↺  Kept: \"{message}\" — "
                  f"say \"enter\" to try again, or \"cancel\" to discard.", file=sys.stderr)
        else:
            print(f"Warning: {why}; nothing was sent. Your text:\n  {message}", file=sys.stderr)
        self.beep("cancel")

    def _handle_dictation(self, text: str) -> Optional[str]:
        self.parts.append(text)
        self.last_activity = time.monotonic()
        full = " ".join(self.parts).strip()

        is_clear, _ = check_clear_trigger(full, self.clear, self.submit)
        if is_clear:
            print(f"✖  Cancelled: \"{full}\"")
            self._after_message()
            self.beep("cancel")
            return None

        is_submit, body = check_trailing_trigger(full, self.submit)
        if not is_submit:
            print(f"   … {full}")
            return None

        # "enter enter" (a repeated or echoed trigger) must not send the word
        # "enter". Only a body that is NOTHING but a trigger is dropped;
        # "Press enter, enter" still sends "Press enter", as documented.
        only_trigger, rest = check_trailing_trigger(body, self.submit)
        if only_trigger and not rest:
            body = ""

        message = process_edit_commands(body, self.edit)
        # Reset state now (under the lock); the actual send happens in _deliver.
        self._after_message()
        if not message:
            print("(nothing to send)")
            return None
        return message

    def _after_message(self):
        if self.cfg["wake_word_enabled"]:
            self._set_state(WAITING, "⏸  Say \"start mic\" for your next message.")
        else:
            self._set_state(DICTATING)

    # ------------------------------------------------------------------ send

    def send_to_claude(self, message: str):
        """Paste via clipboard (handles any characters/keyboard layout), then press Enter."""
        import pyperclip
        from pynput.keyboard import Controller, Key

        if self._kb is None:
            self._kb = Controller()
        # The clipboard lock makes "copy -> paste -> Enter" one transaction
        # against the delayed restore(), so a restore firing at the same
        # instant can never swap the clipboard between our copy and paste.
        with self._clip_lock:
            # Second shutdown check, now that we hold the lock: closes the gap
            # between _deliver's check and the start of the transaction.
            if self.shutdown.is_set():
                raise ShuttingDown()
            try:
                before = pyperclip.paste()
            except Exception:
                before = None
            pending = self._restore_timer
            previous = self._plan_restore(pending, before)

            # --- Nothing has reached Claude until the paste keystroke, so any
            # failure in this block is a plain exception = safe to retry. The
            # pending restore timer is deliberately left untouched here, so a
            # failure can't lose the user's original clipboard.
            pyperclip.copy(message)
            time.sleep(0.08)
            if not self._claude_is_active():
                # Focus moved during the short wait. Put the clipboard back
                # exactly as it was, so the untouched pending timer stays
                # consistent, and let _deliver keep the text for retry.
                if before is not None:
                    try:
                        pyperclip.copy(before)
                    except Exception:
                        pass
                raise RuntimeError("Claude lost focus before pasting")

            # --- Commit: from here on our own restore takes over.
            if pending is not None:
                pending.cancel()
                if self._restore_timer is pending:
                    self._restore_timer = None
            self._paste_and_submit(pyperclip, Key, previous)
            # Housekeeping after a successful send must never turn it into
            # a "failed, retry" (which would paste the text twice).
            try:
                self._schedule_restore(pyperclip, message, previous)
            except Exception as e:
                self._restore_timer = None
                print(f"Warning: could not schedule clipboard restore ({e}).", file=sys.stderr)

    def _plan_restore(self, pending, current: Optional[str]) -> Optional[str]:
        """
        Decide which clipboard value to restore after this send (no side
        effects). If a restore from the previous message is still pending and
        the clipboard still holds that message, the *original* it was going
        to restore is the real user clipboard. If the timer already fired, or
        the user copied something new since, the current clipboard is the one
        to protect.
        """
        if not self.cfg["restore_clipboard"]:
            return None
        if pending is not None and not pending.finished.is_set() \
                and current is not None and current == getattr(pending, "pasted_message", None):
            return getattr(pending, "original_clipboard", None)
        return current

    def _paste_and_submit(self, pyperclip, Key, previous: Optional[str]):
        """
        Press Ctrl/Cmd+V then Enter. ANY failure from the first keystroke on
        raises PastedNotSubmitted: once V may have been pressed, the text may
        already be in Claude's box, and a retry could paste it twice. Being
        conservative here costs only the (rare) chance to retry by voice.
        """
        mod = Key.cmd if platform.system() == "Darwin" else Key.ctrl
        try:
            with self._kb.pressed(mod):
                self._kb.press("v")
                self._kb.release("v")
            time.sleep(0.25)  # let the app finish pasting before submitting
            if not self._claude_is_active():
                # The paste went to Claude (focus was checked just before it),
                # but Enter would now go to some other app. Stop here.
                raise RuntimeError("Claude lost focus before Enter")
            self._kb.press(Key.enter)
            self._kb.release(Key.enter)
        except Exception as e:
            # Best effort: never leave V, Enter or the modifier held down if a
            # press() worked but a release() (or a half-finished press)
            # failed. Releasing a key that isn't down is harmless.
            for key in ("v", Key.enter, mod):
                try:
                    self._kb.release(key)
                except Exception:
                    pass
            # Known, accepted edge: if press() worked and only the first
            # release() failed, this second release may complete the keypress
            # and Claude may already have submitted. We still report "press
            # Enter yourself"; the worst case is a harmless empty Enter, which
            # beats silently assuming success.
            # The paste may already have gone through, so this must not be
            # retried as a full send. Put the user's own clipboard back now;
            # the delayed restore won't be scheduled on this path.
            if previous is not None and self.cfg["restore_clipboard"]:
                try:
                    pyperclip.copy(previous)
                except Exception:
                    pass
            raise PastedNotSubmitted() from e

    def _schedule_restore(self, pyperclip, message: str, previous: Optional[str]):
        if previous is not None and self.cfg["restore_clipboard"]:
            # Restore after a delay so the paste above has definitely read the
            # clipboard — and only if the clipboard still holds our message,
            # so we never overwrite something the user copied meanwhile.
            def restore():
                with self._clip_lock:
                    try:
                        if pyperclip.paste() == message:
                            pyperclip.copy(previous)
                    except Exception:
                        pass
                    finally:
                        # A fired timer must not be mistaken for a pending one
                        # on the next send.
                        if self._restore_timer is timer:
                            self._restore_timer = None
            timer = threading.Timer(0.6, restore)
            timer.daemon = True
            timer.original_clipboard = previous
            timer.pasted_message = message
            self._restore_timer = timer
            timer.start()

    # ------------------------------------------------------------------- run

    def run(self):
        from pynput import keyboard

        self.load_model()
        self.open_stream()
        # One synchronous focus check right away: switches the mic straight
        # back off if Claude isn't in front (instead of up to one poll later),
        # and sets `focused` when focus gating is disabled.
        self._focus_tick()

        print("\n" + "=" * 60)
        print(f" Claude Voice Commander {__version__}")
        if self.cfg["wake_word_enabled"]:
            print(" Say \"start mic\" → speak → say \"enter\" to send.")
        else:
            print(" Speak → say \"enter\" to send.")
        print(" Say \"cancel\" to discard, \"edit X to Y\" to fix a word.")
        print(f" Hotkey {self.cfg['hotkey']} starts/stops dictation. Ctrl+C quits.")
        if self.cfg["only_when_claude_focused"]:
            print(" Listens only while the Claude app is the active window.")
        print("=" * 60 + "\n")

        threading.Thread(target=self._focus_loop, daemon=True).start()
        threading.Thread(target=self._worker_loop, daemon=True).start()

        signal.signal(signal.SIGINT, lambda *_: self.shutdown.set())
        hotkeys = None
        try:
            try:
                hotkeys = keyboard.GlobalHotKeys({self.cfg["hotkey"]: self.toggle_hotkey})
                hotkeys.start()
            except Exception as e:
                # Voice control still works without the hotkey (e.g. macOS
                # Input Monitoring not granted), so warn instead of quitting.
                print(f"Warning: hotkey {self.cfg['hotkey']} unavailable ({e}). "
                      f"Voice commands still work.", file=sys.stderr)
            while not self.shutdown.is_set():
                self.shutdown.wait(0.5)
        finally:
            print("\nShutting down...")
            if hotkeys is not None:
                try:
                    hotkeys.stop()
                except Exception:
                    pass
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


def main():
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")
    if sys.version_info >= (3, 15):
        # Usually unreachable (pip install fails first), but gives a clear
        # message if someone got the packages installed some other way.
        print("Warning: Python 3.15+ isn't supported yet by faster-whisper's "
              "dependencies (onnxruntime). If anything fails, use Python 3.13.", file=sys.stderr)
    VoiceCommander(load_and_validate_config()).run()


if __name__ == "__main__":
    main()
