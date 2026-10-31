"""
Unit tests. Run from the project folder:  python -m unittest discover -s tests -v
Needs only numpy; no microphone, speech model or keyboard access required.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import UtteranceSegmenter, make_beep, resample_to_16k  # noqa: E402
from config_loader import DEFAULT_CONFIG, load_and_validate_config, normalize_hotkey  # noqa: E402
from focus import app_matches  # noqa: E402
from text_processing import (  # noqa: E402
    _strip_clause_trailing_punct,
    apply_case_pattern,
    check_clear_trigger,
    check_trailing_trigger,
    detect_wake_phrase,
    is_probable_hallucination,
    process_edit_commands,
)

EDIT = ["edit"]
SUBMIT = ["enter"]
CLEAR = ["clear", "cancel"]
WAKE = DEFAULT_CONFIG["wake_phrases"]


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **kw)


def edit(text):
    return quiet(process_edit_commands, text, EDIT)


class TestTriggers(unittest.TestCase):
    def test_submit_basic(self):
        self.assertEqual(check_trailing_trigger("What is two plus two enter", SUBMIT),
                         (True, "What is two plus two"))

    def test_submit_keeps_question_mark(self):
        self.assertEqual(check_trailing_trigger("How are you? Enter.", SUBMIT), (True, "How are you?"))

    def test_submit_comma_dropped(self):
        self.assertEqual(check_trailing_trigger("Hello there, enter", SUBMIT), (True, "Hello there"))

    def test_enter_mid_sentence_does_not_submit(self):
        self.assertFalse(check_trailing_trigger("Press enter to continue", SUBMIT)[0])

    def test_hyphenated_reenter_not_submit(self):
        self.assertFalse(check_trailing_trigger("I need to re-enter", SUBMIT)[0])

    def test_enter_alone(self):
        self.assertEqual(check_trailing_trigger("Enter.", SUBMIT), (True, ""))

    def test_clear_and_cancel_enter(self):
        self.assertTrue(check_clear_trigger("never mind cancel", CLEAR, SUBMIT)[0])
        self.assertTrue(check_clear_trigger("never mind, cancel. Enter.", CLEAR, SUBMIT)[0])
        self.assertFalse(check_clear_trigger("send this enter", CLEAR, SUBMIT)[0])


class TestWakePhrase(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(detect_wake_phrase("Start mic.", WAKE), (True, ""))

    def test_variants(self):
        for s in ["start mike", "Start, Mic!", "startmic", "START MICK", "Okay, start mic."]:
            self.assertTrue(detect_wake_phrase(s, WAKE)[0], s)

    def test_remainder_same_breath(self):
        self.assertEqual(detect_wake_phrase("Start mic. what is the time enter", WAKE),
                         (True, "What is the time enter"))

    def test_not_at_start_is_ignored(self):
        self.assertFalse(detect_wake_phrase("I told him we should start mic checks tomorrow", WAKE)[0])

    def test_no_partial_word(self):
        self.assertFalse(detect_wake_phrase("Start Michael's car", WAKE)[0])
        self.assertFalse(detect_wake_phrase("restart mic", WAKE)[0])

    def test_unrelated(self):
        self.assertFalse(detect_wake_phrase("hello there", WAKE)[0])
        self.assertFalse(detect_wake_phrase("", WAKE)[0])


class TestEditCommands(unittest.TestCase):
    def test_comma_then_question(self):
        self.assertEqual(edit("What is the capital of South Korea, edit South Korea to Russia?"),
                         "What is the capital of Russia?")

    def test_comma_no_punct(self):
        self.assertEqual(edit("Tell me about Java, edit Java to Python"), "Tell me about Python")

    def test_no_separator_punct(self):
        self.assertEqual(edit("What is the capital of South Korea edit South Korea to Russia"),
                         "What is the capital of Russia")

    def test_leading_question_mark_wins(self):
        self.assertEqual(edit("What is the capital of South Korea? Edit South Korea equals to Russia."),
                         "What is the capital of Russia?")

    def test_ellipsis_collapses(self):
        self.assertEqual(edit("Which borders of South Korea are open, edit South Korea to Russia..."),
                         "Which borders of Russia are open.")

    def test_dotted_acronym(self):
        self.assertEqual(edit("I am in the U.S.A. right now, edit U.S.A. to U.K.."),
                         "I am in the U.K. right now.")

    def test_lowercase_replacement_gets_case(self):
        self.assertEqual(edit("I love Paris, edit Paris to london"), "I love London")

    def test_sentence_start_case(self):
        self.assertEqual(edit("Cats are great edit cats to dogs"), "Dogs are great")

    def test_chain_with_vanished_target(self):
        self.assertEqual(edit("The cat sat edit cat to dog edit cat to bird"), "The dog sat")

    def test_two_independent_edits(self):
        self.assertEqual(edit("Book a flight to Paris on Monday edit Paris to Rome edit Monday to Friday"),
                         "Book a flight to Rome on Friday")

    def test_edit_as_prose(self):
        self.assertEqual(edit("Please edit this file to be shorter"), "Please edit this file to be shorter")

    def test_no_edit(self):
        self.assertEqual(edit("Just a normal sentence."), "Just a normal sentence.")

    def test_symbols_target(self):
        self.assertEqual(edit("I write C++ daily, edit C++ to Rust"), "I write Rust daily")

    def test_strip_trailing(self):
        self.assertEqual(_strip_clause_trailing_punct("U.K.."), ("U.K.", "."))
        self.assertEqual(_strip_clause_trailing_punct("Russia..."), ("Russia", "..."))
        self.assertEqual(_strip_clause_trailing_punct("."), ("", "."))

    def test_case_pattern(self):
        self.assertEqual(apply_case_pattern("NASA", "esa", False), "ESA")
        self.assertEqual(apply_case_pattern("New York", "los angeles", False), "Los Angeles")
        self.assertEqual(apply_case_pattern("dog", "cat", False), "cat")


class TestHallucination(unittest.TestCase):
    def test_quiet_thank_you_dropped(self):
        self.assertTrue(is_probable_hallucination("Thank you.", 0.01, 0.012))

    def test_loud_thank_you_kept(self):
        self.assertFalse(is_probable_hallucination("Thank you.", 0.2, 0.012))


class TestConfig(unittest.TestCase):
    def _load(self, data):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            with open(p, "w") as f:
                f.write(data if isinstance(data, str) else json.dumps(data))
            return quiet(load_and_validate_config, p)

    def test_defaults_when_missing(self):
        cfg = quiet(load_and_validate_config, "/nonexistent/config.json")
        self.assertEqual(cfg["hotkey"], "<f4>")
        self.assertIn("start mic", cfg["wake_phrases"])
        self.assertTrue(cfg["only_when_claude_focused"])

    def test_nulls_fall_back(self):
        cfg = self._load({"hotkey": None, "whisper_model": None, "language": None,
                          "wake_phrases": None, "beep": None})
        self.assertEqual(cfg["hotkey"], "<f4>")
        self.assertEqual(cfg["whisper_model"], "base.en")
        self.assertIsNone(cfg["language"])
        self.assertEqual(cfg["wake_phrases"], DEFAULT_CONFIG["wake_phrases"])
        self.assertTrue(cfg["beep"])

    def test_bad_json(self):
        self.assertEqual(self._load("{not json")["whisper_model"], "base.en")

    def test_clamping_and_types(self):
        cfg = self._load({"pause_seconds": 99, "silence_threshold": "abc", "min_speech_seconds": True,
                          "dictation_timeout_seconds": -5, "input_device": [1]})
        self.assertEqual(cfg["pause_seconds"], 5.0)
        self.assertEqual(cfg["silence_threshold"], 0.012)
        self.assertEqual(cfg["min_speech_seconds"], 0.25)
        self.assertEqual(cfg["dictation_timeout_seconds"], 0.0)
        self.assertIsNone(cfg["input_device"])

    def test_trigger_punctuation_cleaned_in_config(self):
        # Round-2 bug 5.
        cfg = self._load({"triggers": {"submit": ["Send it!", "send  it", "enter.", "“over…”"]}})
        self.assertEqual(cfg["triggers"]["submit"], ["send it", "enter", "over"])

    def test_punctuation_only_trigger_warns(self):
        # Round-3 bug 3: no silent fallback.
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            with open(p, "w") as f:
                json.dump({"triggers": {"submit": ["!"]}}, f)
            with contextlib.redirect_stderr(err):
                cfg = load_and_validate_config(p)
        self.assertEqual(cfg["triggers"]["submit"], ["enter"])
        self.assertIn("triggers.submit", err.getvalue())

    def test_custom_wake_phrase(self):
        self.assertEqual(self._load({"wake_phrases": "Hey Claude"})["wake_phrases"], ["hey claude"])

    def test_hotkey_normalize(self):
        self.assertEqual(normalize_hotkey("ctrl+shift+d"), "<ctrl>+<shift>+d")
        self.assertEqual(normalize_hotkey("win+F9"), "<cmd>+<f9>")


class TestFocus(unittest.TestCase):
    def test_matches(self):
        for name in ["Claude", "claude.exe", "Claude.app", "claude"]:
            self.assertTrue(app_matches(name, ["claude"]), name)

    def test_no_match(self):
        for name in ["chrome.exe", "Claudette", None, "", "Terminal"]:
            self.assertFalse(app_matches(name, ["claude"]), name)


class TestAudio(unittest.TestCase):
    def test_resample_length(self):
        out = resample_to_16k(np.zeros(48000, np.float32), 48000)
        self.assertEqual(out.size, 16000)

    def test_resample_rejects_bad_rate(self):
        # Round-6 item 4.
        for bad in [0, -48000, float("nan"), float("inf"), float("-inf"), None, True, "48000"]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                resample_to_16k(np.zeros(4800, np.float32), bad)
        for good in [np.int64(48000), np.float32(48000), np.float64(48000), 48000.0]:
            self.assertEqual(resample_to_16k(np.zeros(4800, np.float32), good).size, 1600, repr(good))

    def test_resample_tiny_buffer(self):
        self.assertEqual(resample_to_16k(np.zeros(5, np.float32), 48000).size, 5)

    def test_segmenter(self):
        seg = UtteranceSegmenter(16000, 0.1, 0.01, pause_seconds=0.3, min_speech_seconds=0.2)
        loud = np.full(1600, 0.1, np.float32)
        quiet_chunk = np.zeros(1600, np.float32)
        results = [seg.feed(c) for c in [quiet_chunk, loud, loud, loud, quiet_chunk, quiet_chunk, quiet_chunk]]
        done = [r for r in results if r is not None]
        self.assertEqual(len(done), 1)
        audio, peak = done[0]
        self.assertEqual(audio.size, 7 * 1600)  # 1 pre-roll + 3 loud + 3 quiet
        self.assertAlmostEqual(peak, 0.1, places=5)

    def test_segmenter_drops_click(self):
        seg = UtteranceSegmenter(16000, 0.1, 0.01, pause_seconds=0.2, min_speech_seconds=0.3)
        loud = np.full(1600, 0.1, np.float32)
        z = np.zeros(1600, np.float32)
        self.assertTrue(all(seg.feed(c) is None for c in [loud, z, z, z]))

    def test_beeps(self):
        for kind in ["start", "sent", "cancel"]:
            b = make_beep(kind)
            self.assertGreater(b.size, 0)
            self.assertLessEqual(float(np.abs(b).max()), 0.21)


class TestFlow(unittest.TestCase):
    """The full wake -> dictate -> enter -> wait loop, with sending stubbed out."""

    def setUp(self):
        import main
        cfg = quiet(load_and_validate_config, "/nonexistent/config.json")
        self.vc = main.VoiceCommander(cfg)
        self.sent = []
        self.vc.send_to_claude = self.sent.append
        self.vc.beep = lambda kind: None
        self.vc._claude_is_active = lambda: True
        self.main = main

    def say(self, text):
        quiet(self.vc.handle_text, text)

    def test_ignores_speech_until_wake(self):
        self.say("hello there enter")
        self.assertEqual(self.sent, [])
        self.assertEqual(self.vc.state, self.main.WAITING)

    def test_wake_then_dictate_then_enter(self):
        self.say("Start mic.")
        self.assertEqual(self.vc.state, self.main.DICTATING)
        self.say("What is the capital of France?")
        self.say("Enter.")
        self.assertEqual(self.sent, ["What is the capital of France?"])
        self.assertEqual(self.vc.state, self.main.WAITING)

    def test_second_message_needs_wake_again(self):
        self.say("start mic")
        self.say("first question enter")
        self.say("second question enter")       # no wake phrase -> ignored
        self.say("start mic second question enter")
        self.assertEqual(self.sent, ["first question", "Second question"])

    def test_cancel(self):
        self.say("start mic")
        self.say("something wrong cancel")
        self.assertEqual(self.sent, [])
        self.assertEqual(self.vc.state, self.main.WAITING)

    def test_edit_across_pauses(self):
        self.say("start mic")
        self.say("What is the capital of South Korea,")
        self.say("edit South Korea to Russia? Enter.")
        self.assertEqual(self.sent, ["What is the capital of Russia?"])

    def test_not_sent_when_claude_inactive(self):
        self.vc._claude_is_active = lambda: False
        self.say("start mic hello enter")
        self.assertEqual(self.sent, [])

    def test_hotkey_starts_and_stops(self):
        quiet(self.vc.toggle_hotkey)
        self.assertEqual(self.vc.state, self.main.DICTATING)
        quiet(self.vc.toggle_hotkey)
        self.assertEqual(self.vc.state, self.main.WAITING)

    def test_wake_disabled_stays_dictating(self):
        self.vc.cfg = dict(self.vc.cfg, wake_word_enabled=False)
        self.vc.state = self.main.DICTATING
        self.say("one enter")
        self.say("two enter")
        self.assertEqual(self.sent, ["one", "two"])

    def test_send_failure_keeps_app_working(self):
        # Review bug 3: a paste/keyboard error must not break later messages.
        def boom(msg):
            raise RuntimeError("Accessibility not granted")
        self.vc.send_to_claude = boom
        self.say("start mic hello enter")
        # Round-2 bug 4: the text is kept, so just saying "enter" retries.
        self.assertEqual(self.vc.state, self.main.DICTATING)
        self.assertEqual(self.vc.parts, ["Hello"])
        self.vc.send_to_claude = self.sent.append
        self.say("enter")
        self.assertEqual(self.sent, ["Hello"])
        self.assertEqual(self.vc.state, self.main.WAITING)

    def test_failed_send_when_inactive_can_be_retried(self):
        active = {"v": False}
        self.vc._claude_is_active = lambda: active["v"]
        self.say("start mic what time is it enter")
        self.assertEqual(self.sent, [])
        active["v"] = True
        self.say("enter")
        self.assertEqual(self.sent, ["What time is it"])

    def test_repeated_wake_phrase_ignored(self):
        # Live test: user said "start mic" twice; it ended up in the message.
        self.say("start mic")
        self.say("start mic")
        self.say("start mic hello enter")
        self.assertEqual(self.sent, ["Hello"])

    def test_wake_phrase_later_in_message_kept(self):
        self.say("start mic")
        self.say("I said")
        self.say("start mic enter")
        self.assertEqual(self.sent, ["I said start mic"])

    def test_enter_enter_sends_nothing(self):
        self.say("start mic enter enter")
        self.assertEqual(self.sent, [])
        self.assertEqual(self.vc.state, self.main.WAITING)

    def test_press_enter_enter_still_sends(self):
        self.say("start mic press enter, enter")
        self.assertEqual(self.sent, ["Press enter"])

    def test_retry_skipped_if_user_moved_on(self):
        # Round-3 gap: generation advanced before the failed send returned.
        self.vc._set_state(self.main.WAITING)
        stale_gen = self.vc.generation - 1
        quiet(self.vc._keep_for_retry, "old text", stale_gen, "test")
        self.assertEqual(self.vc.state, self.main.WAITING)
        self.assertEqual(self.vc.parts, [])

    def _expire_timeout(self):
        self.vc.last_activity -= self.vc.cfg["dictation_timeout_seconds"] + 1
        quiet(self.vc._check_timeout)

    def test_retry_text_survives_timeout(self):
        # Round-4 bug 2: kept text waits for the user, however long they're away.
        self.vc._claude_is_active = lambda: False
        self.say("start mic hello enter")
        self._expire_timeout()
        self.assertEqual(self.vc.state, self.main.DICTATING)
        self.assertEqual(self.vc.parts, ["Hello"])
        self.vc._claude_is_active = lambda: True
        self.say("enter")
        self.assertEqual(self.sent, ["Hello"])

    def test_normal_dictation_still_times_out(self):
        self.say("start mic half a thought")
        self._expire_timeout()
        self.assertEqual(self.vc.state, self.main.WAITING)
        self.assertEqual(self.vc.parts, [])

    def test_retry_ends_on_hotkey(self):
        self.vc._claude_is_active = lambda: False
        self.say("start mic hello enter")
        quiet(self.vc.toggle_hotkey)
        self.assertFalse(self.vc.retry_pending)
        self.assertEqual(self.vc.state, self.main.WAITING)

    def test_retry_with_wake_word_disabled(self):
        # No timeout in this mode at all; the kept text waits for "enter".
        self.vc.cfg = dict(self.vc.cfg, wake_word_enabled=False)
        self.vc.state = self.main.DICTATING
        self.vc._claude_is_active = lambda: False
        self.say("hello enter")
        self._expire_timeout()
        self.assertEqual(self.vc.parts, ["hello"])
        self.vc._claude_is_active = lambda: True
        self.say("enter")
        self.assertEqual(self.sent, ["hello"])

    def test_retry_with_extra_words(self):
        # Intended: words spoken in retry mode are appended to the kept text.
        self.vc._claude_is_active = lambda: False
        self.say("start mic book a table")
        self.say("enter")
        self.vc._claude_is_active = lambda: True
        self.say("for two enter")
        self.assertEqual(self.sent, ["Book a table for two"])

    def test_failed_send_retry_can_be_cancelled(self):
        self.vc._claude_is_active = lambda: False
        self.say("start mic hello enter")
        self.vc._claude_is_active = lambda: True
        self.say("cancel")
        self.assertEqual(self.sent, [])
        self.assertEqual(self.vc.state, self.main.WAITING)

    def test_beep_does_not_block_under_lock(self):
        # Round-2 bug 3: playback runs on its own thread.
        import threading
        import types
        release = threading.Event()
        played = []

        def slow_play(tone, sr):
            release.wait(2)
            played.append(sr)

        saved = sys.modules.get("sounddevice")
        sys.modules["sounddevice"] = types.SimpleNamespace(play=slow_play)
        try:
            vc = self.main.VoiceCommander(self.vc.cfg)
            start = time.monotonic()
            with vc.lock:
                vc.beep("start")
            self.assertLess(time.monotonic() - start, 0.5)
            release.set()
            time.sleep(0.2)
            self.assertEqual(played, [44100])
        finally:
            if saved is None:
                sys.modules.pop("sounddevice", None)
            else:
                sys.modules["sounddevice"] = saved

    def test_stale_generation_ignored(self):
        self.say("start mic")
        old_gen = self.vc.generation
        quiet(self.vc.toggle_hotkey)  # user stopped dictation during the decode
        quiet(self.vc.handle_text, "start mic stale enter", old_gen)
        self.assertEqual(self.sent, [])

    def test_send_runs_without_lock(self):
        # Review bug 10: the hotkey must not wait for the paste to finish.
        held = []
        self.vc.send_to_claude = lambda m: held.append(self.vc.lock._is_owned())
        self.say("start mic hi enter")
        self.assertEqual(held, [False])


class FakeStream:
    def __init__(self, fail_start_times=0):
        self.fail_start_times = fail_start_times
        self.started = 0
        self.closed = False

    def start(self):
        if self.fail_start_times > 0:
            self.fail_start_times -= 1
            raise RuntimeError("device busy")
        self.started += 1

    def stop(self):
        pass

    def close(self):
        self.closed = True


class TestRuntime(unittest.TestCase):
    """Runtime paths, with sounddevice / pyperclip / pynput replaced by fakes."""

    def setUp(self):
        import main
        self.main = main
        cfg = quiet(load_and_validate_config, "/nonexistent/config.json")
        self.vc = main.VoiceCommander(cfg)
        self.vc.beep = lambda kind: None
        self.vc._claude_is_active = lambda: True   # send_to_claude re-checks focus
        self._saved = {k: sys.modules.get(k) for k in ["sounddevice", "pyperclip", "pynput", "pynput.keyboard"]}

    def _focus_sequence(self, *values):
        """_claude_is_active returns these in order, then the last one forever."""
        seq = list(values)
        self.vc._claude_is_active = lambda: seq.pop(0) if len(seq) > 1 else seq[0]

    def test_shutdown_during_transcription_not_delivered(self):
        import threading

        class Seg:
            in_speech = False
            def __init__(self):
                self.done = False
            def feed(self, c):
                if self.done:
                    return None
                self.done = True
                return np.zeros(1600, np.float32), 0.5
            def reset(self):
                pass

        vc = self.vc
        vc.segmenter = Seg()
        vc.focused.set()
        delivered = []
        vc.handle_text = lambda *a: delivered.append(a)

        def transcribe(audio, quick):
            vc.shutdown.set()        # Ctrl+C while decoding
            return "start mic hello enter"

        vc._transcribe = transcribe
        vc.audio_q.put(np.zeros(1600, np.float32))
        t = threading.Thread(target=lambda: quiet(vc._worker_loop), daemon=True)
        t.start()
        t.join(2)
        self.assertEqual(delivered, [])

    def test_shutdown_before_delivery_does_not_send(self):
        # Round-8 bug 2: Ctrl+C between recognition and the paste.
        sends = []
        self.vc.send_to_claude = sends.append
        self.vc.shutdown.set()
        quiet(self.vc._deliver, "hello", self.vc.generation)
        self.assertEqual(sends, [])

    def test_focus_detection_unavailable_fails_open_and_warns_once(self):
        # e.g. Wayland / no xdotool: listen everywhere, warn exactly once.
        import main as m
        vc = m.VoiceCommander(self.vc.cfg)
        saved = m.get_foreground_app
        m.get_foreground_app = lambda: None
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                self.assertTrue(vc._claude_is_active())
                self.assertTrue(vc._claude_is_active())
        finally:
            m.get_foreground_app = saved
        self.assertEqual(err.getvalue().count("can't detect the active window"), 1)

    def test_focus_lost_before_paste_kept_for_retry(self):
        # Round-6 bug 1: checked in _deliver, then lost before Ctrl+V.
        clip, pasted = self._fake_input_libs()
        self._focus_sequence(True, False)
        quiet(self.vc.handle_text, "start mic hello enter")
        self.assertEqual(pasted, [])
        self.assertEqual(clip["value"], "ORIGINAL")
        self.assertTrue(self.vc.retry_pending)
        self.assertEqual(self.vc.parts, ["Hello"])

    def test_focus_lost_before_enter_not_submitted(self):
        # Round-6 bug 1: pasted into Claude, then focus moved -> no Enter
        # into another app, no retry.
        clip, pasted = self._fake_input_libs()
        enters = []
        Controller = sys.modules["pynput.keyboard"].Controller
        orig_press = Controller.press

        def press(self_, key):
            if key == "enter":
                enters.append(key)
            return orig_press(self_, key)

        Controller.press = press
        try:
            self._focus_sequence(True, True, False)
            quiet(self.vc.handle_text, "start mic hello enter")
        finally:
            Controller.press = orig_press  # don't leak the patch into other tests
        self.assertEqual(pasted, ["Hello"])
        self.assertEqual(enters, [])
        self.assertFalse(self.vc.retry_pending)
        self.assertEqual(self.vc.state, self.main.WAITING)
        self.assertEqual(clip["value"], "ORIGINAL")

    def test_restore_scheduling_failure_is_not_a_failed_send(self):
        # Round-6 bug 2: housekeeping after a successful send must not retry.
        self._fake_input_libs()
        def broken(*a, **kw):
            raise RuntimeError("can't start thread")
        self.vc._schedule_restore = broken
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.vc.handle_text("start mic hello enter")
        self.assertIn("Sent: Hello", out.getvalue())
        self.assertFalse(self.vc.retry_pending)

    def test_copy_failure_keeps_pending_restore(self):
        # Round-6 bug 2 (takeover): a failed copy must not cancel the pending
        # restore, or the user's original clipboard would be lost.
        clip, _ = self._fake_input_libs()
        self.vc.send_to_claude("first")          # restore of ORIGINAL pending
        good_copy = sys.modules["pyperclip"].copy
        def bad_copy(v):
            raise RuntimeError("clipboard busy")
        sys.modules["pyperclip"].copy = bad_copy
        with self.assertRaises(RuntimeError):
            self.vc.send_to_claude("second")
        sys.modules["pyperclip"].copy = good_copy
        time.sleep(0.9)
        self.assertEqual(clip["value"], "ORIGINAL")

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_open_stream_closes_failed_stream(self):
        # Review bug 7.
        import types
        made = []

        def InputStream(samplerate, **kw):
            s = FakeStream(fail_start_times=1 if samplerate == 16000 else 0)
            s.sr = samplerate
            made.append(s)
            return s

        sd = types.SimpleNamespace(InputStream=InputStream,
                                   query_devices=lambda d, kind: {"default_samplerate": 48000})
        sys.modules["sounddevice"] = sd
        quiet(self.vc.open_stream)
        self.assertTrue(made[0].closed)
        self.assertEqual(self.vc.stream_sr, 48000)
        self.assertFalse(made[1].closed)

    def test_resume_failure_retries(self):
        # Review bug 9: if the mic fails to restart, stay unfocused and retry.
        self.vc.stream = FakeStream(fail_start_times=1)
        self.vc.stream_running = False
        self.vc._claude_is_active = lambda: True
        quiet(self.vc._focus_tick)
        self.assertFalse(self.vc.focused.is_set())
        quiet(self.vc._focus_tick)
        self.assertTrue(self.vc.focused.is_set())

    def test_mic_off_at_startup_when_claude_not_active(self):
        self.vc.stream = FakeStream()
        self.vc.stream_running = True
        self.vc._claude_is_active = lambda: False
        quiet(self.vc._focus_tick)
        self.assertFalse(self.vc.stream_running)

    def test_chunk_after_focus_loss_dropped(self):
        # Review bug 2: audio queued while unfocused never reaches the segmenter.
        import threading
        fed = []

        class Seg:
            in_speech = False
            def feed(self, c):
                fed.append(c)
            def reset(self):
                pass

        self.vc.segmenter = Seg()
        self.vc.focused.clear()
        self.vc.audio_q.put(np.zeros(1600, np.float32))
        t = threading.Thread(target=self.vc._worker_loop, daemon=True)
        t.start()
        time.sleep(0.4)
        self.vc.shutdown.set()
        t.join(1)
        self.assertEqual(fed, [])
        self.assertTrue(self.vc.audio_q.empty())

    def test_focus_lost_during_transcription_keeps_message(self):
        # Round-3 bug 1: message decoded after the user switched away must be
        # kept for retry, not dropped.
        import threading

        class Seg:
            in_speech = False
            def __init__(self):
                self.done = False
            def feed(self, c):
                if self.done:
                    return None
                self.done = True
                return np.zeros(1600, np.float32), 0.5
            def reset(self):
                pass

        vc = self.vc
        vc.segmenter = Seg()
        vc.focused.set()
        vc._set_state(self.main.DICTATING)

        def slow_transcribe(audio, quick):
            vc.focused.clear()       # user Alt-Tabs away mid-decode
            return "hello enter"

        vc._transcribe = slow_transcribe
        vc._claude_is_active = lambda: False
        vc.audio_q.put(np.zeros(1600, np.float32))
        t = threading.Thread(target=lambda: quiet(vc._worker_loop), daemon=True)
        t.start()
        time.sleep(0.5)
        vc.shutdown.set()
        t.join(1)
        self.assertEqual(vc.state, self.main.DICTATING)
        self.assertEqual(vc.parts, ["hello"])

    def test_enter_failure_after_paste_not_retried(self):
        # Round-3 bug 2: paste worked but Enter failed -> no retry (it would
        # paste twice). Round-4 bug 1: the user's clipboard is restored.
        clip, pasted = self._fake_input_libs(fail_enter=True)
        self.vc._claude_is_active = lambda: True
        quiet(self.vc.handle_text, "start mic hello enter")
        self.assertEqual(pasted, ["Hello"])
        self.assertEqual(self.vc.state, self.main.WAITING)
        self.assertEqual(self.vc.parts, [])
        self.assertEqual(clip["value"], "ORIGINAL")

    def test_clipboard_survives_enter_failure_then_normal_send(self):
        # Round-4 bug 1 repro: later successful send must restore ORIGINAL,
        # not the abandoned message.
        clip, _ = self._fake_input_libs(fail_enter=True)
        with self.assertRaises(self.main.PastedNotSubmitted):
            self.vc.send_to_claude("hello")
        clip_after_fail = clip["value"]
        self._fake_input_libs()  # keyboard works again
        self.vc._kb = None       # drop the cached failing keyboard
        sys.modules["pyperclip"].copy(clip_after_fail)
        self.vc.send_to_claude("world")
        time.sleep(0.9)
        self.assertEqual(sys.modules["pyperclip"].paste(), "ORIGINAL")

    def test_paste_release_failure_not_retried(self):
        # Round-5 bug 1: V pressed, then release("v") fails -> treated as
        # "pasted, not sent": no retry (no double paste), clipboard restored.
        clip, pasted = self._fake_input_libs(fail_release_v=True)
        sent_again = []
        self.vc._claude_is_active = lambda: True
        quiet(self.vc.handle_text, "start mic hello enter")
        self.assertEqual(pasted, ["Hello"])
        self.assertEqual(self.vc.state, self.main.WAITING)
        self.assertFalse(self.vc.retry_pending)
        self.vc.send_to_claude = sent_again.append
        quiet(self.vc.handle_text, "enter")   # not in retry mode -> ignored
        self.assertEqual(sent_again, [])
        self.assertEqual(clip["value"], "ORIGINAL")

    def test_all_keys_released_after_paste_failure(self):
        # Round-9 bug 2: V, Enter and the modifier are all released.
        self._fake_input_libs(fail_release_v=True)
        released = []
        Controller = sys.modules["pynput.keyboard"].Controller
        orig_release = Controller.release

        def release(self_, key):
            released.append(key)
            return orig_release(self_, key)

        Controller.release = release
        try:
            with self.assertRaises(self.main.PastedNotSubmitted):
                self.vc.send_to_claude("hello")
        finally:
            Controller.release = orig_release
        cleanup = released[1:]  # first entry is the failing release("v")
        for key in ["v", "enter"]:
            self.assertIn(key, cleanup)
        self.assertTrue({"ctrl", "cmd"} & set(cleanup))

    def test_modifier_release_failure_is_pasted_not_submitted(self):
        import types
        clip = {"value": "ORIGINAL"}
        sys.modules["pyperclip"] = types.SimpleNamespace(
            paste=lambda: clip["value"], copy=lambda v: clip.__setitem__("value", v))

        class Controller:
            @contextlib.contextmanager
            def pressed(self, key):
                yield
                raise RuntimeError("modifier stuck")
            def press(self, key):
                pass
            def release(self, key):
                pass

        kb = types.SimpleNamespace(Controller=Controller,
                                   Key=types.SimpleNamespace(cmd="cmd", ctrl="ctrl", enter="enter"))
        sys.modules["pynput"] = types.SimpleNamespace(keyboard=kb)
        sys.modules["pynput.keyboard"] = kb
        with self.assertRaises(self.main.PastedNotSubmitted):
            self.vc.send_to_claude("hello")
        self.assertEqual(clip["value"], "ORIGINAL")

    def test_shutdown_inside_send_touches_nothing(self):
        # Round-9 race: shutdown set after _deliver's check but before the
        # clipboard transaction.
        clip, pasted = self._fake_input_libs()
        self.vc.shutdown.set()
        with self.assertRaises(self.main.ShuttingDown):
            self.vc.send_to_claude("hello")
        self.assertEqual(clip["value"], "ORIGINAL")
        self.assertEqual(pasted, [])

    def test_shutdown_race_reported_not_retried(self):
        self._fake_input_libs()
        original_send = self.vc.send_to_claude

        def send_with_late_shutdown(msg):
            self.vc.shutdown.set()   # Ctrl+C lands right after _deliver's check
            return original_send(msg)

        self.vc.send_to_claude = send_with_late_shutdown
        quiet(self.vc.handle_text, "start mic hello enter")
        self.assertFalse(self.vc.retry_pending)

    def test_clipboard_copy_failure_is_retryable(self):
        # Nothing reached Claude, so the normal retry path applies.
        import types
        def bad_copy(v):
            raise RuntimeError("clipboard busy")
        sys.modules["pyperclip"] = types.SimpleNamespace(paste=lambda: "X", copy=bad_copy)
        self._fake_keyboard_only()
        self.vc._claude_is_active = lambda: True
        quiet(self.vc.handle_text, "start mic hello enter")
        self.assertTrue(self.vc.retry_pending)
        self.assertEqual(self.vc.parts, ["Hello"])

    def test_restore_cannot_interleave_with_send(self):
        # Round-5 bug 3: restore() waits for the clipboard lock, so it can't
        # swap the clipboard between our copy and our paste.
        import threading
        clip, pasted = self._fake_input_libs()
        self.vc.send_to_claude("first")
        timer = self.vc._restore_timer
        with self.vc._clip_lock:
            # Simulate the timer firing while a send holds the lock.
            t = threading.Thread(target=timer.function, daemon=True)
            t.start()
            time.sleep(0.1)
            self.assertEqual(clip["value"], "first")   # restore is blocked
            clip["value"] = "second"                   # our next send copies
        t.join(1)
        self.assertEqual(clip["value"], "second")      # restore saw it's not ours

    def _fake_keyboard_only(self):
        import types

        class Controller:
            def pressed(self, key):
                return contextlib.nullcontext()
            def press(self, key):
                pass
            def release(self, key):
                pass

        kb = types.SimpleNamespace(Controller=Controller,
                                   Key=types.SimpleNamespace(cmd="cmd", ctrl="ctrl", enter="enter"))
        sys.modules["pynput"] = types.SimpleNamespace(keyboard=kb)
        sys.modules["pynput.keyboard"] = kb

    def _fake_input_libs(self, fail_enter=False, fail_release_v=False):
        import types
        clip = {"value": "ORIGINAL"}
        pasted = []
        pyperclip = types.SimpleNamespace(paste=lambda: clip["value"],
                                          copy=lambda v: clip.__setitem__("value", v))

        class Controller:
            def pressed(self, key):
                return contextlib.nullcontext()
            def press(self, key):
                if key == "enter" and fail_enter:
                    raise RuntimeError("keyboard blocked")
                if key == "v":
                    pasted.append(clip["value"])
            def release(self, key):
                if key == "v" and fail_release_v:
                    raise RuntimeError("release failed")

        kb = types.SimpleNamespace(Controller=Controller,
                                   Key=types.SimpleNamespace(cmd="cmd", ctrl="ctrl", enter="enter"))
        sys.modules["pyperclip"] = pyperclip
        sys.modules["pynput"] = types.SimpleNamespace(keyboard=kb)
        sys.modules["pynput.keyboard"] = kb
        return clip, pasted

    def test_clipboard_restored_after_rapid_sends(self):
        # Review bug 8: two quick sends each paste their own text, and the
        # user's original clipboard comes back afterwards.
        clip, pasted = self._fake_input_libs()
        self.vc.send_to_claude("first")
        self.vc.send_to_claude("second")
        self.assertEqual(pasted, ["first", "second"])
        time.sleep(0.9)
        self.assertEqual(clip["value"], "ORIGINAL")

    def test_clipboard_fresh_copy_after_restore_is_protected(self):
        # Round-2 bug 1: after the restore timer fired, a new user copy must
        # survive the next send.
        clip, _ = self._fake_input_libs()
        self.vc.send_to_claude("first")
        time.sleep(0.9)                      # timer fires, "ORIGINAL" restored
        self.assertEqual(clip["value"], "ORIGINAL")
        clip["value"] = "B"                  # user copies something new
        self.vc.send_to_claude("second")
        time.sleep(0.9)
        self.assertEqual(clip["value"], "B")

    def test_clipboard_not_restored_over_new_copy(self):
        clip, _ = self._fake_input_libs()
        self.vc.send_to_claude("msg")
        clip["value"] = "user copied this"
        time.sleep(0.9)
        self.assertEqual(clip["value"], "user copied this")


class TestReviewTextFixes(unittest.TestCase):
    def test_overlapping_triggers_prefer_longer(self):
        # Review bug 5.
        self.assertEqual(check_trailing_trigger("send it", ["it", "send it"]), (True, ""))

    def test_trigger_with_punctuation(self):
        # Review bug 6.
        self.assertEqual(check_trailing_trigger("hello send it!", ["send it!"]), (True, "hello"))

    def test_invalid_edit_after_valid_kept(self):
        # Review bug 4.
        self.assertEqual(edit("I like Paris, edit Paris to Rome, please edit this file to be shorter"),
                         "I like Rome, please edit this file to be shorter")

    def test_invalid_edit_before_valid_kept(self):
        self.assertEqual(edit("Please edit this doc. I like Paris edit Paris to Rome"),
                         "Please edit this doc. I like Rome")

    def test_unicode_punctuation_around_triggers(self):
        # Round-8: Whisper's single-character ellipsis etc. must not hide "enter".
        for t, body in [("hello enter…", "hello"), ("hello enter。", "hello"),
                        ("'Tis fine enter", "'Tis fine"), ('He said "hi" enter', 'He said "hi"'),
                        ('hello enter"', "hello"), ("“Hello?” enter", '"Hello?"'),
                        ('"Hello? Enter."', "Hello?"), ("“What now? Enter.”", "What now?"),
                        ("hello ENTER！", "hello"), ("Hello… enter", "Hello...")]:
            self.assertEqual(check_trailing_trigger(t, SUBMIT), (True, body), t)
        self.assertTrue(check_clear_trigger("never mind cancel…", CLEAR, SUBMIT)[0])

    def test_ideographic_space(self):
        self.assertEqual(check_trailing_trigger("hello　enter", SUBMIT), (True, "hello"))
        self.assertTrue(detect_wake_phrase("start　mic", WAKE)[0])

    def test_unicode_ellipsis_in_wake_and_edit(self):
        self.assertEqual(detect_wake_phrase("Start mic… hello", WAKE), (True, "Hello"))
        self.assertEqual(edit("Which borders of South Korea are open, edit South Korea to Russia…"),
                         "Which borders of Russia are open.")

    def test_unicode_trigger_in_config(self):
        cfg_trig = check_trailing_trigger("hello send it", ["send it…"])
        self.assertEqual(cfg_trig, (True, "hello"))

    def test_empty_and_punctuation_only_edit_clauses(self):
        # Nothing usable to replace -> text returned unchanged, no crash.
        for s in ["Hello edit", "Hello edit .", "Hello edit to", "Hello edit , to ?",
                  "Hello edit Hello to", "Hello edit to world"]:
            self.assertEqual(edit(s), s.strip(), s)

    def test_overlapping_edit_triggers(self):
        # Longer trigger wins regardless of list order.
        out = quiet(process_edit_commands, "I like Paris, please edit Paris to Rome",
                    ["edit", "please edit"])
        self.assertEqual(out, "I like Rome")

    def test_edit_symbol_and_apostrophe_targets(self):
        self.assertEqual(edit("I use C# daily, edit C# to Python"), "I use Python daily")
        self.assertEqual(edit("I don't like it, edit don't to do not"), "I do not like it")

    def test_wake_leading_words_limit(self):
        self.assertTrue(detect_wake_phrase("okay um start mic", WAKE)[0])
        self.assertFalse(detect_wake_phrase("so okay um start mic", WAKE)[0])
        self.assertFalse(detect_wake_phrase("... !!", WAKE)[0])

    def test_pause_rounds_up(self):
        # Round-9 bug 1: a pause is never shorter than configured.
        for secs, chunks in [(0.45, 5), (0.35, 4), (0.55, 6), (0.8, 8), (0.3, 3)]:
            seg = UtteranceSegmenter(16000, 0.1, 0.01, secs, 0.25)
            self.assertEqual(seg.pause_chunks, chunks, secs)

    def test_min_speech_rounds_up(self):
        # Round-8 bug 1: never accept less loud audio than configured.
        for secs, chunks in [(0.25, 3), (0.26, 3), (0.3, 3), (0.35, 4), (0.2, 2), (0.05, 1)]:
            seg = UtteranceSegmenter(16000, 0.1, 0.01, 0.8, secs)
            self.assertEqual(seg.min_speech_chunks, chunks, secs)
            self.assertGreaterEqual(seg.min_speech_chunks * 0.1 + 1e-9, secs)

    def test_segmenter_force_split(self):
        seg = UtteranceSegmenter(16000, 0.1, 0.01, pause_seconds=1.0, min_speech_seconds=0.1, max_seconds=0.5)
        loud = np.full(1600, 0.1, np.float32)
        results = [seg.feed(loud) for _ in range(5)]
        self.assertIsNotNone(results[-1])
        self.assertEqual(results[-1][0].size, 5 * 1600)


if __name__ == "__main__":
    unittest.main()
