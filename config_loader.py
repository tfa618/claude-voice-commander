"""Load and validate config.json. Never raises on bad input; falls back to defaults."""
import json
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    "hotkey": "<f4>",
    "triggers": {
        "submit": ["enter"],
        "edit": ["edit"],
        "clear": ["clear", "cancel"],
    },
    "wake_word_enabled": True,
    "wake_phrases": ["start mic", "start mike", "start mick", "start mic's"],
    "only_when_claude_focused": True,
    "target_app_names": ["claude"],
    "dictation_timeout_seconds": 60,
    "beep": True,
    "restore_clipboard": True,
    "whisper_model": "base.en",
    "language": "en",
    "input_device": None,
    "silence_threshold": 0.012,
    "pause_seconds": 0.8,
    "min_speech_seconds": 0.25,
}


def _normalize_str_list(val, fallback: list[str]) -> list[str]:
    """Normalize a config string / list into a clean lowercase list; fallback if empty."""
    if isinstance(val, str):
        items = [val.strip().lower()] if val.strip() else []
    elif isinstance(val, list):
        items = [str(x).strip().lower() for x in val
                 if x is not None and not isinstance(x, (dict, list)) and str(x).strip()]
    else:
        items = []
    return items if items else list(fallback)


def normalize_hotkey(hk) -> str:
    """Wrap special keys in <> for pynput; map aliases (win->cmd, control->ctrl)."""
    if not isinstance(hk, str) or not hk.strip():
        return "<f4>"
    alias_map = {
        "win": "cmd", "super": "cmd", "control": "ctrl",
        "return": "enter", "escape": "esc",
        "command": "cmd", "option": "alt",
    }
    parts = [p.strip().lower() for p in hk.split("+") if p.strip()]
    if not parts:
        return "<f4>"
    out = []
    for p in parts:
        inner = p[1:-1].strip() if (p.startswith("<") and p.endswith(">")) else p
        inner = alias_map.get(inner, inner)
        out.append(f"<{inner}>" if len(inner) > 1 else inner)
    return "+".join(out)


def _validate_hotkey(hotkey_str: str) -> str:
    try:
        from pynput import keyboard  # optional at config time (tests run without it)
    except Exception:
        return hotkey_str
    try:
        keyboard.HotKey.parse(hotkey_str)
        return hotkey_str
    except Exception as e:
        print(f"Warning: Invalid hotkey '{hotkey_str}' ({e}). Falling back to '<f4>'.", file=sys.stderr)
        return "<f4>"


def load_and_validate_config(config_path: str = CONFIG_PATH) -> dict:
    raw: dict = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                raw = loaded
            else:
                print(f"Warning: {config_path} root must be a JSON object; using defaults.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to parse {config_path} ({e}). Using defaults.", file=sys.stderr)
    else:
        print(f"Note: {config_path} not found. Using default configuration.", file=sys.stderr)

    d = DEFAULT_CONFIG
    trig_raw = raw.get("triggers") if isinstance(raw.get("triggers"), dict) else {}
    triggers = {}
    for key in ("submit", "edit", "clear"):
        lst = _normalize_str_list(trig_raw.get(key), d["triggers"][key])
        # Store triggers without surrounding punctuation ("send it!" -> "send it")
        # and without duplicates, so anything reading the config sees the form
        # that actually matches. text_processing also does this defensively.
        cleaned = []
        for t in lst:
            t = " ".join(t.strip(".,!?;:\"'…。！？，、；：“”‘’ ").split())
            if t and t not in cleaned:
                cleaned.append(t)
        if not cleaned:
            # e.g. "submit": ["!"] — punctuation alone can't be spoken as a word.
            print(f"Warning: triggers.{key} has no usable words {lst}; "
                  f"using default {d['triggers'][key]}.", file=sys.stderr)
        lst = cleaned or list(d["triggers"][key])
        lst.sort(key=len, reverse=True)  # multi-word triggers match before shorter prefixes
        triggers[key] = lst

    # Don't str() the raw value: JSON null would become "None" -> "<none>".
    hk_raw = raw.get("hotkey", d["hotkey"])
    hotkey = _validate_hotkey(normalize_hotkey(hk_raw if isinstance(hk_raw, str) else d["hotkey"]))

    def _float(key, lo, hi):
        val = raw.get(key, d[key])
        if isinstance(val, bool):
            return d[key]
        try:
            val = float(val)
        except (TypeError, ValueError):
            return d[key]
        if not math.isfinite(val):
            return d[key]
        return max(lo, min(hi, val))

    def _bool(key):
        val = raw.get(key, d[key])
        return val if isinstance(val, bool) else d[key]

    wm = raw.get("whisper_model", d["whisper_model"])
    whisper_model = wm.strip() if isinstance(wm, str) and wm.strip() else d["whisper_model"]
    lang = raw.get("language", d["language"])
    language = lang.strip().lower() if isinstance(lang, str) and lang.strip() else None

    device = raw.get("input_device", None)
    if not (device is None or isinstance(device, (int, str))) or isinstance(device, bool):
        print("Warning: input_device must be null, a number or a name; using default device.", file=sys.stderr)
        device = None

    return {
        "hotkey": hotkey,
        "triggers": triggers,
        "wake_word_enabled": _bool("wake_word_enabled"),
        "wake_phrases": _normalize_str_list(raw.get("wake_phrases"), d["wake_phrases"]),
        "only_when_claude_focused": _bool("only_when_claude_focused"),
        "target_app_names": _normalize_str_list(raw.get("target_app_names"), d["target_app_names"]),
        "dictation_timeout_seconds": _float("dictation_timeout_seconds", 0.0, 3600.0),
        "beep": _bool("beep"),
        "restore_clipboard": _bool("restore_clipboard"),
        "whisper_model": whisper_model,
        "language": language,
        "input_device": device,
        "silence_threshold": _float("silence_threshold", 0.001, 0.5),
        "pause_seconds": _float("pause_seconds", 0.3, 5.0),
        "min_speech_seconds": _float("min_speech_seconds", 0.05, 2.0),
    }
