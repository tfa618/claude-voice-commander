# Claude Voice Commander

Hands-free voice input for the **Claude Desktop app**. Say **"start mic"**, speak your message, say **"enter"**, and it's sent. No clicking, no typing.

```
You:    "start mic"                                  *beep*
You:    "What's the capital of South Korea,"
You:    "edit South Korea to Russia?"
You:    "enter"                                      *beep*
Claude: receives  "What's the capital of Russia?"
        ...answers...
You:    "start mic"  → next message
```

- **Wake phrase** — say "start mic" to begin; after each sent message it goes back to waiting.
- **Voice send** — say "enter" at the end of your message.
- **Only listens while Claude is the active window.** Switch to another app and the microphone turns off.
- **Fix mistakes by voice** — "edit X to Y", or "cancel" to throw the message away.
- **Private** — speech recognition runs on your computer with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Audio never leaves your machine.

> This is a small companion app that runs next to Claude Desktop. It is not an official Anthropic product, and it doesn't use Claude's built-in mic button.

---

## Requirements

- Windows 10/11 or macOS 12+ (Linux with X11 works too)
- [Python 3.9 – 3.14](https://www.python.org/downloads/) (3.13 recommended). **Python 3.15 doesn't work yet**: the speech engine's `onnxruntime` dependency has no 3.15 build, so `pip install` fails with "conflicting dependencies".
- The Claude Desktop app
- A microphone
- About 500 MB free disk space and any reasonably modern CPU (no GPU needed)

## Install

```bash
git clone https://github.com/tfa618/claude-voice-commander.git
cd claude-voice-commander
python -m venv .venv
```

Activate the environment:

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

Then install:

```bash
pip install -r requirements.txt
```

**Linux only:** also install PortAudio and xdotool, e.g. `sudo apt install libportaudio2 xdotool`.

## Run

```bash
python main.py
```

The first run downloads the speech model (~150 MB). Leave the window open; it works in the background. Press **Ctrl+C** in that window to quit.

### macOS permissions

The first time, macOS will ask for permissions for the app you run it from (Terminal, iTerm, VS Code…). In **System Settings → Privacy & Security**, allow it under:

| Permission | Why |
|---|---|
| **Microphone** | to hear you |
| **Accessibility** | to paste and press Enter in Claude |
| **Input Monitoring** | for the F4 hotkey |

Restart the terminal after granting them.

## Voice commands

| Say | What happens |
|---|---|
| **"start mic"** | Starts listening for a message (you'll hear a rising beep). Works only while Claude is the active window. |
| *your message* | Collected across pauses — take your time. |
| **"enter"** (at the end) | Sends the message to Claude, then goes back to waiting for "start mic". |
| **"cancel"** or **"clear"** (at the end) | Throws the current message away. |
| **"edit X to Y"** | Replaces X with Y before sending. Also understands "edit X equals Y". |

You can also say it all in one breath: *"start mic, what time is it in Tokyo, enter."*

**"enter" only counts as the last word before a pause.** "Press enter to continue" won't send anything; "Press enter to continue, enter" sends "Press enter to continue".

**"edit" only counts if the word you're replacing was already said.** "Please edit this file to be shorter" is sent as normal text.

**F4** starts or stops dictation without the wake phrase — handy in a noisy room.

If you say "start mic" and then nothing for 60 seconds, it goes back to waiting.

If a message can't be sent (for example you switched away from Claude just as you said "enter"), your text is kept for as long as you need (the 60-second timeout doesn't apply): switch back to Claude and say **"enter"** to try again, or **"cancel"** (or F4) to drop it. You can also add more words before saying "enter".

In the rare case the text was pasted but the Enter key couldn't be pressed, it's already in Claude's message box. The console tells you, and you just press Enter yourself (no voice retry, so it never gets pasted twice). Your own clipboard is still put back.

## Settings (`config.json`)

| Setting | Default | Meaning |
|---|---|---|
| `wake_phrases` | `["start mic", …]` | Phrases that start dictation. Includes common mishearings ("start mike"). Use something like `["hey claude"]` to change it. |
| `wake_word_enabled` | `true` | `false` = always dictating; just say "enter" after each message. |
| `only_when_claude_focused` | `true` | Listen only while Claude is the active window. |
| `target_app_names` | `["claude"]` | App name(s) that count as "Claude". |
| `triggers.submit` | `["enter"]` | Words that send. Try `["send it"]` if "enter" fires by accident. |
| `triggers.clear` | `["clear", "cancel"]` | Words that discard the message. |
| `triggers.edit` | `["edit"]` | Word that starts a correction. |
| `hotkey` | `"<f4>"` | Start/stop key. Examples: `"<ctrl>+<shift>+m"`, `"<f9>"`. |
| `dictation_timeout_seconds` | `60` | Back to waiting after this much silence. `0` = never. |
| `beep` | `true` | Play sounds on start / send / cancel. |
| `restore_clipboard` | `true` | Put your previous clipboard back after pasting. |
| `whisper_model` | `"base.en"` | Speech model. `"tiny.en"` = faster, `"small.en"` = more accurate but slower. |
| `language` | `"en"` | Language code. For other languages use a non-`.en` model, e.g. `"small"`. |
| `input_device` | `null` | Microphone name or number; `null` = system default. |
| `silence_threshold` | `0.012` | Raise it if background noise keeps triggering; lower it if you're not heard. |
| `pause_seconds` | `0.8` | Silence that ends a phrase. |
| `min_speech_seconds` | `0.25` | Shorter sounds are ignored (clicks, coughs). |

Invalid values are reported and replaced with defaults; the app won't crash on a bad config.

## Troubleshooting

**`pip install` fails with "conflicting dependencies" / "no matching distribution … onnxruntime".**
Your Python is too new (usually 3.15). Install Python 3.13 alongside it, delete the `.venv` folder, and create it again with that version:
```
py -3.13 -m venv .venv          (Windows)
python3.13 -m venv .venv        (macOS / Linux)
```

**Speech model download fails** (`WinError 10053/10054`, "connection was aborted", "snapshot is incomplete").
Something on your network or PC (antivirus, firewall, VPN, or your internet provider) is blocking Python's download. Download the model in your web browser instead:

1. Inside the `claude-voice-commander` folder, create a folder `models`, and inside it a folder `base.en`.
2. Download these 4 files in your browser (right-click → Save link as) into `models\base.en`:
   - https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/config.json
   - https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/model.bin (~145 MB)
   - https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/tokenizer.json
   - https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/vocabulary.txt

   If huggingface.co doesn't open, use the same links with `hf-mirror.com` instead of `huggingface.co`.
3. In `config.json` (the app's settings file), set `"whisper_model": "models/base.en"`.
4. Run `python main.py` again.

Alternatively, before `python main.py`, try `set HF_HUB_DISABLE_XET=1` (Windows) / `export HF_HUB_DISABLE_XET=1` (macOS/Linux), or pause your VPN/antivirus web protection for the one-time download.

**It never reacts to "start mic".**
Make sure Claude is the active (front) window — the console shows `● Claude is active`. The console also prints `(heard: "...")` for anything it didn't recognize as the wake phrase; add that spelling to `wake_phrases`. For better recognition overall, try `"whisper_model": "small.en"`.

**Background noise or other people trigger it.**
Raise `silence_threshold` (e.g. `0.02`), or choose a less common wake phrase.

**The message is typed but not sent / nothing is pasted.**
macOS: grant **Accessibility** to your terminal (see above). Make sure Claude's message box has the cursor.

**"Can't detect the active window" warning.**
This means the app **listens in every app**, not just Claude, until you fix it (it prefers to keep working over silently not working). Linux: install `xdotool` (Wayland isn't supported — use an X11 session). Or set `only_when_claude_focused` to `false` to make that choice explicit.

**It's slow on my computer.**
Use `"whisper_model": "tiny.en"`.

## Development

```bash
python -m unittest discover -s tests -v
```

The tests need only `numpy` — no microphone or speech model.

| File | What it does |
|---|---|
| `main.py` | App runtime: microphone, wake phrase, sending to Claude |
| `text_processing.py` | Trigger words, wake phrase detection, "edit X to Y" |
| `audio_utils.py` | Resampling, speech segmentation, beeps |
| `focus.py` | Detects the active window on Windows / macOS / Linux |
| `config_loader.py` | Reads and validates `config.json` |

## License

MIT — see [LICENSE](LICENSE).
