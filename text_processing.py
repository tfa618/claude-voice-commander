"""
Pure text logic for Claude Voice Commander: trigger words, inline edit
commands, and the wake phrase. No audio / keyboard / model imports here, so
this module is fast to import and fully unit-testable.
"""
import re
import sys
from typing import Optional

TERMINAL_MARKS = ".!?;:"
_SEP_PATTERN = r"\s+(?:\b(?:equals?\s+to|equals?|to)\b|=)\s+"

# Whisper sometimes writes Unicode punctuation: a single "…" for a trailing
# pause, full-width marks, curly quotes. The regexes below only know ASCII,
# so "hello enter…" used to NOT count as "enter". Normalize first.
_PUNCT_MAP = str.maketrans({
    "。": ".", "！": "!", "？": "?", "，": ",", "、": ",", "；": ";", "：": ":",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‘": "'", "’": "'",
    " ": " ", " ": " ", " ": " ", "　": " ",  # incl. ideographic space
})
# Gap BEFORE a trigger: whitespace and sentence punctuation. Quotes are not
# included, so a quotation that ends the message ('He said "hi" enter') keeps
# its closing quote.
_SEP = r"[\s\.,!\?;:]"
# AFTER a trigger (to end of text): the same plus quotes, so a quote Whisper
# put around the whole utterance ('"Hello? Enter."') doesn't block a match.
_EDGE = r"[\s\.,!\?;:\"']"


def normalize_punctuation(text: str) -> str:
    """Map common Unicode punctuation Whisper emits to ASCII equivalents."""
    return text.replace("…", "...").translate(_PUNCT_MAP)

# Multi-word phrases Whisper hallucinates on low-energy noise. Callers gate
# this by peak RMS so legitimate quiet speech is not filtered.
HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "thank you",
    "subtitles by the amara.org community",
    "you",
}


def is_probable_hallucination(text: str, peak_rms: float, silence_threshold: float) -> bool:
    """True if `text` is a known Whisper noise hallucination on a quiet buffer."""
    norm = re.sub(r"[^\w\s\.]", "", text.lower()).strip(" .")
    return norm in HALLUCINATION_PHRASES and peak_rms < 3.0 * silence_threshold


# --------------------------------------------------------------------------
# Trailing triggers ("enter", "cancel", ...)
# --------------------------------------------------------------------------

def _clean_triggers(trigger_list) -> list[str]:
    """
    Strip surrounding punctuation ("send it!" -> "send it": a trailing "!" can
    never satisfy the \\b after it) and sort longest-first, so overlapping
    triggers like ["it", "send it"] prefer the longer match no matter what
    order the caller passed them in.
    """
    cleaned = []
    for t in trigger_list or []:
        # Same normalization as config_loader (strip punctuation, collapse spaces).
        t = " ".join(normalize_punctuation(str(t)).strip(".,!?;:\"' ").split())
        if t and t not in cleaned:
            cleaned.append(t)
    return sorted(cleaned, key=len, reverse=True)


def _trigger_body(trigger: str) -> str:
    """Regex for a trigger; words may be separated by any whitespace."""
    return r"\s+".join(re.escape(w) for w in trigger.split())


def check_trailing_trigger(text: str, trigger_list: list[str]) -> tuple[bool, str]:
    """
    True iff `text` ends with a trigger (case-insensitive). Returns
    (matched, cleaned_body). Sentence-terminal punctuation from the separator
    is preserved; whitespace/commas dropped; the trigger is removed.

    A separator before the trigger is mandatory (except at start-of-string) so
    hyphenated words like "re-enter" do not false-positive.
    """
    stripped = normalize_punctuation(text).strip()
    if not stripped:
        return False, ""
    for trigger in _clean_triggers(trigger_list):
        pattern = re.compile(
            r"(?:^|(?P<sep>" + _SEP + r"+))" + _trigger_body(trigger) + r"\b" + _EDGE + r"*$",
            re.IGNORECASE,
        )
        match = pattern.search(stripped)
        if match:
            cleaned = stripped[: match.start()].rstrip()
            sep = match.group("sep") or ""
            kept = "".join(ch for ch in sep if ch in TERMINAL_MARKS)
            if kept and (not cleaned or cleaned[-1] not in kept):
                cleaned = (cleaned + kept).rstrip()
            # Whisper sometimes quotes the whole utterance ("Hello?" enter);
            # the closing quote went with the trigger, so drop the now
            # unmatched opening one. Double quotes only: a leading single
            # quote is usually an apostrophe ("'Tis", "'90s").
            if cleaned.startswith('"') and cleaned.count('"') % 2 == 1:
                cleaned = cleaned[1:].lstrip()
            return True, cleaned
    return False, stripped


def check_clear_trigger(text, clear_list, submit_list=None):
    """
    True if text ends with a clear/cancel trigger, OR (if submit_list given)
    ends with `<content> <clear> <submit>` — e.g. "…cancel enter".

    Known limitation: any "<content> cancel enter" aborts, even when
    "<content>" is a sentence that legitimately ends in "cancel". The caller
    prints a warning so the loss is never silent.
    """
    is_clear, cleaned = check_trailing_trigger(text, clear_list)
    if is_clear:
        return True, cleaned
    if submit_list:
        is_submit, before_submit = check_trailing_trigger(text, submit_list)
        if is_submit and before_submit:
            is_clear_before, cleaned_before = check_trailing_trigger(before_submit, clear_list)
            if is_clear_before:
                return True, cleaned_before
    return False, text.strip()


# --------------------------------------------------------------------------
# Wake phrase ("start mic")
# --------------------------------------------------------------------------

def detect_wake_phrase(text: str, wake_phrases: list[str], max_leading_words: int = 2) -> tuple[bool, str]:
    """
    Find a wake phrase near the START of an utterance. Returns
    (found, remainder) where remainder is whatever was said after the phrase
    in the same breath ("start mic what time is it" -> "what time is it").

    Why start-only: background speech that merely contains "start mic"
    mid-sentence should not wake the app. Up to `max_leading_words` filler
    words ("okay", "um") are tolerated because Whisper often emits them.
    Words inside the phrase may be joined by space, punctuation or nothing
    ("start mic", "Start, mic.", "startmic").
    """
    stripped = normalize_punctuation(text).strip()
    if not stripped:
        return False, ""
    lead = r"^[\W_]*(?:\w+[\W_]+){0,%d}?" % max_leading_words
    for phrase in sorted((p for p in wake_phrases if p and p.strip()), key=len, reverse=True):
        words = phrase.strip().split()
        body = r"[\s\.,!\?;:\-]*".join(re.escape(w) for w in words)
        m = re.search(lead + r"\b" + body + r"\b[\s\.,!\?;:\-\"']*", stripped, re.IGNORECASE)
        if m:
            remainder = stripped[m.end():].strip()
            # Re-capitalize the first letter of the remainder; Whisper writes
            # "Start mic. what..." -> we want "What...".
            if remainder:
                remainder = remainder[0].upper() + remainder[1:]
            return True, remainder
    return False, stripped


# --------------------------------------------------------------------------
# Inline edit commands ("edit X to Y")
# --------------------------------------------------------------------------

def _strip_clause_trailing_punct(text: str) -> tuple[str, str]:
    """
    Split trailing punctuation off an edit-clause fragment. Returns
    (body, trailing). Multi-dot ellipses are fully stripped, but the final
    dot of a dotted acronym (U.S.A., U.K.) is kept.
    """
    s = text.strip()
    if not s:
        return "", ""
    trailing = ""
    while s and s[-1] in ",;:!?.":
        if s[-1] == "." and re.search(r"\b(?:[A-Za-z]\.){2,}$", s):
            break
        trailing = s[-1] + trailing
        s = s[:-1].strip()
    return s, trailing


def _build_target_regex(target: str) -> re.Pattern:
    """
    Case-insensitive target match with adaptive boundaries: \\b if the edge is
    alnum/underscore; lookaround otherwise, so C++ / C# / U.S.A. match cleanly.
    Empty target -> never-matching pattern (avoids IndexError on target[0]).
    """
    if not target:
        return re.compile(r"(?!)")
    escaped = re.escape(target)
    left = r"\b" if (target[0].isalnum() or target[0] == "_") else r"(?<!\w)"
    right = r"\b" if (target[-1].isalnum() or target[-1] == "_") else r"(?!\w)"
    return re.compile(left + escaped + right, re.IGNORECASE)


def _smart_title(s: str) -> str:
    """Title-case that does not capitalize after apostrophes ("Don't", not "Don'T")."""
    out: list[str] = []
    prev = " "
    for ch in s:
        if ch.isalpha():
            out.append(ch.upper() if (prev.isspace() or prev in "-_/") else ch)
        else:
            out.append(ch)
        prev = ch
    return "".join(out)


def apply_case_pattern(matched_str: str, replacement: str, is_sentence_start: bool) -> str:
    """
    Transfer capitalization from matched_str onto replacement (Whisper often
    emits lowercase). Respects explicit uppercase already in replacement.
    Priority: ALL-CAPS > multi-word Title > capitalized word > as-is.
    """
    if not replacement or not matched_str:
        return replacement
    if any(ch.isupper() for ch in replacement):
        return replacement
    alpha = [ch for ch in matched_str if ch.isalpha()]
    if not alpha:
        return replacement
    if len(alpha) > 1 and all(ch.isupper() for ch in alpha):
        return replacement.upper()
    words = matched_str.split()
    if len(words) > 1 and all(w[0].isupper() for w in words if w and w[0].isalpha()):
        return _smart_title(replacement)
    if not is_sentence_start and matched_str[0].isupper():
        # Mid-sentence capital = proper noun; title-case the replacement.
        return _smart_title(replacement)
    if is_sentence_start and matched_str[0].isupper():
        # Capital only because it starts the sentence; capitalize first letter only.
        return replacement[0].upper() + replacement[1:]
    return replacement


def _split_edit_clause(clause_body: str, current_text: str) -> Optional[tuple[str, str, str]]:
    """
    Split 'X <sep> Y' into (X, Y, trailing_punct). With several separators
    ('go to store to come home'), prefer the split whose target appears in
    current_text (longest wins); else fall back to the last separator.
    """
    matches = list(re.finditer(_SEP_PATTERN, clause_body, re.IGNORECASE))
    if not matches:
        return None
    candidates = []
    for m in matches:
        clean_target, _ = _strip_clause_trailing_punct(clause_body[: m.start()])
        clean_repl, trailing = _strip_clause_trailing_punct(clause_body[m.end():])
        if clean_target and clean_repl:
            candidates.append((clean_target, clean_repl, trailing))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    matching = [c for c in candidates if _build_target_regex(c[0]).search(current_text)]
    if matching:
        matching.sort(key=lambda c: len(c[0]), reverse=True)
        return matching[0]
    return candidates[-1]


def _is_sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or before[-1] in ".!?"


def process_edit_commands(text: str, edit_triggers: list[str]) -> str:
    """
    Apply inline edit clauses: "Capital of South Korea, edit South Korea to
    Russia?" -> "Capital of Russia?".

    - A clause only counts if its target appears in the text BEFORE it;
      otherwise "edit" is ordinary prose ("Please edit this file to be
      shorter") and the text is returned untouched.
    - Substitutions run in order on an evolving string, first occurrence
      only. If an earlier edit removed a later clause's target, that clause
      is skipped with a warning.
    """
    text = normalize_punctuation(text)  # "Russia…" handled like "Russia..."
    trig_union = "|".join(_trigger_body(t) for t in _clean_triggers(edit_triggers))
    if not trig_union or not text.strip():
        return text.strip()

    clause_start_re = re.compile(
        r"([\.,!\?;:]*)\s*\b(?:" + trig_union + r")\b[\s,:]*", re.IGNORECASE,
    )
    all_starts = list(clause_start_re.finditer(text))
    if not all_starts:
        return text.strip()

    # Each "edit" occurrence owns the span up to the next "edit". A span is
    # either a real edit clause (consumed) or prose (kept verbatim).
    spans = []  # (match, span_end, parsed-or-None)
    for idx, m in enumerate(all_starts):
        next_pos = all_starts[idx + 1].start() if idx + 1 < len(all_starts) else len(text)
        body = text[m.end(): next_pos]
        parsed = None
        if re.search(_SEP_PATTERN, body, re.IGNORECASE):
            preceding = text[: m.start()].strip()
            cand = _split_edit_clause(body, preceding)
            if cand and _build_target_regex(cand[0]).search(preceding):
                parsed = cand
            elif cand:
                print(f"Warning: Edit target '{cand[0]}' not found in preceding text; treating as prose.",
                      file=sys.stderr)
        spans.append((m, next_pos, parsed))

    valid = [(m, parsed) for m, _, parsed in spans if parsed]
    if not valid:
        return text.strip()

    # Rebuild the text without the consumed clauses. Prose spans (an "edit"
    # that wasn't a real command, even one AFTER a real command) are kept, so
    # "...edit Paris to Rome, please edit this file to be shorter" doesn't
    # lose its second half.
    pieces = [text[: all_starts[0].start()].strip()]
    for m, end, parsed in spans:
        if parsed is None:
            pieces.append(text[m.start(): end].strip())
    cleaned = ""
    for p in pieces:
        if not p:
            continue
        if cleaned and p[0] not in ".,!?;:":
            cleaned += " "
        cleaned += p

    # Punctuation priority:
    #   leading terminal mark (".!?;:" right before "edit")  beats
    #   trailing terminal mark (after the replacement)        beats
    #   commas / whitespace                                   (never preserved)
    # The leading mark is the one that ended the user's sentence; we cut it
    # off along with the clause, so it must be re-attached at the end.
    preserved = ""
    lead_marks = [ch for ch in valid[0][0].group(1) if ch in TERMINAL_MARKS]
    if lead_marks:
        preserved = lead_marks[0]
    else:
        for _, (_, _, trailing) in valid:
            marks = [ch for ch in trailing if ch in TERMINAL_MARKS]
            if marks:
                preserved = marks[0]  # "..." collapses to a single "."
                break

    for _, (target, repl, _) in valid:
        rx = _build_target_regex(target)
        hit = rx.search(cleaned)
        if not hit:
            print(f"Warning: Edit target '{target}' vanished after an earlier edit; skipping.",
                  file=sys.stderr)
            continue
        new = apply_case_pattern(hit.group(0), repl, _is_sentence_start(cleaned, hit.start()))
        cleaned = cleaned[: hit.start()] + new + cleaned[hit.end():]

    cleaned = cleaned.rstrip(" ,")
    if preserved and (not cleaned or cleaned[-1] not in ".!?"):
        cleaned += preserved
    return cleaned
