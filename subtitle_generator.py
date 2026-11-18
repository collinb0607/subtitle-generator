#!/usr/bin/env python3
"""
Subtitle Generator - a cross-platform (Windows / macOS / Linux) tool that
listens to a video (or audio file) and writes its subtitles, using OpenAI's
Whisper speech recognition running on your own computer. Check and fix every
caption, then save an .srt or .vtt file, a video with a switchable subtitle
track, or a video with the captions burned into the picture.

The speech models are downloaded once, the first time each is used. Nothing
is ever uploaded. On Windows PCs with an NVIDIA graphics card, Whisper can run
on the card, many times faster (it needs a one-time download of NVIDIA's
cuBLAS library).

Requirements:
    Python 3.9+ (with Tkinter, included in the python.org installers)
    faster-whisper  -> Whisper speech recognition (CTranslate2)
    imageio-ffmpeg  -> bundles an FFmpeg binary: reads the sound, burns in and adds subtitles
    Pillow          -> the player's picture
    sounddevice     -> the player's sound (optional: the player is silent without it)
    certifi         -> trusted certificates for downloading the models
    tkinterdnd2     -> drag and drop (optional: the app works without it)
    The app offers to install them on first run. Manual install, using the Python that runs the app:
        python -m pip install faster-whisper imageio-ffmpeg pillow certifi sounddevice tkinterdnd2

Run:
    python subtitle_generator.py
"""

import hashlib
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import colorchooser, filedialog, messagebox, ttk

# Filled in by load_libraries(), so the window can still open (and offer to
# install them) when they're missing.
np = WhisperModel = Image = ImageTk = None

PACKAGES = ["faster-whisper", "imageio-ffmpeg", "pillow", "certifi", "sounddevice", "tkinterdnd2"]
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".wmv", ".flv", ".mts", ".m2ts", ".ts",
              ".3gp", ".mpg", ".mpeg"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
SUB_EXTS = {".srt", ".vtt"}
FILE_TYPES = [("Videos and audio", " ".join("*" + e for e in sorted(VIDEO_EXTS | AUDIO_EXTS))),
              ("All files", "*.*")]
HOME = os.path.join(os.path.expanduser("~"), ".subtitle-generator")
SAMPLE_RATE = 16000  # what Whisper listens at

# The Whisper models, converted for CTranslate2 (MIT licence), pinned to an exact
# version. Every file is checked against its SHA-256 after downloading.
HF = "https://huggingface.co/{repo}/resolve/{rev}/{file}"
MODELS = {
    "Accurate — Whisper large-v3 turbo": dict(
        name="whisper-large-v3-turbo", repo="dropbox-dash/faster-whisper-large-v3-turbo",
        rev="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf", files=[
            ("config.json", 2263, "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e"),
            ("model.bin", 1617884929, "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"),
            ("preprocessor_config.json", 340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
            ("tokenizer.json", 2710337, "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"),
            ("vocabulary.json", 1068114, "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1")],
        about="The most accurate, in 99 languages. 1.6 GB download. On a recent PC's processor it works "
              "about twice as fast as the video plays; on an NVIDIA graphics card about 20 times as fast."),
    "Fast — Whisper small": dict(
        name="whisper-small", repo="Systran/faster-whisper-small",
        rev="536b0662742c02347bc0e980a01041f333bce120", files=[
            ("config.json", 2370, "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828"),
            ("model.bin", 483546902, "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"),
            ("tokenizer.json", 2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
            ("vocabulary.txt", 459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913")],
        about="Good for clear speech, and quicker on a computer without an NVIDIA graphics card: about "
              "five times as fast as the video plays. 485 MB download. Makes more mistakes with accents, "
              "noise and rare words."),
}
# NVIDIA's cuBLAS library (from its official PyPI package), which CTranslate2
# needs to run Whisper on an NVIDIA graphics card. Windows only: macOS has no
# CUDA, and on Linux it comes from the system or pip.
CUDA_WHEEL = dict(
    url="https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
        "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
    size=553162896, sha256="623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661",
    dlls=["cublas64_12.dll", "cublasLt64_12.dll"])
CPU_DEVICE, GPU_DEVICE = "This computer's processor", "NVIDIA graphics card"

LANGUAGES = {"Detect automatically": None, "English": "en", "Spanish": "es", "French": "fr", "German": "de",
             "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Polish": "pl", "Russian": "ru",
             "Ukrainian": "uk", "Turkish": "tr", "Arabic": "ar", "Hindi": "hi", "Japanese": "ja",
             "Korean": "ko", "Chinese": "zh", "Vietnamese": "vi", "Indonesian": "id", "Thai": "th",
             "Swedish": "sv", "Norwegian": "no", "Danish": "da", "Finnish": "fi", "Greek": "el",
             "Czech": "cs", "Romanian": "ro", "Hungarian": "hu", "Hebrew": "he", "Tagalog": "tl",
             "Malay": "ms", "Catalan": "ca", "Croatian": "hr", "Serbian": "sr", "Bulgarian": "bg",
             "Slovak": "sk", "Persian": "fa", "Urdu": "ur", "Bengali": "bn", "Tamil": "ta"}
LANG_NAMES = {v: k for k, v in LANGUAGES.items() if v}
# ISO 639-2 codes, for labelling the subtitle track inside a video.
ISO3 = {"en": "eng", "es": "spa", "fr": "fre", "de": "ger", "it": "ita", "pt": "por", "nl": "dut", "pl": "pol",
        "ru": "rus", "uk": "ukr", "tr": "tur", "ar": "ara", "hi": "hin", "ja": "jpn", "ko": "kor", "zh": "chi",
        "vi": "vie", "id": "ind", "th": "tha", "sv": "swe", "no": "nor", "da": "dan", "fi": "fin", "el": "gre",
        "cs": "cze", "ro": "rum", "hu": "hun", "he": "heb", "tl": "tgl", "ms": "may", "ca": "cat", "hr": "hrv",
        "sr": "srp", "bg": "bul", "sk": "slo", "fa": "per", "ur": "urd", "bn": "ben", "ta": "tam"}
CAPTION_MODES = {
    "Sentences, up to 2 lines (YouTube, films)": dict(line=42, lines=2, secs=6.0),
    "Short phrases, 1 line (Shorts, Reels, TikTok)": dict(line=22, lines=1, secs=2.6),
}
# Fonts for burned-in captions: (regular, bold) file names on Windows and macOS.
FONTS = {"Arial": (("arial.ttf", "arialbd.ttf"), ("Arial.ttf", "Arial Bold.ttf")),
         "Verdana": (("verdana.ttf", "verdanab.ttf"), ("Verdana.ttf", "Verdana Bold.ttf")),
         "Trebuchet MS": (("trebuc.ttf", "trebucbd.ttf"), ("Trebuchet MS.ttf", "Trebuchet MS Bold.ttf")),
         "Georgia": (("georgia.ttf", "georgiab.ttf"), ("Georgia.ttf", "Georgia Bold.ttf")),
         "Impact": (("impact.ttf", "impact.ttf"), ("Impact.ttf", "Impact.ttf"))}
SIZES = {"Small": 0.045, "Medium": 0.06, "Large": 0.075, "Extra large": 0.095}  # of the picture's short side
LOOKS = ["Outline", "Box behind the text", "Shadow only"]
POSITIONS = {"Bottom": 2, "Middle": 5, "Top": 8}  # ASS alignment
ANIMATIONS = ["None", "Highlight each word", "Words pop in", "One word at a time"]
# One-click looks: every burned-in setting at once. Changing one afterwards shows "Custom".
PRESETS = {
    "Clean": dict(font="Arial", size="Medium", bold=True, colour="#ffffff", look="Outline",
                  position="Bottom", animation="None", highlight="#4da3ff"),
    "Karaoke": dict(font="Arial", size="Large", bold=True, colour="#ffffff", look="Outline",
                    position="Bottom", animation="Highlight each word", highlight="#4da3ff"),
    "Bold pop": dict(font="Impact", size="Extra large", bold=False, colour="#ffffff", look="Outline",
                     position="Middle", animation="Words pop in", highlight="#ffd400"),
    "One word": dict(font="Impact", size="Extra large", bold=False, colour="#ffd400", look="Outline",
                     position="Middle", animation="One word at a time", highlight="#ffd400"),
    "Boxed": dict(font="Arial", size="Medium", bold=False, colour="#ffffff", look="Box behind the text",
                  position="Bottom", animation="None", highlight="#4da3ff"),
}


def load_libraries():
    """Import the speech and image libraries. Returns the pip names of missing
    packages (tkinterdnd2 is optional and checked separately)."""
    global np, WhisperModel, Image, ImageTk
    import importlib
    importlib.invalidate_caches()  # see packages installed since startup
    missing = []
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError:
        missing.append("faster-whisper")
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing.append("pillow")
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        missing.append("imageio-ffmpeg")
    try:
        import certifi  # noqa: F401
    except ImportError:
        missing.append("certifi")
    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        missing.append("sounddevice")
    try:
        import tkinterdnd2  # noqa: F401
    except ImportError:
        missing.append("tkinterdnd2")
    return missing


# Hide the console window that would otherwise flash up on Windows.
_POPEN_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def install_requirements(packages):
    """pip-install the packages into *this* Python interpreter (see the other
    Offline Toolkit apps: several Pythons are the usual cause of 'installed but not found')."""
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):  # pythonw has no console; use python.exe for pip
        exe = exe[:-len("pythonw.exe")] + "python.exe"
    r = subprocess.run([exe, "-m", "pip", "install", "--upgrade"] + packages,
                       capture_output=True, text=True, errors="replace", **_POPEN_FLAGS)
    add_user_site()
    return r.returncode == 0, (r.stderr or r.stdout).strip().splitlines()[-3:]


def add_user_site():
    """Make packages pip just put in the per-user folder importable right away."""
    import site
    path = site.getusersitepackages() if site.ENABLE_USER_SITE else None  # off in venvs
    if path and os.path.isdir(path) and path not in sys.path:
        site.addsitedir(path)


# --------------------------------------------------------------------------- #
#  FFmpeg: reading the video, playing sound, burning in, adding a track
# --------------------------------------------------------------------------- #
def ffmpeg_ok(path):
    """True if this FFmpeg can make H.264 MP4s and draw subtitles (libass)."""
    try:
        enc = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True, text=True,
                             errors="replace", timeout=15, **_POPEN_FLAGS).stdout
        flt = subprocess.run([path, "-hide_banner", "-filters"], capture_output=True, text=True,
                             errors="replace", timeout=15, **_POPEN_FLAGS).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "libx264" in enc and " subtitles " in flt and "mov_text" in enc


def find_ffmpeg():
    """The FFmpeg bundled by imageio-ffmpeg, or one on the PATH that can do the job; else None."""
    try:
        import importlib
        importlib.invalidate_caches()
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_ok(path):
            return path
    except Exception:
        pass
    path = shutil.which("ffmpeg")
    return path if path and ffmpeg_ok(path) else None


def probe_media(ffmpeg, path):
    """dict(duration, width, height, has_audio, audio_codec, error) from `ffmpeg -i` (header only).
    width/height are as displayed (phone videos are often stored sideways with a rotate flag)."""
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True,
                         errors="replace", **_POPEN_FLAGS).stderr
    info = dict(duration=None, width=None, height=None, has_audio=False, audio_codec=None, error=None)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m:
        info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    for line in out.splitlines():
        if "Video:" in line and "attached pic" not in line and info["width"] is None:
            m = re.search(r"\b(\d{2,5})x(\d{2,5})\b", line)
            if m:
                info["width"], info["height"] = int(m[1]), int(m[2])
        if "Audio:" in line and not info["has_audio"]:
            info["has_audio"] = True
            m = re.search(r"Audio:\s*(\w+)", line)
            info["audio_codec"] = m[1] if m else None
    m = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", out) or re.search(r"rotate\s*:\s*(-?\d+)", out)
    if m and info["width"] and round(abs(float(m[1]))) % 180 == 90:
        info["width"], info["height"] = info["height"], info["width"]
    if not info["duration"]:
        text = out.lower()
        if "no such file" in text:
            info["error"] = "File not found."
        elif "permission denied" in text:
            info["error"] = "Permission denied — the file may be open in another program."
        elif "moov atom not found" in text or "invalid data" in text:
            info["error"] = "The file looks incomplete or damaged."
        else:
            info["error"] = "This doesn't look like a video or audio file."
    elif not info["has_audio"]:
        info["error"] = "This video has no sound, so there's nothing to write subtitles from."
    return info


def read_audio(ffmpeg, path, check_cancel):
    """The whole soundtrack as 16 kHz mono float32 samples (what Whisper listens to)."""
    p = subprocess.Popen([ffmpeg, "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar",
                          str(SAMPLE_RATE), "-f", "f32le", "pipe:1"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, **_POPEN_FLAGS)
    chunks = []
    try:
        while True:
            check_cancel()
            c = p.stdout.read(1 << 20)
            if not c:
                break
            chunks.append(c)
    finally:
        if p.poll() is None:
            p.kill()
        err = p.stderr.read().decode("utf-8", "replace")
        p.wait()
    data = b"".join(chunks)
    if not data:
        raise RuntimeError("Couldn't read the sound: " + (err.strip().splitlines() or ["unknown error"])[-1])
    return np.frombuffer(data[:len(data) // 4 * 4], np.float32)


def run_ffmpeg(ffmpeg, args, duration, on_progress, check_cancel, cwd=None):
    """Run FFmpeg, reporting progress 0-1 from its -progress output. Raises on failure."""
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-y", "-progress", "pipe:1", "-nostats"] + args
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, **_POPEN_FLAGS)
    errors = []
    threading.Thread(target=lambda: errors.extend(p.stderr.read().decode("utf-8", "replace").splitlines()),
                     daemon=True).start()
    try:
        for raw in p.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    t = int(line.split("=", 1)[1]) / 1e6
                    on_progress(min(1.0, t / duration) if duration else 0)
                except ValueError:
                    pass
            if check_cancel(raise_it=False):
                p.kill()
                break
        p.wait()
    finally:
        if p.poll() is None:
            p.kill()
    check_cancel()
    if p.returncode != 0:
        tail = [e for e in errors if e.strip()][-2:]
        raise RuntimeError("FFmpeg failed: " + (" ".join(tail) or f"exit code {p.returncode}"))


def font_dirs():
    if sys.platform == "win32":
        return [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")]
    if sys.platform == "darwin":
        return ["/System/Library/Fonts/Supplemental", "/Library/Fonts", "/System/Library/Fonts",
                os.path.expanduser("~/Library/Fonts")]
    return []


def font_files(name):
    """The installed (regular, bold) files of a font, or None if they aren't found."""
    names = FONTS[name][0 if sys.platform == "win32" else 1]
    found = []
    for fn in names:
        hit = next((os.path.join(d, fn) for d in font_dirs() if os.path.isfile(os.path.join(d, fn))), None)
        if hit is None:
            return None
        found.append(hit)
    return found


def available_fonts():
    if sys.platform not in ("win32", "darwin"):
        return list(FONTS)  # Linux: fontconfig finds them by name (or a look-alike)
    return [n for n in FONTS if font_files(n)] or ["Arial"]


# --------------------------------------------------------------------------- #
#  Captions: from words to lines, and subtitle files
# --------------------------------------------------------------------------- #
def break_lines(text, line_len, max_lines):
    """Split a caption into at most two balanced lines (at a space when there is one)."""
    text = " ".join(text.split()) if " " in text else text.strip()
    if max_lines < 2 or len(text) <= line_len:
        return text
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if spaces:
        best = min(spaces, key=lambda i: max(i, len(text) - i - 1))
        return text[:best] + "\n" + text[best + 1:]
    mid = len(text) // 2  # no spaces (Chinese, Japanese, Thai...): split in the middle
    return text[:mid] + "\n" + text[mid:]


def build_captions(words, mode):
    """Group recognised words (start, end, text-with-its-leading-space) into captions:
    a list of dict(start, end, text). Breaks at sentence ends, pauses, the length
    limit and the time limit, so each caption is easy to read in time."""
    m = CAPTION_MODES[mode]
    max_chars = m["line"] * m["lines"]
    caps, cur = [], []

    def joined(ws):
        return "".join(w[2] for w in ws).strip()

    def flush():
        if cur:
            text = joined(cur)
            if text:
                caps.append(dict(start=cur[0][0], end=cur[-1][1], text=break_lines(text, m["line"], m["lines"]),
                                 words=[[st, en, raw.strip()] for st, en, raw in cur]))
            cur.clear()
    for start, end, raw in words:
        if not raw.strip():
            continue
        if cur:
            prev = cur[-1][2].strip()
            size = len(joined(cur + [(start, end, raw)]))
            pause = start - cur[-1][1]
            ends = raw.strip()[-1:] in (".?!。？！,;:，、" if m["lines"] == 1 else ".?!。？！")
            if (pause > 0.8 or (prev[-1:] in ".?!。？！" and len(joined(cur)) >= (1 if m["lines"] == 1 else 15))
                    or (m["lines"] == 1 and prev[-1:] in ",;:，、")):
                flush()
            elif ends and size <= max_chars * 1.3 and end - cur[0][0] <= m["secs"] * 1.3:
                pass  # a sentence's last word stays with it, rather than alone on the next caption
            elif size > max_chars or end - cur[0][0] > m["secs"]:
                # Too long: end at the last comma if that leaves a reasonable caption,
                # so a phrase isn't cut in the middle; else here.
                cut = next((i for i in range(len(cur) - 1, 0, -1) if cur[i - 1][2].strip()[-1:] in ",;:，、"
                            and len(joined(cur[:i])) >= max_chars * 0.4), None)
                if cut is not None:
                    rest = cur[cut:]
                    del cur[cut:]
                    flush()
                    cur.extend(rest)
                    if len(joined(cur + [(start, end, raw)])) > max_chars or end - cur[0][0] > m["secs"]:
                        flush()
                else:
                    flush()
        cur.append((start, end, raw))
    flush()
    # Keep each caption on screen long enough to read, without overlapping the next.
    for i, c in enumerate(caps):
        nxt = caps[i + 1]["start"] if i + 1 < len(caps) else None
        want = max(c["end"], c["start"] + 0.8) + 0.15
        c["end"] = min(want, nxt - 0.04) if nxt is not None else want
        c["end"] = max(c["end"], c["start"] + 0.2)
    return caps


# ---- each caption's words, and their times -------------------------------------------- #
def _tokens(text):
    """A caption's words, as shown (split at spaces and line breaks)."""
    return text.split()


def even_words(tokens, start, end):
    """Times for words that have none: the caption's time shared out by their length."""
    if not tokens:
        return []
    total = sum(len(t) for t in tokens) or 1
    out, t = [], start
    for tok in tokens:
        d = (end - start) * len(tok) / total
        out.append([t, t + d, tok])
        t += d
    return out


def caption_words(c):
    """The caption's words with times: its recognised ones if they still match the text,
    else shared out evenly (an opened .srt, or text typed in)."""
    words, tokens = c.get("words"), _tokens(c["text"])
    if words and [w[2] for w in words] == tokens:
        return [[max(c["start"], w[0]), min(c["end"], max(w[1], w[0])), w[2]] for w in words]
    return even_words(tokens, c["start"], c["end"])


def retime_words(old_words, new_text, start, end):
    """Word times for edited text: words that didn't change keep their times; new ones
    share out the time between their neighbours (by length)."""
    import difflib
    new = _tokens(new_text)
    old = [w[2] for w in old_words]
    out = [None] * len(new)
    for a, b, n in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
        for k in range(n):
            out[b + k] = list(old_words[a + k])
    i = 0
    while i < len(new):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(new) and out[j] is None:
            j += 1
        lo = out[i - 1][1] if i > 0 else start
        hi = out[j][0] if j < len(new) else end
        if hi - lo < 0.05 * (j - i):  # no room between the kept words: share out the whole caption
            return even_words(new, start, end)
        out[i:j] = even_words(new[i:j], lo, hi)
        i = j
    return out


def fit_words(words, old_start, old_end, start, end):
    """Word times for a caption moved or resized from (old_start, old_end) to (start, end):
    moved as they are if it only moved, else stretched or squeezed with it."""
    span = (old_end - old_start) or 1.0
    k = (end - start) / span
    return [[start + (w[0] - old_start) * k, start + (w[1] - old_start) * k, w[2]] for w in words]


def fmt_ts(t, sep=","):
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_short(t):
    """0:05.1 / 1:02:03.4, for the list."""
    t = max(0.0, t)
    h, rest = divmod(t, 3600)
    m, s = divmod(rest, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h else f"{int(m)}:{s:04.1f}"


def parse_ts(text):
    """'1:02:03.5', '2:03,5', '63.5' -> seconds. Raises ValueError."""
    text = text.strip().replace(",", ".")
    if not text:
        raise ValueError
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    if secs < 0:
        raise ValueError
    return secs


def to_srt(caps):
    return "\n".join(f"{i}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n{c['text']}\n"
                     for i, c in enumerate(caps, 1))


def to_vtt(caps):
    return "WEBVTT\n\n" + "\n".join(f"{fmt_ts(c['start'], '.')} --> {fmt_ts(c['end'], '.')}\n{c['text']}\n"
                                    for c in caps)


def parse_subtitles(text):
    """Captions from an .srt or .vtt file's text."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    caps = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.split("\n") if l.strip()]
        for k, line in enumerate(lines):
            m = re.match(r"\s*([\d:.,]+)\s*-->\s*([\d:.,]+)", line)
            if m:
                try:
                    start, end = parse_ts(m[1]), parse_ts(m[2])
                except ValueError:
                    break
                body = "\n".join(re.sub(r"<[^>]+>", "", l) for l in lines[k + 1:]).strip()
                if body:
                    caps.append(dict(start=start, end=max(end, start + 0.1), text=body))
                break
    caps.sort(key=lambda c: c["start"])
    return caps


def _ass_colour(hexcode, alpha=0):
    r, g, b = int(hexcode[1:3], 16), int(hexcode[3:5], 16), int(hexcode[5:7], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _ass_text(text):
    return text.replace("\\", "\u29f5").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def _ass_lines(c):
    """The caption's words, each with whether a line break comes before it."""
    out = []
    for n, line in enumerate(c["text"].split("\n")):
        for k, tok in enumerate(line.split()):
            out.append((tok, n > 0 and k == 0))
    return out


def _animated_events(c, st):
    """ASS events that animate a caption word by word, as libass draws them:
    (start, end, text) for each moment between two words."""
    words = caption_words(c)
    toks = _ass_lines(c)
    if not words or len(words) != len(toks):
        return [(c["start"], c["end"], _ass_text(c["text"]))]
    hi = "&H" + _ass_colour(st.get("highlight") or "#4da3ff")[4:]  # an override's colour: &HBBGGRR&
    anim = st["animation"]
    events = []
    for i, w in enumerate(words):
        start = c["start"] if i == 0 else w[0]
        end = c["end"] if i == len(words) - 1 else max(words[i + 1][0], start + 0.01)
        if anim == "One word at a time":
            # The word pops: a little bigger, settling to its size.
            text = "{\\fscx118\\fscy118\\t(0,90,\\fscx100\\fscy100)}" + _ass_text(w[2])
        else:
            parts = []
            for k, (tok, newline) in enumerate(toks):
                lead = "\\N" if newline else (" " if k else "")
                word = _ass_text(tok)
                if anim == "Highlight each word" and k == i:
                    word = "{\\c" + hi + "&}" + word + "{\\r}"
                elif anim == "Words pop in":
                    if k == i:  # the newest word fades in, in the highlight colour
                        word = "{\\alpha&HFF&\\c" + hi + "&\\t(0,80,\\alpha&H00&)}" + word + "{\\r}"
                    elif k > i:  # words still to come keep their place but don't show, so nothing jumps
                        word = "{\\alpha&HFF&}" + word + "{\\r}"
                parts.append(lead + word)
            text = "".join(parts)
        events.append((start, end, text))
    return events


def to_ass(caps, width, height, st):
    """An Advanced SubStation file: the captions in the chosen look, for burning in."""
    size = max(8, round(min(width, height) * SIZES[st["size"]]))
    if st.get("animation") == "One word at a time":
        size = round(size * 1.35)  # a single word can be big
    look = st["look"]
    if look == "Box behind the text":
        border, outline, shadow, back = 3, max(2, round(size * 0.22)), 0, _ass_colour("#000000", 0x60)
        outline_col = _ass_colour("#000000", 0x50)
    elif look == "Shadow only":
        border, outline, shadow, back = 1, 0, max(1, round(size * 0.07)), _ass_colour("#000000", 0x40)
        outline_col = _ass_colour("#000000")
    else:
        border, outline, shadow, back = 1, max(1, round(size * 0.07)), max(1, round(size * 0.03)), \
            _ass_colour("#000000", 0x80)
        outline_col = _ass_colour("#000000")
    align = POSITIONS[st["position"]]
    margin_v = round(height * 0.06)
    margin_h = round(width * 0.05)
    style = (f"Style: Default,{st['font']},{size},{_ass_colour(st['colour'])},&H000000FF,{outline_col},{back},"
             f"{-1 if st['bold'] else 0},0,0,0,100,100,0,0,{border},{outline},{shadow},{align},"
             f"{margin_h},{margin_h},{margin_v},1")
    lines = []
    for c in caps:
        evs = (_animated_events(c, st) if st.get("animation", "None") != "None"
               else [(c["start"], c["end"], _ass_text(c["text"]))])
        lines += [f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Default,,0,0,0,,{t}" for a, b, t in evs]
    events = "\n".join(lines)
    return (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\n"
            f"ScaledBorderAndShadow: yes\nYCbCr Matrix: None\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            f"Alignment, MarginL, MarginR, MarginV, Encoding\n{style}\n\n[Events]\n"
            f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n{events}\n")


def _ass_time(t):
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def prepare_subs_dir(caps, width, height, st):
    """A temporary folder holding subs.ass and the chosen font's files, so FFmpeg
    (run inside it) needs no awkward paths in its filter and always finds the font."""
    d = tempfile.mkdtemp(prefix="subgen-")
    with open(os.path.join(d, "subs.ass"), "w", encoding="utf-8") as f:
        f.write(to_ass(caps, width, height, st))
    os.makedirs(os.path.join(d, "fonts"))
    files = font_files(st["font"]) if sys.platform in ("win32", "darwin") else None
    for path in files or []:  # elsewhere fontconfig finds the font by name
        try:
            shutil.copy(path, os.path.join(d, "fonts"))
        except OSError:
            pass
    return d


SUB_FILTER = "subtitles=subs.ass:fontsdir=fonts"


# --------------------------------------------------------------------------- #
#  Downloads: the models, and NVIDIA's cuBLAS
# --------------------------------------------------------------------------- #
class Cancelled(Exception):
    pass


class DownloadError(Exception):
    pass


def _ssl_context():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_file(url, dest, size, sha256, on_progress, check_cancel):
    """Download to dest, checking its SHA-256. Written under a temporary name, so a
    cancelled or broken download is never used."""
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "Subtitle-Generator"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as r, open(part, "wb") as f:
            done = 0
            while True:
                check_cancel()
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                on_progress(done)
        if h.hexdigest() != sha256:
            raise DownloadError("The downloaded file is damaged (its checksum doesn't match). Try again.")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)


def model_dir(cfg):
    return os.path.join(HOME, "models", cfg["name"])


def model_ready(cfg):
    d = model_dir(cfg)
    return all(os.path.isfile(os.path.join(d, fn)) and os.path.getsize(os.path.join(d, fn)) == size
               for fn, size, _ in cfg["files"])


def download_model(cfg, on_progress, check_cancel):
    total = sum(s for _, s, _ in cfg["files"])
    base = 0
    for fn, size, sha in cfg["files"]:
        dest = os.path.join(model_dir(cfg), fn)
        if not (os.path.isfile(dest) and os.path.getsize(dest) == size):
            download_file(HF.format(repo=cfg["repo"], rev=cfg["rev"], file=fn), dest, size, sha,
                          lambda d, b=base: on_progress(b + d, total), check_cancel)
        base += size


def cuda_dirs():
    """Folders that may hold cuBLAS: the app's own download, and (running from
    source) the nvidia-cublas-cu12 pip package."""
    dirs = [os.path.join(HOME, "cuda")]
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cublas")
        if spec and spec.submodule_search_locations:
            dirs += [os.path.join(p, "bin") for p in spec.submodule_search_locations]
    except Exception:
        pass
    return dirs


def cuda_ready():
    if sys.platform != "win32":
        return True  # Linux: the system's CUDA (if any); macOS never offers the graphics card
    return any(all(os.path.isfile(os.path.join(d, n)) for n in CUDA_WHEEL["dlls"]) for d in cuda_dirs())


def use_cuda_dirs():
    """Let CTranslate2 find cuBLAS when it loads it."""
    for d in cuda_dirs():
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except (AttributeError, OSError):
                pass
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def download_cuda(on_progress, check_cancel):
    """NVIDIA's cuBLAS package is a zip: keep just the two libraries Whisper needs."""
    import zipfile
    dest_dir = os.path.join(HOME, "cuda")
    wheel = os.path.join(HOME, "cublas.whl")
    download_file(CUDA_WHEEL["url"], wheel, CUDA_WHEEL["size"], CUDA_WHEEL["sha256"],
                  lambda d: on_progress(d, CUDA_WHEEL["size"]), check_cancel)
    try:
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(wheel) as z:
            for member in z.namelist():
                name = member.rsplit("/", 1)[-1]
                if name in CUDA_WHEEL["dlls"]:
                    part = os.path.join(dest_dir, name + ".part")
                    with z.open(member) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    os.replace(part, os.path.join(dest_dir, name))
    finally:
        os.remove(wheel)
    if not cuda_ready():
        raise DownloadError("NVIDIA's package didn't contain the expected files.")


def gpu_present():
    """True if there's an NVIDIA graphics card (with its driver) that CTranslate2 can use."""
    if sys.platform == "darwin":
        return False
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def explain_error(e):
    text = str(e)
    if isinstance(e, PermissionError):
        return "Permission denied — the file may be open in another program."
    if isinstance(e, MemoryError) or "bad allocation" in text or "out of memory" in text.lower():
        return "Not enough memory. Try the Fast model, or close other programs."
    if "getaddrinfo" in text or "urlopen" in repr(e) or "timed out" in text:
        return "Couldn't download. Check your internet connection and try again."
    return text or type(e).__name__


def fmt_mb(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def fmt_duration(secs):
    secs = int(round(secs))
    h, rest = divmod(secs, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def enable_drag_and_drop(root, on_drop):
    """Accept files dragged onto the window, using tkinterdnd2 if it's installed."""
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        TkinterDnD.require(root)
    except Exception:
        return False

    def drop(event):
        on_drop(list(root.tk.splitlist(event.data)))
        return event.action  # "copy": the files stay where they are
    widgets, todo = [], [root]
    while todo:
        w = todo.pop()
        widgets.append(w)
        todo += w.winfo_children()
    for w in widgets:
        try:
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<Drop>>", drop)
        except Exception:
            pass
    return True


# --------------------------------------------------------------------------- #
#  The look: white cards on a faint lavender, a dark purple header, blue actions
# --------------------------------------------------------------------------- #
THEME = dict(
    bg="#f3f1f9", card="#ffffff", border="#e2ddf0", text="#1c1433", muted="#6e6787",
    purple="#2b1655", purple2="#4a2a8a", blue="#2f7cf6", blue_hover="#1d67dc", blue_soft="#e7f0ff",
    blue_text="#1756c4", lavender="#d9ccff", lavender_line="#7b5cd6", disabled="#c9c4d8", now="#f1ecff",
    video="#140b2b")


def ui_family():
    try:
        families = set(tkfont.families())
    except tk.TclError:
        return "TkDefaultFont"
    for name in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Cantarell", "DejaVu Sans"):
        if name in families:
            return name
    return "TkDefaultFont"


def _indicator_images(root, style):
    """Checkboxes and radio buttons drawn as images: a blue box with a white tick, a blue
    ring with a dot (clam draws a tick as an X)."""
    try:
        from PIL import Image as PILImage, ImageDraw, ImageTk as PILImageTk
    except ImportError:
        return
    t, n, k = THEME, 18, 4  # drawn 4x and shrunk, for smooth edges
    big = n * k

    def make(kind, on, enabled):
        im = PILImage.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        blue = t["blue"] if enabled else t["disabled"]
        box = (k, k, big - k, big - k)
        if kind == "check":
            if on:
                d.rounded_rectangle(box, radius=5 * k, fill=blue)
                d.line([(5 * k, 9.5 * k), (8 * k, 12.5 * k), (13.5 * k, 6 * k)], fill="#ffffff", width=int(2.2 * k),
                       joint="curve")
            else:
                d.rounded_rectangle(box, radius=5 * k, fill="#ffffff", outline=t["disabled"] if not enabled
                                    else "#b9b0d6", width=int(1.4 * k))
        else:
            if on:
                d.ellipse(box, fill="#ffffff", outline=blue, width=int(1.6 * k))
                d.ellipse((6 * k, 6 * k, big - 6 * k, big - 6 * k), fill=blue)
            else:
                d.ellipse(box, fill="#ffffff", outline=t["disabled"] if not enabled else "#b9b0d6", width=int(1.4 * k))
        return PILImageTk.PhotoImage(im.resize((n, n), PILImage.LANCZOS), master=root)
    root._indicator_images = imgs = {(kind, on, en): make(kind, on, en) for kind in ("check", "radio")
                                     for on in (False, True) for en in (False, True)}
    for kind, widget in (("check", "Checkbutton"), ("radio", "Radiobutton")):
        name = f"Card.{kind}.indicator"
        try:
            style.element_create(name, "image", imgs[(kind, False, True)],
                                 ("disabled", "selected", imgs[(kind, True, False)]),
                                 ("disabled", imgs[(kind, False, False)]),
                                 ("selected", imgs[(kind, True, True)]), padding=(0, 0, 6, 0), sticky="")
        except tk.TclError:
            pass  # already made
        style.layout(f"Card.T{widget}", [(f"{widget}.padding", {"sticky": "nswe", "children": [
            (name, {"side": "left", "sticky": ""}),
            (f"{widget}.focus", {"side": "left", "sticky": "w", "children": [
                (f"{widget}.label", {"sticky": "nswe"})]})]})])


def apply_theme(root):
    """ttk's "clam" theme, recoloured (the native themes ignore colours)."""
    t, fam = THEME, ui_family()
    root.configure(bg=t["bg"])
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=t["bg"], foreground=t["text"], font=(fam, 10), bordercolor=t["border"],
                focuscolor=t["blue"], lightcolor=t["card"], darkcolor=t["border"], troughcolor=t["bg"])
    s.configure("Card.TFrame", background=t["card"])
    s.configure("TLabel", background=t["bg"])
    s.configure("Card.TLabel", background=t["card"])
    s.configure("Muted.Card.TLabel", background=t["card"], foreground=t["muted"], font=(fam, 9))
    s.configure("File.Card.TLabel", background=t["card"], foreground=t["text"], font=(fam, 11, "bold"))
    s.configure("Status.TLabel", background=t["bg"], foreground=t["muted"], font=(fam, 9))
    for name in ("Card.TLabel", "Muted.Card.TLabel", "File.Card.TLabel"):
        s.map(name, background=[("disabled", t["card"])], foreground=[("disabled", t["disabled"])])
    _indicator_images(root, s)
    for kind in ("TCheckbutton", "TRadiobutton"):
        s.configure(f"Card.{kind}", background=t["card"], foreground=t["text"], indicatorbackground=t["card"],
                    indicatorforeground=t["blue"], upperbordercolor=t["border"], lowerbordercolor=t["border"])
        s.map(f"Card.{kind}", background=[("active", t["card"])],
              indicatorbackground=[("selected", t["blue"]), ("active", t["blue_soft"])],
              indicatorforeground=[("selected", "#ffffff")],
              foreground=[("disabled", t["disabled"])])
    s.configure("TCombobox", fieldbackground="#ffffff", background="#ffffff", foreground=t["text"],
                arrowcolor=t["purple2"], bordercolor=t["border"], padding=(6, 3), selectbackground="#ffffff",
                selectforeground=t["text"])
    s.map("TCombobox", fieldbackground=[("readonly", "#ffffff"), ("disabled", t["bg"])],
          foreground=[("disabled", t["disabled"])], arrowcolor=[("disabled", t["disabled"])],
          bordercolor=[("focus", t["blue"]), ("hover", t["lavender_line"])])
    s.configure("TEntry", fieldbackground="#ffffff", bordercolor=t["border"], padding=(6, 3))
    s.map("TEntry", bordercolor=[("focus", t["blue"])], fieldbackground=[("disabled", t["bg"])])
    s.configure("TButton", background="#ffffff", foreground=t["purple2"], bordercolor=t["border"],
                padding=(12, 5), font=(fam, 10, "bold"), relief="flat")
    s.map("TButton", background=[("disabled", t["card"]), ("pressed", t["now"]), ("active", "#f5f1ff")],
          foreground=[("disabled", t["disabled"])], bordercolor=[("active", t["lavender_line"])])
    s.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground=t["text"],
                rowheight=28, borderwidth=0, font=(fam, 10))
    s.map("Treeview", background=[("selected", t["blue_soft"])], foreground=[("selected", t["blue_text"])])
    s.configure("Treeview.Heading", background=t["card"], foreground=t["muted"], font=(fam, 9, "bold"),
                relief="flat", borderwidth=0, padding=(6, 4))
    s.map("Treeview.Heading", background=[("active", t["card"])])
    s.configure("Horizontal.TProgressbar", troughcolor="#e9e5f5", background=t["blue"], bordercolor="#e9e5f5",
                lightcolor=t["blue"], darkcolor=t["blue"], thickness=6)
    s.configure("Vertical.TScrollbar", background="#ddd6ee", troughcolor=t["card"], bordercolor=t["card"],
                arrowcolor=t["muted"], lightcolor="#ddd6ee", darkcolor="#ddd6ee", gripcount=0)
    root.option_add("*TCombobox*Listbox.background", "#ffffff")
    root.option_add("*TCombobox*Listbox.foreground", t["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", t["blue"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", (fam, 10))
    return fam


class PillButton(tk.Canvas):
    """A rounded button: blue for the main actions, dark purple for the player."""
    COLOURS = {"primary": ("blue", "blue_hover", "#ffffff"), "dark": ("purple2", "purple", "#ffffff")}

    def __init__(self, parent, text, command, kind="primary", bg=None, height=38, padx=22, font=None):
        bg = bg or THEME["card"]
        self._font = tkfont.Font(font=font or (ui_family(), 11, "bold"))
        width = self._font.measure(text) + 2 * padx
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.text, self.command, self.kind = text, command, kind
        self.enabled, self.hover = True, False
        self._bw, self._bh = width, height
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", self._click)
        self._draw()

    def _round_rect(self, x0, y0, x1, y1, r, **kw):
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
               x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        return self.create_polygon(pts, smooth=True, **kw)

    def _draw(self):
        self.delete("all")
        base, hover, fg = self.COLOURS[self.kind]
        fill = THEME["disabled"] if not self.enabled else THEME[hover if self.hover else base]
        self._round_rect(1, 1, self._bw - 1, self._bh - 1, self._bh // 2, fill=fill, outline=fill)
        self.create_text(self._bw / 2, self._bh / 2, text=self.text, fill=fg, font=self._font)

    def _set_hover(self, on):
        self.hover = on
        self._draw()

    def _click(self, event):
        if self.enabled and 0 <= event.x <= self._bw and 0 <= event.y <= self._bh:
            self.command()

    def set_text(self, text):
        self.text = text
        self._draw()

    def set_enabled(self, on):
        self.enabled = bool(on)
        self.config(cursor="hand2" if on else "")
        self._draw()


def card(parent, number, title):
    """A white card with a numbered title; returns (card, body)."""
    t = THEME
    outer = tk.Frame(parent, bg=t["card"], highlightthickness=1, highlightbackground=t["border"])
    head = tk.Frame(outer, bg=t["card"])
    head.pack(fill="x", padx=14, pady=(12, 4))
    badge = tk.Canvas(head, width=24, height=24, bg=t["card"], highlightthickness=0)
    badge.create_oval(1, 1, 23, 23, fill=t["blue"], outline=t["blue"])
    badge.create_text(12, 12, text=str(number), fill="#ffffff", font=(ui_family(), 10, "bold"))
    badge.pack(side="left")
    tk.Label(head, text=title, bg=t["card"], fg=t["purple2"], font=(ui_family(), 12, "bold")).pack(side="left", padx=8)
    body = ttk.Frame(outer, style="Card.TFrame")
    body.pack(fill="both", expand=True, padx=14, pady=(2, 12))
    return outer, body, head


def header(parent, on_support):
    """The app's bar: dark purple fading to blue, with the name and a support link."""
    t, fam = THEME, ui_family()
    c = tk.Canvas(parent, height=64, highlightthickness=0, bd=0, bg=t["purple"])

    def draw(event=None):
        c.delete("all")
        w = max(c.winfo_width(), 400)
        a, b = (0x2b, 0x16, 0x55), (0x2f, 0x5c, 0xd6)
        steps = 64
        for i in range(steps):
            k = (i / (steps - 1)) ** 2.2  # mostly purple, turning blue on the right
            col = "#%02x%02x%02x" % tuple(round(a[j] + (b[j] - a[j]) * k) for j in range(3))
            c.create_rectangle(i * w / steps, 0, (i + 1) * w / steps + 1, 64, fill=col, outline=col)
        # The logo: a white speech bubble with "CC".
        c.create_polygon(18, 14, 52, 14, 56, 18, 56, 40, 52, 44, 30, 44, 22, 51, 24, 44, 18, 44, 14, 40, 14, 18,
                         smooth=True, fill="#ffffff", outline="#ffffff")
        c.create_text(35, 29, text="CC", fill=t["purple2"], font=(fam, 12, "bold"))
        c.create_text(70, 22, text="Subtitle Generator", anchor="w", fill="#ffffff", font=(fam, 16, "bold"))
        c.create_text(71, 44, text="AI subtitles on your own computer · free, private, no limits", anchor="w",
                      fill="#cfc6ff", font=(fam, 9))
        link = c.create_text(w - 18, 32, text="♥ Support this project", anchor="e", fill="#ffffff",
                             font=(fam, 10, "underline"), tags="support")
        c.tag_bind("support", "<Button-1>", lambda e: on_support())
        c.tag_bind("support", "<Enter>", lambda e: c.config(cursor="hand2"))
        c.tag_bind("support", "<Leave>", lambda e: c.config(cursor=""))
        return link
    c.bind("<Configure>", draw)
    return c


# --------------------------------------------------------------------------- #
#  The player: the video with its captions, and its sound, in step
# --------------------------------------------------------------------------- #
PLAY_RATE = 24000  # the player's sound: mono, 16-bit


class Player:
    """Plays the video's picture (small frames from FFmpeg, with the captions drawn by
    libass, exactly as they'll be burned in) and its sound (sounddevice), keeping the
    picture on the sound's clock. Thread-safe enough for one UI thread calling it."""
    FPS = 30

    def __init__(self, ffmpeg):
        self.ffmpeg = ffmpeg
        self.path = self.info = None
        self.audio = None  # int16 samples at PLAY_RATE, once read
        self.t = 0.0  # the playhead while paused
        self.playing = False
        self.stop_at = None  # pause here (playing one caption)
        self.size = (2, 2)
        self._gen = 0  # frames from an older pipe are dropped
        self._proc = self._dir = None
        self._frames = queue.Queue(maxsize=12)
        self._pending = []
        self._stream = None
        self._pos = 0  # sound samples handed out
        self._latency = 0.0
        self._mark = None  # (sample, sound card time it's heard at), from the last block
        self._t0 = self._wall0 = 0.0
        self._subs = None  # () -> subs folder or None: the captions to draw

    # ---- setting up
    def load(self, path, info, size, subs):
        self.stop()
        self.path, self.info, self.size, self._subs = path, info, size, subs
        self.audio, self.t = None, 0.0
        threading.Thread(target=self._read_sound, args=(path,), daemon=True).start()

    def _read_sound(self, path):
        try:
            r = subprocess.run([self.ffmpeg, "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar",
                                str(PLAY_RATE), "-f", "s16le", "pipe:1"], capture_output=True, **_POPEN_FLAGS)
            if path == self.path:
                self.audio = np.frombuffer(r.stdout[:len(r.stdout) // 2 * 2], np.int16)
        except Exception as e:
            log_error(f"Player sound: {type(e).__name__}: {e}")

    @property
    def duration(self):
        return (self.info or {}).get("duration") or 0.0

    # ---- the clock
    def clock(self):
        if not self.playing:
            return self.t
        if self._stream is not None:
            mark = self._mark
            if mark is not None:
                try:  # the sample being heard: the last block's first sample plus the time since it began
                    idx, dac = mark
                    return self._t0 + max(0.0, idx / PLAY_RATE + (self._stream.time - dac))
                except Exception:
                    pass
            return self._t0 + max(0.0, self._pos / PLAY_RATE - self._latency)
        return self._t0 + (time.monotonic() - self._wall0)

    # ---- controls
    def play(self, stop_at=None):
        if not self.path:
            return
        if self.t >= self.duration - 0.05:
            self.t = 0.0
        self.stop_at = stop_at
        self._start(self.t)

    def pause(self):
        if self.playing:
            self.t = min(self.clock(), self.duration)
        self.playing = False
        self.stop_at = None
        self._stop_sound()
        self._stop_video()

    def seek(self, t):
        t = max(0.0, min(t, self.duration))
        if self.playing:
            self._start(t)
        else:
            self.t = t

    def refresh(self):
        """The captions changed: draw the new ones from here on."""
        if self.playing:
            self._start_video(self.clock())

    def stop(self):
        self.pause()
        self.path = None

    def _start(self, t):
        self._stop_sound()
        self._t0, self._wall0, self.playing = t, time.monotonic(), True
        self._start_video(t)
        self._start_sound(t)

    # ---- sound
    def _start_sound(self, t):
        self._pos, self._mark = 0, None
        if self.audio is None:
            return
        try:
            import sounddevice as sd
        except Exception:
            return  # no sound library: the picture plays on the computer's clock
        start = int(t * PLAY_RATE)
        audio = self.audio

        def fill(out, frames, when, _status):
            i = start + self._pos
            chunk = audio[i:i + frames]
            out[:len(chunk), 0] = chunk
            out[len(chunk):, 0] = 0
            if when is not None:
                self._mark = (self._pos, when.outputBufferDacTime)  # this block is heard from then
            self._pos += frames
        try:
            self._stream = sd.OutputStream(samplerate=PLAY_RATE, channels=1, dtype="int16", callback=fill,
                                           latency="low")
            self._latency = float(self._stream.latency or 0)
            self._stream.start()
        except Exception as e:  # no sound device: play the picture anyway
            log_error(f"Player sound device: {type(e).__name__}: {e}")
            self._stream = None
            self._wall0 = time.monotonic()

    def _stop_sound(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None

    # ---- picture
    def _start_video(self, t):
        self._stop_video()
        self._gen += 1
        gen = self._gen
        w, h = self.size
        vf = f"fps={self.FPS},scale={w}:{h},setpts=PTS+{t:.3f}/TB"
        d = self._subs() if self._subs else None
        if d:
            vf += "," + SUB_FILTER
        try:
            self._proc = subprocess.Popen(
                [self.ffmpeg, "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", self.path, "-an", "-sn", "-vf", vf,
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=d, **_POPEN_FLAGS)
        except OSError as e:
            log_error(f"Player picture: {e}")
            return
        self._dir = d
        proc = self._proc
        threading.Thread(target=self._read_frames, args=(proc, gen, t, w * h * 3), daemon=True).start()

    def _read_frames(self, proc, gen, t, size):
        n = 0
        while gen == self._gen:
            data = proc.stdout.read(size)
            if len(data) < size:
                break
            while gen == self._gen:
                try:
                    self._frames.put((gen, t + n / self.FPS, data), timeout=0.2)
                    break
                except queue.Full:
                    continue
            n += 1

    def _stop_video(self):
        self._gen += 1
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        self._proc = None
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None
        self._pending = []
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    def frame_for(self, t):
        """The newest frame due by time t (bytes), or None if none is new."""
        best = None
        while True:
            if not self._pending:
                try:
                    item = self._frames.get_nowait()
                except queue.Empty:
                    break
                if item[0] != self._gen:
                    continue
                self._pending.append(item)
            gen, ft, data = self._pending[0]
            if ft > t + 0.5 / self.FPS:
                break
            best = data
            self._pending.pop(0)
        return best


class Timeline(tk.Canvas):
    """The captions as blocks along the video's time, with the playhead. Click to jump
    there (or to pick a caption), drag to scrub."""
    H = 64
    WINDOW = 30.0  # seconds shown at once in a long video

    def __init__(self, parent, on_seek, on_pick):
        t = THEME
        super().__init__(parent, height=self.H, bg=t["card"], highlightthickness=0, bd=0)
        self.on_seek, self.on_pick = on_seek, on_pick
        self.caps, self.duration, self.playhead, self.selected = [], 0.0, 0.0, None
        self.view = (0.0, 1.0)
        self._drag = False
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)

    def set(self, caps=None, duration=None, playhead=None, selected=-1):
        if caps is not None:
            self.caps = caps
        if duration is not None:
            self.duration = duration
        if playhead is not None:
            self.playhead = playhead
        if selected != -1:
            self.selected = selected
        self._follow()
        self.draw()

    def _follow(self):
        d = self.duration
        if self._drag:
            return  # never scroll under the mouse
        if d <= 60:
            self.view = (0.0, max(d, 0.01))
            return
        a, b = self.view
        if b - a != self.WINDOW or not (a + self.WINDOW * 0.1 <= self.playhead <= b - self.WINDOW * 0.15):
            a = max(0.0, min(self.playhead - self.WINDOW * 0.25, d - self.WINDOW))
            self.view = (a, a + self.WINDOW)

    def _x(self, t):
        a, b = self.view
        w = max(self.winfo_width(), 50)
        return 8 + (t - a) / (b - a) * (w - 16)

    def _t(self, x):
        a, b = self.view
        w = max(self.winfo_width(), 50)
        return a + (x - 8) / (w - 16) * (b - a)

    def draw(self):
        t = THEME
        self.delete("all")
        w = max(self.winfo_width(), 50)
        if self.duration <= 0:
            self.create_text(w / 2, self.H / 2, text="The timeline appears when a video is open.", fill=t["muted"],
                             font=(ui_family(), 9))
            return
        fam = ui_family()
        # Overview: the whole video, and the part shown.
        self.create_rectangle(8, 2, w - 8, 6, fill="#ece8f7", outline="")
        a, b = self.view
        d = self.duration
        self.create_rectangle(8 + a / d * (w - 16), 2, 8 + b / d * (w - 16), 6, fill=t["lavender_line"], outline="")
        # Caption blocks.
        self.create_rectangle(8, 14, w - 8, 44, fill="#f6f4fc", outline="")
        for k, c in enumerate(self.caps):
            if c["end"] < a or c["start"] > b:
                continue
            x0, x1 = self._x(c["start"]), max(self._x(c["end"]), self._x(c["start"]) + 2)
            sel = k == self.selected
            self.create_rectangle(x0, 16, x1, 42, fill=t["blue"] if sel else t["lavender"],
                                  outline=t["blue_hover"] if sel else t["lavender_line"], tags=("cap", f"c{k}"))
            if x1 - x0 > 30:
                label = c["text"].replace("\n", " ")
                room = int((x1 - x0 - 8) / 6)
                if room > 2:
                    self.create_text(x0 + 4, 29, text=label[:room] + ("…" if len(label) > room else ""),
                                     anchor="w", fill="#ffffff" if sel else t["purple"], font=(fam, 8),
                                     tags=("cap", f"c{k}"))
        # Seconds along the bottom.
        span = b - a
        step = next(s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800) if span / s <= 12)
        s0 = int(a // step + 1) * step
        for s in range(int(s0), int(b) + 1, step):
            x = self._x(s)
            self.create_line(x, 46, x, 50, fill=t["disabled"])
            self.create_text(x, 57, text=fmt_duration(s), fill=t["muted"], font=(fam, 8))
        # The playhead.
        x = self._x(self.playhead)
        self.create_line(x, 10, x, 48, fill=t["purple"], width=2)
        self.create_polygon(x - 5, 9, x + 5, 9, x, 15, fill=t["purple"], outline=t["purple"])

    def _press(self, e):
        if self.duration <= 0:
            return
        if e.y <= 8:  # the overview: jump anywhere
            w = max(self.winfo_width(), 50)
            self.on_seek(max(0.0, min(self.duration, (e.x - 8) / (w - 16) * self.duration)), done=True)
            return
        hit = [tag for item in self.find_overlapping(e.x, e.y, e.x, e.y) for tag in self.gettags(item)
               if tag.startswith("c") and tag != "cap"]
        if hit and 14 <= e.y <= 44:
            self.on_pick(int(hit[-1][1:]))
            return
        self._drag = True
        self.on_seek(max(0.0, min(self.duration, self._t(e.x))), done=False)

    def _motion(self, e):
        if self._drag:
            self.on_seek(max(0.0, min(self.duration, self._t(e.x))), done=False)

    def _release(self, e):
        if self._drag:
            t = max(0.0, min(self.duration, self._t(e.x)))  # in the view the press was made in
            self._drag = False
            self.on_seek(t, done=True)


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
class SubtitleApp:
    VIDEO_W, VIDEO_H = 448, 252  # the player's picture

    def __init__(self, root):
        self.root = root
        root.title("Subtitle Generator")
        self.missing = load_libraries()
        self.family = apply_theme(root)
        self.ffmpeg = find_ffmpeg() if "imageio-ffmpeg" not in self.missing else None
        self.ui_queue = queue.Queue()
        self.cancelled = False
        self.running = False
        self.video = None  # dict(path, info)
        self.caps = []
        self.editing = None  # index of the caption in the editor
        self.now_row = None  # the caption the player is on
        self.lang = None  # the language spoken (detected or chosen)
        self.dirty = False
        self.models = {}  # (model label, device) -> loaded WhisperModel
        self.player = Player(self.ffmpeg) if self.ffmpeg else None
        self.preview_req = None
        self.preview_cv = threading.Condition()
        self.video_photo = None
        self._setting_preset = False
        self._refresh_job = None
        self.has_gpu = gpu_present() if not self.missing else False
        self._pump_ui()
        threading.Thread(target=self._preview_worker, daemon=True).start()
        root.report_callback_exception = self._report_error
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        home = os.path.expanduser("~")
        videos = next((p for p in (os.path.join(home, "Videos"), os.path.join(home, "Movies")) if os.path.isdir(p)), home)
        fonts = available_fonts()
        clean = PRESETS["Clean"]
        self.v = {
            "model": tk.StringVar(value="Accurate — Whisper large-v3 turbo" if self.has_gpu else "Fast — Whisper small"),
            "language": tk.StringVar(value="Detect automatically"),
            "device": tk.StringVar(value=GPU_DEVICE if self.has_gpu else CPU_DEVICE),
            "mode": tk.StringVar(value=list(CAPTION_MODES)[0]),
            "srt": tk.BooleanVar(value=True), "vtt": tk.BooleanVar(value=False),
            "track": tk.BooleanVar(value=False), "burn": tk.BooleanVar(value=False),
            "preset": tk.StringVar(value="Clean"),
            "font": tk.StringVar(value=clean["font"] if clean["font"] in fonts else fonts[0]),
            "size": tk.StringVar(value=clean["size"]), "bold": tk.BooleanVar(value=clean["bold"]),
            "colour": tk.StringVar(value=clean["colour"]), "look": tk.StringVar(value=clean["look"]),
            "position": tk.StringVar(value=clean["position"]), "animation": tk.StringVar(value=clean["animation"]),
            "highlight": tk.StringVar(value=clean["highlight"]),
            "where": tk.StringVar(value="same"), "out_dir": tk.StringVar(value=videos),
            "reveal": tk.BooleanVar(value=True),
        }
        self.fonts = fonts
        self._build_ui()
        self.v["model"].trace_add("write", lambda *_: self._on_model())
        self.v["where"].trace_add("write", lambda *_: self._on_where())
        self.v["preset"].trace_add("write", lambda *_: self._on_preset())
        for k in ("colour", "highlight"):
            self.v[k].trace_add("write", lambda *_, k=k: self.swatches[k].config(bg=self.v[k].get()))
        for k in ("font", "size", "bold", "colour", "look", "position", "animation", "highlight"):
            self.v[k].trace_add("write", lambda *_: self._on_style())
        for k in ("srt", "vtt", "track", "burn"):
            self.v[k].trace_add("write", lambda *_: self._on_outputs())
        self._on_model()
        self._on_where()
        self._on_outputs()
        self._update_buttons()
        self.dnd = enable_drag_and_drop(root, self._on_drop)
        self.drop_lbl.config(text="Drag a video or audio file here, or click Choose." if self.dnd
                             else "Click Choose to pick a video or audio file.")
        for seq, fn in (("<space>", lambda: self.toggle_play()), ("<Left>", lambda: self.skip(-2)),
                        ("<Right>", lambda: self.skip(2)), ("<bracketleft>", lambda: self.set_edge("start")),
                        ("<bracketright>", lambda: self.set_edge("end"))):
            root.bind(seq, lambda e, fn=fn: None if isinstance(e.widget, (tk.Entry, ttk.Entry, tk.Text,
                                                                          ttk.Combobox)) else fn())
        if self._required_missing():
            self.status.config(text="Some components are missing.")
            root.after(200, self.offer_install)
        elif not self.ffmpeg:
            self.status.config(text="FFmpeg is missing or too old.")
            root.after(200, lambda: messagebox.showerror(
                "FFmpeg problem", "The FFmpeg bundled with imageio-ffmpeg couldn't be used, and no suitable FFmpeg "
                "(with H.264 and subtitle drawing) is installed. Try reinstalling: "
                f'"{sys.executable}" -m pip install --force-reinstall imageio-ffmpeg'))

    # ---- layout --------------------------------------------------------- #
    def _build_ui(self):
        t, fam = THEME, self.family
        import webbrowser
        header(self.root, lambda: webbrowser.open(SUPPORT_URL)).pack(fill="x")
        main = tk.Frame(self.root, bg=t["bg"])
        main.pack(fill="both", expand=True, padx=14, pady=(12, 6))
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(1, weight=1)
        gap = 6

        # 1. Video
        c1, f, _ = card(main, 1, "Video or audio")
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, gap), pady=(0, gap * 2))
        f.columnconfigure(1, weight=1)
        self.choose_btn = ttk.Button(f, text="Choose…", command=self.pick_video)
        self.choose_btn.grid(row=0, column=0, sticky="w")
        self.open_subs_btn = ttk.Button(f, text="Open subtitle file…", command=self.open_subtitles)
        self.open_subs_btn.grid(row=0, column=1, sticky="w", padx=8)
        self.file_lbl = ttk.Label(f, text="No video yet", style="File.Card.TLabel")
        self.file_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.drop_lbl = ttk.Label(f, text="", style="Muted.Card.TLabel", wraplength=380)
        self.drop_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # 2. Recognition
        c2, f, _ = card(main, 2, "Write the subtitles")
        c2.grid(row=0, column=1, sticky="nsew", padx=(gap, 0), pady=(0, gap * 2))
        f.columnconfigure(4, weight=1)
        rows = (("Model", "model", list(MODELS), 33, 0, 0), ("Language", "language", list(LANGUAGES), 20, 0, 2),
                ("Run on", "device", [CPU_DEVICE] + ([GPU_DEVICE] if self.has_gpu else []), 33, 1, 0),
                ("Captions", "mode", list(CAPTION_MODES), 40, 1, 2))
        for label, key, values, width, row, col in rows:
            ttk.Label(f, text=label, style="Card.TLabel").grid(row=row, column=col, sticky="w", padx=(0 if col == 0 else 14, 6), pady=3)
            cb = ttk.Combobox(f, values=values, textvariable=self.v[key], state="readonly", width=width, height=20)
            cb.grid(row=row, column=col + 1, sticky="w", pady=3)
            if key == "device":
                self.device_cb = cb
        self.go_btn = PillButton(f, "Make subtitles", self.transcribe)
        self.go_btn.grid(row=0, column=5, rowspan=2, sticky="e", padx=(12, 0))
        self.model_lbl = ttk.Label(f, text="", style="Muted.Card.TLabel", wraplength=760)
        self.model_lbl.grid(row=2, column=0, columnspan=6, sticky="w", pady=(6, 0))

        # 3. Check and fix: the list, the player, the editor, the timeline
        c3, f, _ = card(main, 3, "Check and fix")
        c3.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, gap * 2))
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        listf = ttk.Frame(f, style="Card.TFrame")
        listf.grid(row=0, column=0, sticky="nsew")
        listf.columnconfigure(0, weight=1)
        listf.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(listf, columns=("start", "end", "text"), show="headings", selectmode="browse")
        for col, text, width, anchor, stretch in (("start", "Start", 78, "e", False), ("end", "End", 78, "e", False),
                                                  ("text", "Caption", 360, "w", True)):
            self.tree.heading(col, text=text, anchor="w" if col == "text" else "e")
            self.tree.column(col, width=width, anchor=anchor, stretch=stretch)
        self.tree.tag_configure("now", background=t["now"])
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(listf, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())
        self.tree.bind("<Double-1>", lambda e: self.play_line())
        self.list_lbl = ttk.Label(listf, text="", style="Muted.Card.TLabel")
        self.list_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        side = ttk.Frame(f, style="Card.TFrame")
        side.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
        box = tk.Frame(side, bg=t["video"], width=self.VIDEO_W, height=self.VIDEO_H)
        box.grid(row=0, column=0, sticky="n")
        box.pack_propagate(False)
        self.screen = tk.Label(box, bg=t["video"], bd=0, fg="#a99fd0", text="Open a video to see it here",
                               font=(fam, 10))
        self.screen.pack(fill="both", expand=True)
        bar = ttk.Frame(side, style="Card.TFrame")
        bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.play_btn = PillButton(bar, "▶  Play", self.toggle_play, kind="dark", height=32, padx=16,
                                   font=(fam, 10, "bold"))
        self.play_btn.pack(side="left")
        self.transport = []
        for text, cmd, tip in (("⟲ 2 s", lambda: self.skip(-2), "Back 2 seconds (Left arrow)"),
                               ("2 s ⟳", lambda: self.skip(2), "Forward 2 seconds (Right arrow)")):
            b = ttk.Button(bar, text=text, command=cmd, width=6)
            b.pack(side="left", padx=(6, 0))
            self.transport.append(b)
        self.time_lbl = ttk.Label(bar, text="0:00.0 / 0:00.0", style="Muted.Card.TLabel", width=17, anchor="e")
        self.time_lbl.pack(side="right")

        ed = ttk.Frame(side, style="Card.TFrame")
        ed.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ed.columnconfigure(5, weight=1)
        ttk.Label(ed, text="From", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.start_e = ttk.Entry(ed, width=12)
        self.start_e.grid(row=0, column=1, sticky="w", padx=(6, 4))
        ttk.Label(ed, text="to", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(4, 0))
        self.end_e = ttk.Entry(ed, width=12)
        self.end_e.grid(row=0, column=3, sticky="w", padx=6)
        for e in (self.start_e, self.end_e):
            e.bind("<Return>", lambda ev: self._apply_times())
            e.bind("<FocusOut>", lambda ev: self._apply_times())
        times = ttk.Frame(ed, style="Card.TFrame")
        times.grid(row=0, column=5, sticky="e")
        self.edge_btns = []
        for text, cmd in (("Start here [", lambda: self.set_edge("start")), ("End here ]", lambda: self.set_edge("end")),
                          ("− 0.1 s", lambda: self.nudge(-0.1)), ("+ 0.1 s", lambda: self.nudge(0.1))):
            b = ttk.Button(times, text=text, command=cmd)
            b.pack(side="left", padx=(4, 0))
            self.edge_btns.append(b)
        self.text_w = tk.Text(ed, height=3, width=46, wrap="word", font=(fam, 12), undo=True, relief="flat",
                              highlightthickness=1, highlightbackground=t["border"], highlightcolor=t["blue"],
                              bg="#ffffff", fg=t["text"], insertbackground=t["blue"], padx=8, pady=6,
                              selectbackground=t["blue_soft"], selectforeground=t["text"])
        self.text_w.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(8, 6))
        self.text_w.bind("<KeyRelease>", lambda e: self._apply_text())
        btns = ttk.Frame(ed, style="Card.TFrame")
        btns.grid(row=2, column=0, columnspan=6, sticky="ew")
        self.edit_btns = []
        for text, cmd in (("Split at cursor", self.split_line), ("Join with next", self.merge_next),
                          ("Delete", self.delete_line)):
            b = ttk.Button(btns, text=text, command=cmd)
            b.pack(side="left", padx=(0, 6))
            self.edit_btns.append(b)
        self.edit_hint = ttk.Label(btns, text="Space plays and pauses. Double-click a caption to hear it.",
                                   style="Muted.Card.TLabel")
        self.edit_hint.pack(side="right")
        self.timeline = Timeline(f, self._on_timeline_seek, self._on_timeline_pick)
        self.timeline.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        # 4. Save
        c4, f, _ = card(main, 4, "Save")
        c4.grid(row=2, column=0, columnspan=2, sticky="ew")
        f.columnconfigure(0, weight=1)
        outs = ttk.Frame(f, style="Card.TFrame")
        outs.grid(row=0, column=0, sticky="w")
        for key, text in (("srt", "Subtitle file (.srt)"), ("vtt", "Web subtitle file (.vtt)"),
                          ("track", "Video with subtitles you can switch on"),
                          ("burn", "Video with captions burned in")):
            self.__dict__[f"{key}_cb"] = ttk.Checkbutton(outs, text=text, variable=self.v[key], style="Card.TCheckbutton")
            self.__dict__[f"{key}_cb"].pack(side="left", padx=(0, 20))
        look = self.look_frame = ttk.Frame(f, style="Card.TFrame")
        look.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.swatches = {}
        row1 = (("Look", "preset", list(PRESETS) + ["Custom"], 10), ("Animation", "animation", ANIMATIONS, 19))
        row2 = (("Font", "font", self.fonts, 12), ("Size", "size", list(SIZES), 10), ("Style", "look", LOOKS, 18),
                ("Position", "position", list(POSITIONS), 8))
        for r, items in enumerate((row1, row2)):
            line = ttk.Frame(look, style="Card.TFrame")
            line.grid(row=r, column=0, sticky="w", pady=2)
            if r == 0:
                ttk.Label(line, text="Burned-in captions", style="Card.TLabel",
                          font=(fam, 10, "bold")).pack(side="left", padx=(0, 10))
            for label, key, values, width in items:
                ttk.Label(line, text=label, style="Card.TLabel").pack(side="left", padx=(8, 4))
                ttk.Combobox(line, values=values, textvariable=self.v[key], state="readonly",
                             width=width).pack(side="left")
            if r == 0:
                self._swatch(line, "Word colour", "highlight")
                self.anim_note = ttk.Label(line, text="", style="Muted.Card.TLabel")
                self.anim_note.pack(side="left", padx=(12, 0))
            else:
                ttk.Checkbutton(line, text="Bold", variable=self.v["bold"], style="Card.TCheckbutton").pack(
                    side="left", padx=(12, 0))
                self._swatch(line, "Colour", "colour")
        where = ttk.Frame(f, style="Card.TFrame")
        where.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        where.columnconfigure(2, weight=1)
        ttk.Radiobutton(where, text="Next to the video", value="same", variable=self.v["where"],
                        style="Card.TRadiobutton").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(where, text="In this folder:", value="folder", variable=self.v["where"],
                        style="Card.TRadiobutton").grid(row=0, column=1, sticky="w", padx=(14, 6))
        self.dir_e = ttk.Entry(where, textvariable=self.v["out_dir"])
        self.dir_e.grid(row=0, column=2, sticky="ew")
        self.dir_btn = ttk.Button(where, text="Choose…", command=self.pick_folder)
        self.dir_btn.grid(row=0, column=3, padx=8)
        self.save_btn = PillButton(where, "Save", self.save, padx=30)
        self.save_btn.grid(row=0, column=4, padx=(6, 0))

        foot = tk.Frame(self.root, bg=t["bg"])
        foot.pack(fill="x", padx=14, pady=(0, 10))
        foot.columnconfigure(1, weight=1)
        self.status = ttk.Label(foot, text="Ready.", style="Status.TLabel", wraplength=780)
        self.status.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(foot, mode="determinate", maximum=1000, length=240)
        self.progress.grid(row=0, column=2, sticky="e", padx=(12, 8))
        self.cancel_btn = ttk.Button(foot, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.grid(row=0, column=3, sticky="e")

    def _swatch(self, parent, label, key):
        ttk.Label(parent, text=label, style="Card.TLabel").pack(side="left", padx=(12, 4))
        sw = tk.Label(parent, width=3, relief="flat", bd=0, bg=self.v[key].get(), cursor="hand2",
                      highlightthickness=1, highlightbackground=THEME["border"])
        sw.pack(side="left", ipady=3)
        sw.bind("<Button-1>", lambda e: self.pick_colour(key))
        self.swatches[key] = sw

    # ---- settings helpers ------------------------------------------------ #
    def _on_outputs(self):
        burn = self.v["burn"].get()

        def walk(w):
            for child in w.winfo_children():
                yield child
                yield from walk(child)
        for w in walk(self.look_frame):
            try:
                if isinstance(w, ttk.Combobox):
                    w.config(state="readonly" if burn else "disabled")
                elif isinstance(w, (ttk.Checkbutton, ttk.Label)):
                    w.config(state="normal" if burn else "disabled")
            except tk.TclError:
                pass
        self._note_animation()
        self._captions_changed()

    def _note_animation(self):
        anim = self.v["animation"].get() != "None"
        text = ""
        if anim and self.v["burn"].get() and (self.v["srt"].get() or self.v["vtt"].get() or self.v["track"].get()):
            text = "Animation is for the burned-in video: subtitle files show plain text."
        self.anim_note.config(text=text)

    def _on_preset(self):
        name = self.v["preset"].get()
        if name not in PRESETS or self._setting_preset:
            return
        self._setting_preset = True
        try:
            for k, val in PRESETS[name].items():
                if k == "font" and val not in self.fonts:
                    val = self.fonts[0]
                self.v[k].set(val)
        finally:
            self._setting_preset = False
        self._on_style()

    def _on_style(self):
        if not self._setting_preset:
            current = self._style()
            match = next((n for n, p in PRESETS.items() if all(current[k] == p[k] for k in p)), "Custom")
            if self.v["preset"].get() != match:
                self._setting_preset = True  # just the name: the settings are already these
                try:
                    self.v["preset"].set(match)
                finally:
                    self._setting_preset = False
        self._note_animation()
        self._captions_changed()

    def pick_colour(self, key="colour"):
        if not self.v["burn"].get():
            return
        _, hexcode = colorchooser.askcolor(color=self.v[key].get(),
                                           title="Caption colour" if key == "colour" else "Highlighted word colour")
        if hexcode:
            self.v[key].set(hexcode)

    def _style(self):
        return dict(font=self.v["font"].get(), size=self.v["size"].get(), bold=self.v["bold"].get(),
                    colour=self.v["colour"].get(), look=self.v["look"].get(), position=self.v["position"].get(),
                    animation=self.v["animation"].get(), highlight=self.v["highlight"].get())

    def _shown_style(self):
        """How the player draws the captions: as burned in, or plainly (like a player shows a subtitle file)."""
        if self.v["burn"].get():
            return self._style()
        return dict(font=self.fonts[0], size="Medium", bold=False, colour="#ffffff", look="Box behind the text",
                    position="Bottom", animation="None", highlight="#ffffff")

    def _player_subs(self):
        """A folder with the captions for the player to draw (called as it starts)."""
        if not self.video or not self.video["info"]["width"] or not self.caps:
            return None
        info = self.video["info"]
        return prepare_subs_dir([dict(c) for c in self.caps], info["width"], info["height"], self._shown_style())

    def _captions_changed(self):
        """The captions or their look changed: redraw the frame, and the player soon after."""
        self._request_preview()
        self.timeline.set(caps=self.caps)
        if self.player is not None and self.player.playing:
            if self._refresh_job:
                self.root.after_cancel(self._refresh_job)
            self._refresh_job = self.root.after(300, self.player.refresh)

    # ---- choosing the video ------------------------------------------------ #
    def _use_file(self, path):
        if self.running or self._required_missing() or not self.ffmpeg:
            if self._required_missing():
                self.offer_install()
            return
        if not self._confirm_discard():
            return
        info = probe_media(self.ffmpeg, path)
        if info["error"]:
            messagebox.showwarning("Can't use this file", f"{os.path.basename(path)}: {info['error']}")
            return
        self.player.stop()
        self.video = dict(path=path, info=info)
        self.lang = None
        kind = f"{info['width']}×{info['height']} video" if info["width"] else "audio"
        self.file_lbl.config(text=os.path.basename(path))
        self.drop_lbl.config(text=f"{kind}, {fmt_duration(info['duration'])}. Drop another file to switch.")
        for key in ("track", "burn"):
            self.__dict__[f"{key}_cb"].config(state="normal" if info["width"] else "disabled")
            if not info["width"]:
                self.v[key].set(False)
        if info["width"]:
            scale = min(self.VIDEO_W / info["width"], self.VIDEO_H / info["height"])
            size = (max(2, int(info["width"] * scale) // 2 * 2), max(2, int(info["height"] * scale) // 2 * 2))
        else:
            size = (2, 2)
        self.player.load(path, info, size, self._player_subs)
        self.video_photo = None
        self._set_caps([], dirty=False)
        self.timeline.set(caps=[], duration=info["duration"], playhead=0.0, selected=None)
        self._show_time()
        if not info["width"]:
            self.screen.config(image="", text="Audio only: no picture")
        self.status.config(text="Click Make subtitles, or Open subtitle file to use ones you already have.")
        self._update_buttons()

    # ---- the caption list -------------------------------------------------- #
    def _set_caps(self, caps, dirty=True, keep=None):
        self.caps = caps
        self.dirty = dirty
        self.now_row = None
        self.tree.delete(*self.tree.get_children())
        for i, c in enumerate(caps):
            self.tree.insert("", "end", iid=str(i), values=self._row(c))
        if keep is not None and caps:
            k = str(min(keep, len(caps) - 1))
            self.tree.selection_set(k)
            self.tree.see(k)
        else:
            self._load_editor(None)
        self._update_list_label()
        self._update_buttons()
        self._captions_changed()

    def _row(self, c):
        return (fmt_short(c["start"]), fmt_short(c["end"]), c["text"].replace("\n", "  ⏎  "))

    def _sel(self):
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def _on_select(self):
        k = self._sel()
        self._load_editor(k)
        if k is not None and k < len(self.caps) and self.player is not None and not self.running:
            c = self.caps[k]
            t = self.player.clock()
            if not (c["start"] <= t < c["end"]):  # jump to it, unless the player is already in it
                self.player.seek(min(c["start"] + 0.12, c["end"]))  # just after a fade-in
            self._show_time()
        self.timeline.set(selected=k)

    def _load_editor(self, k):
        self.editing = k if k is not None and k < len(self.caps) else None
        self.text_w.config(state="normal")  # a disabled widget ignores changes
        for e in (self.start_e, self.end_e):
            e.config(state="normal")
            e.delete(0, "end")
        self.text_w.delete("1.0", "end")
        if self.editing is not None:
            c = self.caps[self.editing]
            self.start_e.insert(0, fmt_ts(c["start"], "."))
            self.end_e.insert(0, fmt_ts(c["end"], "."))
            self.text_w.insert("1.0", c["text"])
            self.text_w.edit_reset()
        self._update_buttons()
        self._request_preview()

    def _apply_text(self):
        k = self.editing
        if k is None or k >= len(self.caps) or self.running:
            return
        c = self.caps[k]
        text = self.text_w.get("1.0", "end-1c").strip()
        if text and text != c["text"]:
            c["words"] = retime_words(caption_words(c), text, c["start"], c["end"])
            c["text"] = text
            self.tree.item(str(k), values=self._row(c))
            self.dirty = True
            self._update_list_label()
            self._captions_changed()

    def _change_times(self, k, start, end):
        """Give caption k new times (its words follow), keeping the list in time order."""
        c = self.caps[k]
        c["words"] = fit_words(caption_words(c), c["start"], c["end"], start, end)
        c["start"], c["end"] = start, end
        order = sorted(range(len(self.caps)), key=lambda i: self.caps[i]["start"])
        self._set_caps([self.caps[i] for i in order], keep=order.index(k))

    def _apply_times(self):
        k = self.editing
        if k is None or k >= len(self.caps) or self.running:
            return
        c = self.caps[k]
        try:
            start, end = parse_ts(self.start_e.get()), parse_ts(self.end_e.get())
            if end <= start:
                raise ValueError
        except ValueError:
            self.status.config(text="Times look like 00:01:02.500 (hours:minutes:seconds), and the end must be "
                                    "after the start.")
            return
        if abs(start - c["start"]) < 0.0005 and abs(end - c["end"]) < 0.0005:
            return
        self._change_times(k, start, end)

    def set_edge(self, which):
        """The selected caption starts (or ends) where the player is."""
        k = self.editing
        if k is None or self.running or self.player is None:
            return
        c, t = self.caps[k], round(self.player.clock(), 3)
        start, end = (t, c["end"]) if which == "start" else (c["start"], t)
        if end - start < 0.1:
            self.status.config(text="A caption has to end after it starts: move the player first.")
            return
        self._change_times(k, start, end)
        self.status.config(text=f"Caption {k + 1} now {'starts' if which == 'start' else 'ends'} at {fmt_short(t)}.")

    def nudge(self, secs):
        k = self.editing
        if k is None or self.running:
            return
        c = self.caps[k]
        if c["start"] + secs < 0:
            return
        self._change_times(k, c["start"] + secs, c["end"] + secs)

    def split_line(self):
        k = self.editing
        if k is None or self.running:
            return
        c = self.caps[k]
        text = self.text_w.get("1.0", "end-1c")
        pos = len(self.text_w.get("1.0", "insert"))
        a, b = text[:pos].strip(), text[pos:].strip()
        if not a or not b:
            self.status.config(text="Put the text cursor where the caption should be split.")
            return
        words = retime_words(caption_words(c), (a + " " + b), c["start"], c["end"])
        n = len(_tokens(a))
        mid = words[n][0] if n < len(words) else c["start"] + (c["end"] - c["start"]) * len(a) / (len(a) + len(b))
        m = CAPTION_MODES[self.v["mode"].get()]
        first = dict(start=c["start"], end=mid, text=break_lines(a.replace("\n", " "), m["line"], m["lines"]),
                     words=words[:n])
        second = dict(start=mid, end=c["end"], text=break_lines(b.replace("\n", " "), m["line"], m["lines"]),
                      words=words[n:])
        self._set_caps(self.caps[:k] + [first, second] + self.caps[k + 1:], keep=k)

    def merge_next(self):
        k = self.editing
        if k is None or k + 1 >= len(self.caps) or self.running:
            return
        a, b = self.caps[k], self.caps[k + 1]
        m = CAPTION_MODES[self.v["mode"].get()]
        text = (a["text"] + " " + b["text"]).replace("\n", " ")
        merged = dict(start=a["start"], end=max(a["end"], b["end"]), text=break_lines(text, m["line"], 2),
                      words=caption_words(a) + caption_words(b))
        self._set_caps(self.caps[:k] + [merged] + self.caps[k + 2:], keep=k)

    def delete_line(self):
        k = self.editing
        if k is None or self.running:
            return
        self._set_caps(self.caps[:k] + self.caps[k + 1:], keep=k if k < len(self.caps) - 1 else k - 1)

    # ---- the player ----------------------------------------------------------- #
    def play_line(self):
        """Play the selected caption, then pause at its end."""
        k = self._sel()
        if k is None or not self.video or self.running:
            return
        c = self.caps[k]
        self.player.pause()
        self.player.seek(max(0.0, c["start"] - 0.15))
        self.player.play(stop_at=c["end"] + 0.1)
        self._started()

    def toggle_play(self):
        if self.player is None or not self.video or self.running:
            return
        if self.player.playing:
            self.player.pause()
            self._stopped()
        else:
            self.player.play()
            self._started()

    def skip(self, secs):
        if self.player is None or not self.video or self.running:
            return
        self.player.seek(self.player.clock() + secs)
        self._show_time()
        if not self.player.playing:
            self._request_preview()

    def _started(self):
        self.play_btn.set_text("❚❚  Pause")
        self._tick()

    def _stopped(self):
        self.play_btn.set_text("▶  Play")
        self._show_time()
        self._request_preview()

    def _tick(self):
        p = self.player
        if p is None or not p.playing:
            return
        t = p.clock()
        if (p.stop_at is not None and t >= p.stop_at) or t >= p.duration:
            p.pause()
            self._stopped()
            return
        data = p.frame_for(t)
        if data is not None:
            w, h = p.size
            im = Image.frombuffer("RGB", (w, h), data, "raw", "RGB", 0, 1)
            if self.video_photo is None or self.video_photo.width() != w or self.video_photo.height() != h:
                self.video_photo = ImageTk.PhotoImage(im)
                self.screen.config(image=self.video_photo, text="")
            else:
                self.video_photo.paste(im)
        self._show_time(t)
        self.root.after(12, self._tick)

    def _show_time(self, t=None):
        if self.player is None:
            return
        t = self.player.clock() if t is None else t
        self.time_lbl.config(text=f"{fmt_short(t)} / {fmt_short(self.player.duration)}")
        self.timeline.set(playhead=t)
        k = next((i for i, c in enumerate(self.caps) if c["start"] <= t < c["end"]), None)
        if k != self.now_row:
            for row in (self.now_row, k):
                if row is not None and self.tree.exists(str(row)):
                    self.tree.item(str(row), tags=("now",) if row == k else ())
            self.now_row = k
            if k is not None and self.player.playing:
                self.tree.see(str(k))

    def _on_timeline_seek(self, t, done):
        if self.player is None or self.running:
            return
        was = self.player.playing
        if not done:  # scrubbing: show where, play from there when let go
            if was:
                self.player.pause()
                self._scrub_resume = True
            self.player.seek(t)
            self._show_time(t)
            self._request_preview()
            return
        self.player.seek(t)
        self._show_time(t)
        if getattr(self, "_scrub_resume", False) or was:
            self._scrub_resume = False
            if not self.player.playing:
                self.player.play()
            self._started()
        else:
            self._request_preview()

    def _on_timeline_pick(self, k):
        if 0 <= k < len(self.caps):
            self.tree.selection_set(str(k))
            self.tree.see(str(k))
            if self.player is not None:
                self.player.seek(min(self.caps[k]["start"] + 0.12, self.caps[k]["end"]))
                self._show_time()
                if not self.player.playing:
                    self._request_preview()

    def _update_buttons(self):
        busy = self.running
        has_file = self.video is not None
        self.go_btn.set_enabled(has_file and not busy)
        self.open_subs_btn.config(state="normal" if has_file and not busy else "disabled")
        self.choose_btn.config(state="disabled" if busy else "normal")
        self.save_btn.set_enabled(bool(self.caps) and not busy)
        sel = self.editing is not None and not busy
        for b in self.edit_btns + self.edge_btns:
            b.config(state="normal" if sel else "disabled")
        playable = has_file and not busy
        self.play_btn.set_enabled(playable)
        for b in self.transport:
            b.config(state="normal" if playable else "disabled")
        state = "normal" if sel else "disabled"
        self.text_w.config(state=state, bg="#ffffff" if sel else THEME["bg"])
        for e in (self.start_e, self.end_e):
            e.config(state=state)

    # ---- the paused frame ---------------------------------------------------------- #
    def _request_preview(self):
        """While paused: the frame at the playhead, with the captions as the player draws them."""
        if not getattr(self, "screen", None) or not self.video or not self.video["info"]["width"]:
            return
        if self.player is not None and self.player.playing:
            return
        t = self.player.clock() if self.player else 0.0
        near = [dict(c) for c in self.caps if c["end"] >= t - 1 and c["start"] <= t + 1]
        with self.preview_cv:
            self.preview_req = (self.video["path"], self.video["info"], t, near, self._shown_style(),
                                self.player.size)
            self.preview_cv.notify()

    def _preview_worker(self):
        while True:
            with self.preview_cv:
                while self.preview_req is None:
                    self.preview_cv.wait()
                req, self.preview_req = self.preview_req, None
            time.sleep(0.08)  # typing or scrubbing: wait for a pause
            with self.preview_cv:
                if self.preview_req is not None:
                    continue
            try:
                im = self._render_preview(*req)
                self.ui(lambda im=im: self._show_frame(im))
            except Exception as e:
                log_error(f"Preview: {type(e).__name__}: {e}")

    def _render_preview(self, path, info, t, caps, style, size):
        import io
        pw, ph = size
        t = max(0.0, min(t, (info.get("duration") or 0) - 0.25))  # past the last frame there's nothing to show
        # Seeking restarts the frame clock at 0: put it back on the video's time for the captions.
        vf = f"scale={pw}:{ph},setpts=PTS+{max(0.0, t):.3f}/TB"
        cwd = None
        if caps:
            cwd = prepare_subs_dir(caps, info["width"], info["height"], style)
            vf += "," + SUB_FILTER
        try:
            r = subprocess.run([self.ffmpeg, "-nostdin", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", path,
                                "-frames:v", "1", "-vf", vf, "-f", "image2pipe", "-c:v", "png", "pipe:1"],
                               capture_output=True, timeout=30, cwd=cwd, **_POPEN_FLAGS)
        finally:
            if cwd:
                shutil.rmtree(cwd, ignore_errors=True)
        if not r.stdout:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
        return Image.open(io.BytesIO(r.stdout)).convert("RGB")

    def _show_frame(self, im):
        if self.player is not None and self.player.playing:
            return  # the player draws its own
        self.video_photo = ImageTk.PhotoImage(im)
        self.screen.config(image=self.video_photo, text="")

    def on_close(self):
        if self.running and not messagebox.askyesno("Still working", "Stop and close?"):
            return
        if not self.running and not self._confirm_discard():
            return
        self.cancel()
        if self.player is not None:
            self.player.stop()
        self.root.destroy()


    def _required_missing(self):
        return [m for m in self.missing if m not in ("tkinterdnd2", "sounddevice")]  # both optional

    # ---- dependency installer ------------------------------------------ #
    def offer_install(self):
        if getattr(sys, "frozen", False):
            messagebox.showerror("Missing components", "Missing: " + ", ".join(self.missing) +
                                 "\n\nThis copy of Subtitle Generator is incomplete. Please download it again.")
            return
        names = ", ".join(self.missing)
        if not messagebox.askyesno(
                "Install components?",
                f"Subtitle Generator needs: {names}\n\nInstall now? (about 150 MB download)\n\n"
                f"This runs pip for the Python running this app:\n{sys.executable}"):
            self.status.config(text="Components missing — nothing can be done until they're installed.")
            return
        self._set_busy(True)
        self.cancel_btn.config(state="disabled")
        self.status.config(text="Installing… this can take a few minutes.")
        packages = list(self.missing)

        def work():
            try:
                ok, tail = install_requirements(packages)
            except Exception as e:
                ok, tail = False, [str(e)]
            self.ui(lambda: self._install_done(ok, tail))
        threading.Thread(target=work, daemon=True).start()

    def _install_done(self, ok, tail):
        self._set_busy(False)
        self.missing = load_libraries()
        if ok and not self._required_missing():
            self.ffmpeg = find_ffmpeg()
            self.has_gpu = gpu_present()
            self.device_cb.config(values=[CPU_DEVICE] + ([GPU_DEVICE] if self.has_gpu else []))
            if not self.dnd:
                self.dnd = enable_drag_and_drop(self.root, self._on_drop)
            self.status.config(text="Installed! Ready.")
        else:
            self.status.config(text="Install failed — see the error message.")
            messagebox.showerror("Install failed", "Couldn't install automatically:\n\n" + "\n".join(tail) +
                                 "\n\nTry running this in a terminal:\n"
                                 f'"{sys.executable}" -m pip install ' + " ".join(PACKAGES))

    # ---- settings helpers ------------------------------------------------ #
    def _on_model(self):
        cfg = MODELS[self.v["model"].get()]
        text = cfg["about"]
        if not model_ready(cfg):
            text += "  (Downloads the first time you use it.)"
        if not self.has_gpu and sys.platform != "darwin":
            text += "  No NVIDIA graphics card was found, so it runs on the processor."
        self.model_lbl.config(text=text)

    def _on_where(self):
        state = "normal" if self.v["where"].get() == "folder" else "disabled"
        self.dir_e.config(state=state)
        self.dir_btn.config(state=state)

    def pick_folder(self):
        d = filedialog.askdirectory(title="Choose output folder", initialdir=self.v["out_dir"].get())
        if d:
            self.v["out_dir"].set(d)

    def ui(self, fn):
        """Run fn on the UI thread (Tkinter isn't thread-safe; workers only queue callbacks)."""
        self.ui_queue.put(fn)

    def _pump_ui(self):
        self.root.after(50, self._pump_ui)  # reschedule first, so one failing callback can't stop the pump
        while True:
            try:
                fn = self.ui_queue.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                self._report_error(*sys.exc_info())

    # ---- choosing the video ------------------------------------------------ #
    def pick_video(self):
        path = filedialog.askopenfilename(title="Choose a video or audio file", filetypes=FILE_TYPES)
        if path:
            self._use_file(path)

    def _on_drop(self, paths):
        if self.running:
            self.status.config(text="Wait for this to finish first.")
            return
        media = [p for p in paths if os.path.splitext(p)[1].lower() in VIDEO_EXTS | AUDIO_EXTS]
        subs = [p for p in paths if os.path.splitext(p)[1].lower() in SUB_EXTS]
        if media:
            self._use_file(media[0])
        if subs and (media or self.video):
            self._load_subtitles(subs[0])
        elif subs:
            self.status.config(text="Add the video first, then its subtitle file.")
        elif not media:
            self.status.config(text="That isn't a video or audio file this app knows.")
        elif len(media) > 1:
            self.status.config(text=f"Using {os.path.basename(media[0])} (one video at a time).")

    def _confirm_discard(self):
        if self.dirty and self.caps:
            return messagebox.askyesno("Unsaved subtitles", "The subtitles you made haven't been saved. "
                                       "Continue and lose them?")
        return True

    def open_subtitles(self):
        if self.running:
            return
        if not self.video:
            messagebox.showinfo("Open subtitles", "Choose the video first, then open its subtitle file.")
            return
        path = filedialog.askopenfilename(title="Open a subtitle file",
                                          filetypes=[("Subtitles", "*.srt *.vtt"), ("All files", "*.*")])
        if path:
            self._load_subtitles(path)

    def _load_subtitles(self, path):
        if not self._confirm_discard():
            return
        try:
            with open(path, "rb") as f:
                raw = f.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp1252", "replace")
            caps = parse_subtitles(text)
        except OSError as e:
            messagebox.showerror("Open subtitles", explain_error(e))
            return
        if not caps:
            messagebox.showwarning("Open subtitles", "No captions were found in that file.")
            return
        self._set_caps(caps, dirty=False)
        self.status.config(text=f"Opened {len(caps)} captions from {os.path.basename(path)}.")

    def _update_list_label(self):
        if not self.caps:
            text = "The captions appear here as they're written." if self.video else ""
        else:
            text = f"{len(self.caps)} captions"
            if self.lang:
                text += f", {LANG_NAMES.get(self.lang, self.lang)}"
            text += ". " + ("Edited — not saved yet." if self.dirty else "")
        self.list_lbl.config(text=text)

    # ---- making subtitles ---------------------------------------------------- #
    def _confirm_downloads(self, cfg, gpu):
        needs = []
        if not model_ready(cfg):
            needs.append(f"the {self.v['model'].get().split('—')[-1].strip()} speech model "
                         f"({fmt_mb(sum(s for _, s, _ in cfg['files']))}, from Hugging Face)")
        if gpu and not cuda_ready():
            needs.append(f"NVIDIA's cuBLAS library for the graphics card ({fmt_mb(CUDA_WHEEL['size'])}, "
                         "from NVIDIA's official package; about 770 MB once unpacked)")
        if not needs:
            return True
        return messagebox.askyesno(
            "One-time download", "This needs a one-time download of " + " and ".join(needs) +
            ".\n\nAfter that it works offline. Your video is never uploaded.\n\nDownload now?")

    def transcribe(self):
        if self.running or not self.video:
            return
        if self.caps and not self._confirm_discard():
            return
        label = self.v["model"].get()
        cfg = MODELS[label]
        gpu = self.v["device"].get() == GPU_DEVICE and self.has_gpu
        if not self._confirm_downloads(cfg, gpu):
            return
        self._set_caps([], dirty=False)
        self.lang = None
        self._launch(lambda: self._transcribe_worker(label, cfg, gpu, LANGUAGES[self.v["language"].get()],
                                                     self.v["mode"].get()))

    def _load_model(self, label, cfg, gpu):
        key = (label, gpu)
        if key not in self.models:
            self.ui(lambda: self.status.config(text="Loading the speech model…"))
            if gpu:
                use_cuda_dirs()
            model = WhisperModel(model_dir(cfg), device="cuda" if gpu else "cpu",
                                 compute_type="float16" if gpu else "int8",
                                 cpu_threads=0 if gpu else (os.cpu_count() or 4))
            self.models = {key: model}  # one at a time: they're big
        return self.models[key]

    def _transcribe_worker(self, label, cfg, gpu, lang, mode):
        t0 = time.monotonic()
        try:
            if not model_ready(cfg):
                def prog(done, total):
                    self.ui(lambda: (self.progress.config(value=done / total * 1000), self.status.config(
                        text=f"Downloading the speech model: {fmt_mb(done)} of {fmt_mb(total)}…")))
                download_model(cfg, prog, self._check_cancel)
                self.ui(self._on_model)
            if gpu and not cuda_ready():
                def prog(done, total):
                    self.ui(lambda: (self.progress.config(value=done / total * 1000), self.status.config(
                        text=f"Downloading NVIDIA's cuBLAS: {fmt_mb(done)} of {fmt_mb(total)}…")))
                download_cuda(prog, self._check_cancel)
            t0 = time.monotonic()  # the speed shown doesn't count the downloads
            self.ui(lambda: (self.progress.config(value=0), self.status.config(text="Reading the sound…")))
            audio = read_audio(self.ffmpeg, self.video["path"], self._check_cancel)
            duration = len(audio) / SAMPLE_RATE
            note = ""
            try:
                words = self._recognise(label, cfg, gpu, audio, duration, lang, mode, t0)
            except RuntimeError as e:
                if not gpu or "cancel" in str(e).lower():
                    raise
                # The graphics card couldn't be used (old driver, not enough memory...): use the processor.
                log_error(f"GPU failed, using CPU: {e}")
                note = " The graphics card couldn't be used, so the processor was used instead."
                self.models.clear()
                words = self._recognise(label, cfg, False, audio, duration, lang, mode, t0)
            caps = build_captions(words, mode)
            secs = time.monotonic() - t0
            self.ui(lambda: self._transcribed(caps, secs, duration, note))
        except Cancelled:
            self.ui(lambda: self._finish("Cancelled."))
        except Exception as e:
            log_error(f"Transcribe {self.video['path']}\n{type(e).__name__}: {e}")
            self.ui(lambda e=e: self._finish(f"Couldn't make subtitles: {explain_error(e)}", error=True))

    def _recognise(self, label, cfg, gpu, audio, duration, lang, mode, t0):
        model = self._load_model(label, cfg, gpu)
        where = "graphics card" if gpu else "processor"
        self.ui(lambda: self.status.config(text=f"Listening (on the {where})…"))
        segments, info = model.transcribe(audio, language=lang, word_timestamps=True, vad_filter=True,
                                          condition_on_previous_text=False, beam_size=5)
        self.lang = info.language
        words, last_ui, began = [], 0.0, time.monotonic()  # time left is judged from listening alone
        for seg in segments:
            self._check_cancel()
            for w in seg.words or []:
                words.append((float(w.start), float(w.end), w.word))
            if not seg.words and seg.text.strip():
                words.append((float(seg.start), float(seg.end), " " + seg.text.strip()))
            now = time.monotonic()
            if now - last_ui > 0.4:
                last_ui = now
                caps = build_captions(words, mode)
                frac = min(1.0, float(seg.end) / max(duration, 0.1))
                elapsed = now - began
                left = f", about {fmt_duration(elapsed / frac - elapsed)} left" if frac > 0.1 else ""
                self.ui(lambda caps=caps, frac=frac, left=left: self._partial(caps, frac, where, left))
        return words

    def _partial(self, caps, frac, where, left):
        self._set_caps(caps)
        if caps:
            self.tree.see(str(len(caps) - 1))
        self.progress.config(value=frac * 1000)
        self.status.config(text=f"Listening (on the {where}): {frac * 100:.0f}%{left}…")

    def _transcribed(self, caps, secs, duration, note):
        self._set_busy(False)
        self._set_caps(caps, dirty=True, keep=0 if caps else None)
        self.progress.config(value=1000)
        if not caps:
            self.status.config(text="No speech was found in this file." + note)
            return
        lang = LANG_NAMES.get(self.lang, self.lang or "")
        self.status.config(text=f"Done in {fmt_duration(secs)} ({duration / max(secs, 0.1):.1f}× as fast as it plays): "
                                f"{len(caps)} captions in {lang}. Check them, then Save.{note}")

    # ---- saving --------------------------------------------------------------- #
    def _outputs(self):
        src = self.video["path"]
        folder = os.path.dirname(src) if self.v["where"].get() == "same" else self.v["out_dir"].get().strip()
        stem, ext = os.path.splitext(os.path.basename(src))
        ext = ext.lower()
        out = []
        if self.v["srt"].get():
            out.append(("srt", os.path.join(folder, stem + ".srt")))
        if self.v["vtt"].get():
            out.append(("vtt", os.path.join(folder, stem + ".vtt")))
        if self.video["info"]["width"]:
            if self.v["track"].get():
                keep = ext if ext in (".mp4", ".m4v", ".mov", ".mkv", ".webm") else ".mkv"
                out.append(("track", os.path.join(folder, f"{stem} (subtitles){keep}")))
            if self.v["burn"].get():
                out.append(("burn", os.path.join(folder, f"{stem} (captions).mp4")))
        return out

    def save(self):
        if self.running or not self.caps:
            return
        self._apply_text()
        if self.v["where"].get() == "folder" and not os.path.isdir(self.v["out_dir"].get().strip()):
            messagebox.showwarning("Check settings", "The output folder doesn't exist.")
            return
        outs = self._outputs()
        if not outs:
            messagebox.showwarning("Nothing to save", "Tick at least one thing to save.")
            return
        src = os.path.normcase(os.path.abspath(self.video["path"]))
        if any(os.path.normcase(os.path.abspath(p)) == src for _, p in outs):
            messagebox.showwarning("Check settings", "That would overwrite the original file.")
            return
        existing = [p for _, p in outs if os.path.exists(p)]
        if existing and not messagebox.askyesno(
                "Replace files?", "These already exist:\n\n" + "\n".join(os.path.basename(p) for p in existing) +
                "\n\nReplace them?"):
            return
        caps = [dict(c) for c in self.caps]
        self._launch(lambda: self._save_worker(outs, caps, self._style()))

    def _save_worker(self, outs, caps, style):
        info, src = self.video["info"], self.video["path"]
        done, first = [], None
        try:
            for k, (kind, path) in enumerate(outs):
                label = os.path.basename(path)
                self.ui(lambda label=label: self.status.config(text=f"Saving {label}…"))

                def prog(f, k=k):
                    self.ui(lambda: self.progress.config(value=(k + f) / len(outs) * 1000))
                part = path + ".part" + os.path.splitext(path)[1]
                try:
                    if kind in ("srt", "vtt"):
                        with open(part, "w", encoding="utf-8", newline="\n") as f:
                            f.write(to_srt(caps) if kind == "srt" else to_vtt(caps))
                    elif kind == "track":
                        self._add_track(src, part, caps, info, prog)
                    else:
                        self._burn(src, part, caps, info, style, prog)
                    os.replace(part, path)
                finally:
                    if os.path.exists(part):
                        os.remove(part)
                done.append(label)
                first = first or path
            self.ui(lambda: self._saved(done, first))
        except Cancelled:
            self.ui(lambda: self._finish("Cancelled." + (f" Saved: {', '.join(done)}." if done else "")))
        except Exception as e:
            log_error(f"Save\n{type(e).__name__}: {e}")
            self.ui(lambda e=e: self._finish(f"Couldn't save: {explain_error(e)}", error=True))

    def _add_track(self, src, dest, caps, info, prog):
        """Copy the video and sound as they are (no quality loss, quick) and add the subtitles as a track."""
        d = tempfile.mkdtemp(prefix="subgen-")
        try:
            srt = os.path.join(d, "subs.srt")
            with open(srt, "w", encoding="utf-8") as f:
                f.write(to_srt(caps))
            ext = os.path.splitext(dest)[1].lower()
            codec = {".mkv": "srt", ".webm": "webvtt"}.get(ext, "mov_text")
            fmt = {".mkv": "matroska", ".webm": "webm", ".mov": "mov"}.get(ext, "mp4")
            args = ["-i", src, "-i", srt, "-map", "0:v?", "-map", "0:a?", "-map", "1:0", "-c", "copy", "-c:s", codec,
                    "-disposition:s:0", "default"]
            if self.lang in ISO3:
                args += ["-metadata:s:s:0", f"language={ISO3[self.lang]}"]
            if fmt in ("mp4", "mov"):
                args += ["-movflags", "+faststart"]
            run_ffmpeg(self.ffmpeg, args + ["-f", fmt, dest], info["duration"], prog, self._check_cancel)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def _burn(self, src, dest, caps, info, style, prog):
        """Draw the captions onto the picture (a new H.264 MP4)."""
        d = prepare_subs_dir(caps, info["width"], info["height"], style)
        try:
            audio = ["-c:a", "copy"] if info["audio_codec"] in ("aac", "mp3") else ["-c:a", "aac", "-b:a", "192k"]
            args = ["-i", src, "-map", "0:v:0", "-map", "0:a?", "-vf", SUB_FILTER, "-c:v", "libx264", "-preset", "fast",
                    "-crf", "20", "-pix_fmt", "yuv420p"] + audio + ["-movflags", "+faststart", "-f", "mp4", dest]
            run_ffmpeg(self.ffmpeg, args, info["duration"], prog, self._check_cancel, cwd=d)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def _saved(self, done, first):
        self._set_busy(False)
        self.dirty = False
        self._update_list_label()
        self.progress.config(value=1000)
        self.status.config(text="Saved: " + ", ".join(done) + ".")
        if first and self.v["reveal"].get():
            reveal_file(first)

    # ---- running ------------------------------------------------------------- #
    def _launch(self, fn):
        self.cancelled = False
        self._set_busy(True)
        threading.Thread(target=fn, daemon=True).start()

    def _set_busy(self, busy):
        self.running = busy
        self.cancel_btn.config(state="normal" if busy else "disabled")
        if busy:
            self.progress.config(value=0)
        self._update_buttons()

    def cancel(self):
        self.cancelled = True
        if self.running:
            self.status.config(text="Stopping…")

    def _check_cancel(self, raise_it=True):
        if self.cancelled and raise_it:
            raise Cancelled()
        return self.cancelled

    def _finish(self, text, error=False):
        self._set_busy(False)
        self.progress.config(value=0)
        self.status.config(text=text)
        if error:
            messagebox.showerror("Subtitle Generator", text)

    def _report_error(self, exc, val, tb):
        import traceback
        self._set_busy(False)
        detail = "".join(traceback.format_exception(exc, val, tb))
        log_error(detail)
        self.status.config(text=f"Unexpected error: {val}")
        messagebox.showerror("Unexpected error", detail[-1500:])

LOG_PATH = os.path.join(tempfile.gettempdir(), "subtitle_generator_errors.log")


def log_error(text):
    """Append an error to a log file (errors can otherwise vanish without a console)."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{text}\n")
    except OSError:
        pass


def reveal_file(path):
    """Open the file's folder with the file selected."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except Exception:
        pass


def _say(text):
    """Self-test progress, for the build log (a windowed Windows app has no stderr)."""
    if sys.stderr:
        print(f"[self-test {time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True)


def self_test():
    """Used by the release build: exit code 0 if the packaged app can download the
    Fast model, read a video's sound, run Whisper on it, burn in animated captions, add a
    subtitle track, load the player's sound library and drag and drop."""
    try:
        if load_libraries():
            return 1
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return 2
        tmp = tempfile.mkdtemp()
        video = os.path.join(tmp, "t.mp4")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=3", "-f", "lavfi",
                        "-i", "sine=frequency=440:d=3", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a",
                        "aac", video], check=True, capture_output=True, **_POPEN_FLAGS)
        info = probe_media(ffmpeg, video)
        if info["error"] or info["width"] != 640:
            return 4
        cfg = MODELS["Fast — Whisper small"]
        if not model_ready(cfg):
            _say("downloading the Fast model")
            download_model(cfg, lambda d, t: None, lambda: None)
        audio = read_audio(ffmpeg, video, lambda: None)
        _say(f"read {len(audio) / SAMPLE_RATE:.1f} s of sound; loading Whisper")
        model = WhisperModel(model_dir(cfg), device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
        for vad in (False, True):
            _say(f"transcribing (voice detection {'on' if vad else 'off'})")
            segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=vad)
            list(segments)  # a tone has no words; this checks the whole pipeline runs
        _say("Whisper OK; burning in captions")
        caps = build_captions([(0.2, 0.6, " Hello"), (0.7, 1.0, " there,"), (1.1, 1.6, " everyone."),
                               (1.8, 2.2, " Again.")], list(CAPTION_MODES)[0])
        if len(caps) != 2 or caps[0]["text"] != "Hello there, everyone." or parse_subtitles(to_srt(caps)) != \
                [dict(start=c["start"], end=c["end"], text=c["text"]) for c in caps]:
            return 5
        style = dict(font=available_fonts()[0], size="Large", bold=True, colour="#ffff00", look="Outline",
                     position="Bottom", animation="Highlight each word", highlight="#4da3ff")
        if sum(1 for line in to_ass(caps, 640, 360, style).splitlines() if line.startswith("Dialogue")) != 4:
            return 8  # one event per word
        _say("animated captions OK; checking the player's sound library")
        import sounddevice  # noqa: F401  (bundled with its PortAudio library)
        d = prepare_subs_dir([dict(start=0, end=3, text="Self test")], 640, 360, style)
        burned = os.path.join(tmp, "burned.mp4")
        r = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", video, "-vf", SUB_FILTER, "-c:v", "libx264", "-c:a",
                            "copy", burned], capture_output=True, cwd=d, **_POPEN_FLAGS)
        plain = subprocess.run([ffmpeg, "-v", "error", "-ss", "1", "-i", video, "-frames:v", "1", "-f", "rawvideo",
                                "-pix_fmt", "gray", "pipe:1"], capture_output=True, **_POPEN_FLAGS).stdout
        drawn = subprocess.run([ffmpeg, "-v", "error", "-ss", "1", "-i", burned, "-frames:v", "1", "-f", "rawvideo",
                                "-pix_fmt", "gray", "pipe:1"], capture_output=True, **_POPEN_FLAGS).stdout
        if r.returncode != 0 or len(plain) != len(drawn) or \
                sum(abs(a - b) > 40 for a, b in zip(plain[-640 * 80:], drawn[-640 * 80:])) < 200:
            return 6  # the captions weren't drawn (no libass, or no font)
        srt = os.path.join(tmp, "s.srt")
        with open(srt, "w", encoding="utf-8") as f:
            f.write(to_srt(caps))
        track = os.path.join(tmp, "track.mp4")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", video, "-i", srt, "-map", "0:v", "-map", "0:a", "-map", "1:0",
                        "-c", "copy", "-c:s", "mov_text", track], check=True, capture_output=True, **_POPEN_FLAGS)
        if "Subtitle: mov_text" not in probe_text(ffmpeg, track):
            return 7
        _say("videos OK; loading drag and drop")
        root = tk.Tk()
        root.withdraw()
        from tkinterdnd2 import TkinterDnD
        TkinterDnD.require(root)
        root.destroy()
        _say("drag and drop OK")
        return 0
    except Exception as e:
        log_error(f"Self-test failed: {type(e).__name__}: {e}")
        return 3


def probe_text(ffmpeg, path):
    return subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True,
                          errors="replace", **_POPEN_FLAGS).stderr


def quit_now(code):
    """End the process. Whisper's engine (CTranslate2) can leave worker threads that keep
    a finished program from exiting on macOS, so skip the slow, hang-prone cleanup:
    every file has already been written and closed by this point."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


SUPPORT_URL = "https://ko-fi.com/collinbrooksdev"


def support_link(parent, bg=None, font=None):
    """A small "Support this project" link that opens the Ko-fi page in the browser."""
    import webbrowser
    kw = {"bg": bg} if bg else {}
    link = tk.Label(parent, text="\u2665 Support this project", fg="#c8364f", cursor="hand2",
                    font=font or ("", 9, "underline"), **kw)
    link.bind("<Button-1>", lambda e: webbrowser.open(SUPPORT_URL))
    return link


# The app's icon (64 px PNG), for the window and the taskbar. The .exe and the Mac app get
# theirs from assets/ when they're built.
APP_ICON = (
    "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"
    "RU5ErkJggg=="
)


def set_app_icon(root):
    try:
        if sys.platform == "win32":
            # Its own taskbar button and icon, not Python's, when run as a script.
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("OfflineToolkit.SubtitleGenerator")
        root._app_icon = tk.PhotoImage(master=root, data=APP_ICON)
        root.iconphoto(True, root._app_icon)
    except Exception:
        pass  # just the default icon


def main():
    if "--self-test" in sys.argv:
        code = self_test()
        _say(f"finished with code {code}")
        quit_now(code)
    root = tk.Tk()  # the theme is set by the app (apply_theme)
    set_app_icon(root)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = min(1260, sw - 40), min(960, sh - 80)
    root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
    root.minsize(1080, 760)
    app = SubtitleApp(root)
    files = [a for a in sys.argv[1:] if os.path.isfile(a)]
    if files:
        root.after(300, lambda: app._on_drop(files))
    root.mainloop()
    quit_now(0)


if __name__ == "__main__":
    main()
