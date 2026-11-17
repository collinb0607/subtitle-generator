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
    Pillow          -> shows the preview
    certifi         -> trusted certificates for downloading the models
    tkinterdnd2     -> drag and drop (optional: the app works without it)
    The app offers to install them on first run. Manual install, using the Python that runs the app:
        python -m pip install faster-whisper imageio-ffmpeg pillow certifi tkinterdnd2

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
from tkinter import colorchooser, filedialog, messagebox, ttk

# Filled in by load_libraries(), so the window can still open (and offer to
# install them) when they're missing.
np = WhisperModel = Image = ImageTk = None

PACKAGES = ["faster-whisper", "imageio-ffmpeg", "pillow", "certifi", "tkinterdnd2"]
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
                caps.append(dict(start=cur[0][0], end=cur[-1][1], text=break_lines(text, m["line"], m["lines"])))
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


def to_ass(caps, width, height, st):
    """An Advanced SubStation file: the captions in the chosen look, for burning in."""
    size = max(8, round(min(width, height) * SIZES[st["size"]]))
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
    events = "\n".join(f"Dialogue: 0,{_ass_time(c['start'])},{_ass_time(c['end'])},Default,,0,0,0,,{_ass_text(c['text'])}"
                       for c in caps)
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


class SoundPlayer:
    """Plays a short WAV file without blocking, and stops it."""
    def __init__(self):
        self.proc = None

    def play(self, path):
        self.stop()
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        for player in (["afplay"] if sys.platform == "darwin" else ["paplay", "aplay", "ffplay"]):
            exe = shutil.which(player)
            if exe:
                args = [exe, "-nodisp", "-autoexit", path] if player == "ffplay" else [exe, path]
                self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return

    def stop(self):
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
class SubtitleApp:
    PREVIEW_W, PREVIEW_H = 384, 216

    def __init__(self, root):
        self.root = root
        root.title("Subtitle Generator")
        self.missing = load_libraries()
        self.ffmpeg = find_ffmpeg() if "imageio-ffmpeg" not in self.missing else None
        self.ui_queue = queue.Queue()
        self.cancelled = False
        self.running = False
        self.video = None  # dict(path, info)
        self.caps = []
        self.editing = None  # index of the caption in the editor
        self.lang = None  # the language spoken (detected or chosen)
        self.dirty = False
        self.models = {}  # (model label, device) -> loaded WhisperModel
        self.player = SoundPlayer()
        self.tmp = tempfile.mkdtemp(prefix="subgen-ui-")
        self.preview_req = None
        self.preview_cv = threading.Condition()
        self.preview_photo = None
        self.has_gpu = gpu_present() if not self.missing else False
        self._pump_ui()
        threading.Thread(target=self._preview_worker, daemon=True).start()
        root.report_callback_exception = self._report_error
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        home = os.path.expanduser("~")
        videos = next((p for p in (os.path.join(home, "Videos"), os.path.join(home, "Movies")) if os.path.isdir(p)), home)
        fonts = available_fonts()
        self.v = {
            "model": tk.StringVar(value="Accurate — Whisper large-v3 turbo" if self.has_gpu else "Fast — Whisper small"),
            "language": tk.StringVar(value="Detect automatically"),
            "device": tk.StringVar(value=GPU_DEVICE if self.has_gpu else CPU_DEVICE),
            "mode": tk.StringVar(value=list(CAPTION_MODES)[0]),
            "srt": tk.BooleanVar(value=True), "vtt": tk.BooleanVar(value=False),
            "track": tk.BooleanVar(value=False), "burn": tk.BooleanVar(value=False),
            "font": tk.StringVar(value="Arial" if "Arial" in fonts else fonts[0]),
            "size": tk.StringVar(value="Medium"), "bold": tk.BooleanVar(value=True),
            "colour": tk.StringVar(value="#ffffff"), "look": tk.StringVar(value=LOOKS[0]),
            "position": tk.StringVar(value="Bottom"),
            "where": tk.StringVar(value="same"), "out_dir": tk.StringVar(value=videos),
            "reveal": tk.BooleanVar(value=True),
        }
        self.fonts = fonts
        self._build_ui()
        self.v["model"].trace_add("write", lambda *_: self._on_model())
        self.v["where"].trace_add("write", lambda *_: self._on_where())
        self.v["colour"].trace_add("write", lambda *_: self.swatch.config(bg=self.v["colour"].get()))
        for k in ("font", "size", "bold", "colour", "look", "position"):
            self.v[k].trace_add("write", lambda *_: self._request_preview())
        for k in ("srt", "vtt", "track", "burn"):
            self.v[k].trace_add("write", lambda *_: self._on_outputs())
        self._on_model()
        self._on_where()
        self._on_outputs()
        self._update_buttons()
        self.dnd = enable_drag_and_drop(root, self._on_drop)
        self.drop_lbl.config(text="Drag a video or audio file here, or click Choose." if self.dnd
                             else "Click Choose to pick a video or audio file.")
        if self._required_missing():
            self.status.config(text="Some components are missing.")
            root.after(200, self.offer_install)
        elif not self.ffmpeg:
            self.status.config(text="FFmpeg is missing or too old.")
            root.after(200, lambda: messagebox.showerror(
                "FFmpeg problem", "The FFmpeg bundled with imageio-ffmpeg couldn't be used, and no suitable FFmpeg "
                "(with H.264 and subtitle drawing) is installed. Try reinstalling: "
                f'"{sys.executable}" -m pip install --force-reinstall imageio-ffmpeg'))

    def _required_missing(self):
        return [m for m in self.missing if m != "tkinterdnd2"]

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

    # ---- layout --------------------------------------------------------- #
    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        # 1. Video
        f = ttk.LabelFrame(main, text="1. Video or audio", padding=8)
        f.grid(row=0, column=0, sticky="ew", **pad)
        f.columnconfigure(1, weight=1)
        self.choose_btn = ttk.Button(f, text="Choose…", command=self.pick_video)
        self.choose_btn.grid(row=0, column=0, sticky="w")
        self.file_lbl = ttk.Label(f, text="", font=("", 10, "bold"))
        self.file_lbl.grid(row=0, column=1, sticky="w", padx=8)
        self.drop_lbl = ttk.Label(f, text="", foreground="gray")
        self.drop_lbl.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.open_subs_btn = ttk.Button(f, text="Open subtitle file…", command=self.open_subtitles)
        self.open_subs_btn.grid(row=0, column=2, sticky="e")

        # 2. Recognition
        f = ttk.LabelFrame(main, text="2. Write the subtitles (runs on this computer)", padding=8)
        f.grid(row=1, column=0, sticky="ew", **pad)
        f.columnconfigure(5, weight=1)
        ttk.Label(f, text="Model").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(f, values=list(MODELS), textvariable=self.v["model"], state="readonly",
                     width=32).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(f, text="Language").grid(row=0, column=2, sticky="e", **pad)
        ttk.Combobox(f, values=list(LANGUAGES), textvariable=self.v["language"], state="readonly",
                     width=20, height=20).grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(f, text="Run on").grid(row=1, column=0, sticky="w", **pad)
        self.device_cb = ttk.Combobox(f, values=[CPU_DEVICE] + ([GPU_DEVICE] if self.has_gpu else []),
                                      textvariable=self.v["device"], state="readonly", width=32)
        self.device_cb.grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(f, text="Captions").grid(row=1, column=2, sticky="e", **pad)
        ttk.Combobox(f, values=list(CAPTION_MODES), textvariable=self.v["mode"], state="readonly",
                     width=44).grid(row=1, column=3, columnspan=2, sticky="w", **pad)
        self.go_btn = ttk.Button(f, text="Make subtitles", command=self.transcribe)
        self.go_btn.grid(row=0, column=6, rowspan=2, sticky="e", padx=(6, 0))
        self.model_lbl = ttk.Label(f, text="", foreground="gray", wraplength=900)
        self.model_lbl.grid(row=2, column=0, columnspan=7, sticky="w", **pad)

        # 3. Check and edit
        f = ttk.LabelFrame(main, text="3. Check and fix", padding=8)
        f.grid(row=2, column=0, sticky="nsew", **pad)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(f, columns=("start", "end", "text"), show="headings", selectmode="browse")
        for col, text, width, anchor, stretch in (("start", "Start", 80, "e", False), ("end", "End", 80, "e", False),
                                                  ("text", "Caption", 400, "w", True)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=stretch)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())
        self.tree.bind("<Double-1>", lambda e: self.play_line())
        self.list_lbl = ttk.Label(f, text="", foreground="gray")
        self.list_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        side = ttk.Frame(f)
        side.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(10, 0))
        self.preview = tk.Label(side, bg="#202020", width=self.PREVIEW_W, height=self.PREVIEW_H,
                                text="", fg="#bbbbbb")
        self.preview.grid(row=0, column=0, columnspan=4, sticky="n")
        self.preview.grid_propagate(False)
        ed = ttk.Frame(side)
        ed.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ed.columnconfigure(5, weight=1)
        ttk.Label(ed, text="From").grid(row=0, column=0, sticky="w")
        self.start_e = ttk.Entry(ed, width=11)
        self.start_e.grid(row=0, column=1, sticky="w", padx=(4, 10))
        ttk.Label(ed, text="to").grid(row=0, column=2, sticky="w")
        self.end_e = ttk.Entry(ed, width=11)
        self.end_e.grid(row=0, column=3, sticky="w", padx=4)
        for e in (self.start_e, self.end_e):
            e.bind("<Return>", lambda ev: self._apply_times())
            e.bind("<FocusOut>", lambda ev: self._apply_times())
        self.play_btn = ttk.Button(ed, text="▶ Play", width=7, command=self.play_line)
        self.play_btn.grid(row=0, column=5, sticky="e")
        self.text_w = tk.Text(ed, height=3, width=46, wrap="word", font=("", 11), undo=True,
                              relief="solid", borderwidth=1)
        self.text_w.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(6, 4))
        self.text_w.bind("<KeyRelease>", lambda e: self._apply_text())
        btns = ttk.Frame(ed)
        btns.grid(row=2, column=0, columnspan=6, sticky="ew")
        self.edit_btns = []
        for text, cmd in (("Split at cursor", self.split_line), ("Join with next", self.merge_next),
                          ("Delete", self.delete_line)):
            b = ttk.Button(btns, text=text, command=cmd)
            b.pack(side="left", padx=(0, 6))
            self.edit_btns.append(b)
        self.edit_hint = ttk.Label(ed, text="Click a caption to fix it. Double-click (or ▶) to hear it.",
                                   foreground="gray", wraplength=self.PREVIEW_W)
        self.edit_hint.grid(row=3, column=0, columnspan=6, sticky="w", pady=(6, 0))

        # 4. Save
        f = ttk.LabelFrame(main, text="4. Save", padding=8)
        f.grid(row=3, column=0, sticky="ew", **pad)
        f.columnconfigure(0, weight=1)
        outs = ttk.Frame(f)
        outs.grid(row=0, column=0, sticky="w", **pad)
        for key, text in (("srt", "Subtitle file (.srt)"), ("vtt", "Web subtitle file (.vtt)"),
                          ("track", "Video with subtitles you can switch on"),
                          ("burn", "Video with captions burned in")):
            self.__dict__[f"{key}_cb"] = ttk.Checkbutton(outs, text=text, variable=self.v[key])
            self.__dict__[f"{key}_cb"].pack(side="left", padx=(0, 18))
        look = self.look_frame = ttk.Frame(f)
        look.grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(look, text="Burned-in look:").pack(side="left", padx=(0, 8))
        for label, key, values, width in (("Font", "font", self.fonts, 12), ("Size", "size", list(SIZES), 10),
                                          ("Style", "look", LOOKS, 18), ("Position", "position", list(POSITIONS), 8)):
            ttk.Label(look, text=label).pack(side="left", padx=(6, 3))
            ttk.Combobox(look, values=values, textvariable=self.v[key], state="readonly", width=width).pack(side="left")
        ttk.Checkbutton(look, text="Bold", variable=self.v["bold"]).pack(side="left", padx=(10, 6))
        ttk.Label(look, text="Colour").pack(side="left", padx=(6, 3))
        self.swatch = tk.Label(look, width=3, relief="solid", borderwidth=1, bg=self.v["colour"].get(), cursor="hand2")
        self.swatch.pack(side="left")
        self.swatch.bind("<Button-1>", lambda e: self.pick_colour())
        where = ttk.Frame(f)
        where.grid(row=2, column=0, sticky="ew", **pad)
        where.columnconfigure(2, weight=1)
        ttk.Radiobutton(where, text="Next to the video", value="same", variable=self.v["where"]).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(where, text="In this folder:", value="folder", variable=self.v["where"]).grid(row=0, column=1, sticky="w", padx=(12, 4))
        self.dir_e = ttk.Entry(where, textvariable=self.v["out_dir"])
        self.dir_e.grid(row=0, column=2, sticky="ew")
        self.dir_btn = ttk.Button(where, text="Choose…", command=self.pick_folder)
        self.dir_btn.grid(row=0, column=3, padx=6)
        self.save_btn = ttk.Button(where, text="Save", command=self.save)
        self.save_btn.grid(row=0, column=4, padx=(6, 0))

        f = ttk.Frame(main)
        f.grid(row=4, column=0, sticky="ew", **pad)
        f.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(f, mode="determinate", maximum=1000)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.cancel_btn = ttk.Button(f, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.grid(row=0, column=1)
        foot = ttk.Frame(main)  # the status line, with the support link on its right
        foot.grid(row=5, column=0, sticky="ew", **pad)
        foot.columnconfigure(0, weight=1)
        self.status = ttk.Label(foot, text="Ready.", wraplength=850)
        self.status.grid(row=0, column=0, sticky="w")
        support_link(foot).grid(row=0, column=1, sticky="ne", padx=(12, 0))

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

    def _on_outputs(self):
        burn = self.v["burn"].get()
        for w in self.look_frame.winfo_children():
            try:
                w.config(state=("readonly" if isinstance(w, ttk.Combobox) else "normal") if burn else "disabled")
            except tk.TclError:
                pass
        self._request_preview()

    def pick_colour(self):
        if not self.v["burn"].get():
            return
        _, hexcode = colorchooser.askcolor(color=self.v["colour"].get(), title="Caption colour")
        if hexcode:
            self.v["colour"].set(hexcode)

    def pick_folder(self):
        d = filedialog.askdirectory(title="Choose output folder", initialdir=self.v["out_dir"].get())
        if d:
            self.v["out_dir"].set(d)

    def _style(self):
        return dict(font=self.v["font"].get(), size=self.v["size"].get(), bold=self.v["bold"].get(),
                    colour=self.v["colour"].get(), look=self.v["look"].get(), position=self.v["position"].get())

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
        self.video = dict(path=path, info=info)
        self._set_caps([], dirty=False)
        self.lang = None
        kind = f"{info['width']}×{info['height']} video" if info["width"] else "audio"
        self.file_lbl.config(text=f"{os.path.basename(path)}   ({kind}, {fmt_duration(info['duration'])})")
        for key in ("track", "burn"):
            self.__dict__[f"{key}_cb"].config(state="normal" if info["width"] else "disabled")
            if not info["width"]:
                self.v[key].set(False)
        self.status.config(text="Click Make subtitles, or Open subtitle file to use ones you already have.")
        self._update_buttons()

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

    # ---- the caption list -------------------------------------------------- #
    def _set_caps(self, caps, dirty=True, keep=None):
        self.caps = caps
        self.dirty = dirty
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

    def _row(self, c):
        return (fmt_short(c["start"]), fmt_short(c["end"]), c["text"].replace("\n", "  ⏎  "))

    def _update_list_label(self):
        if not self.caps:
            text = "The captions appear here as they're written." if self.video else ""
        else:
            text = f"{len(self.caps)} captions"
            if self.lang:
                text += f", {LANG_NAMES.get(self.lang, self.lang)}"
            text += ". " + ("Edited — not saved yet." if self.dirty else "")
        self.list_lbl.config(text=text)

    def _sel(self):
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def _on_select(self):
        self._load_editor(self._sel())

    def _load_editor(self, k):
        self.editing = k if k is not None and k < len(self.caps) else None
        self.text_w.config(state="normal")  # a disabled widget ignores changes
        for e in (self.start_e, self.end_e):
            e.config(state="normal")
        for e in (self.start_e, self.end_e):
            e.delete(0, "end")
        self.text_w.delete("1.0", "end")
        if k is not None and k < len(self.caps):
            c = self.caps[k]
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
        text = self.text_w.get("1.0", "end-1c").strip()
        if text and text != self.caps[k]["text"]:
            self.caps[k]["text"] = text
            self.tree.item(str(k), values=self._row(self.caps[k]))
            self.dirty = True
            self._update_list_label()
            self._request_preview()

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
            self.edit_hint.config(text="Times look like 00:01:02.500 (hours:minutes:seconds), and the end must "
                                       "be after the start.", foreground="red")
            return
        self.edit_hint.config(text="Click a caption to fix it. Double-click (or ▶) to hear it.", foreground="gray")
        if abs(start - c["start"]) < 0.0005 and abs(end - c["end"]) < 0.0005:
            return
        c["start"], c["end"] = start, end
        order = sorted(range(len(self.caps)), key=lambda i: self.caps[i]["start"])
        self._set_caps([self.caps[i] for i in order], keep=order.index(k))

    def split_line(self):
        k = self.editing
        if k is None or self.running:
            return
        c = self.caps[k]
        text = self.text_w.get("1.0", "end-1c")
        pos = len(self.text_w.get("1.0", "insert"))
        a, b = text[:pos].strip(), text[pos:].strip()
        if not a or not b:
            self.edit_hint.config(text="Put the text cursor where the caption should be split.", foreground="red")
            return
        mid = c["start"] + (c["end"] - c["start"]) * len(a) / max(1, len(a) + len(b))
        m = CAPTION_MODES[self.v["mode"].get()]
        first = dict(start=c["start"], end=mid, text=break_lines(a.replace("\n", " "), m["line"], m["lines"]))
        second = dict(start=mid, end=c["end"], text=break_lines(b.replace("\n", " "), m["line"], m["lines"]))
        self._set_caps(self.caps[:k] + [first, second] + self.caps[k + 1:], keep=k)

    def merge_next(self):
        k = self.editing
        if k is None or k + 1 >= len(self.caps) or self.running:
            return
        a, b = self.caps[k], self.caps[k + 1]
        m = CAPTION_MODES[self.v["mode"].get()]
        text = (a["text"] + " " + b["text"]).replace("\n", " ")
        merged = dict(start=a["start"], end=max(a["end"], b["end"]), text=break_lines(text, m["line"], 2))
        self._set_caps(self.caps[:k] + [merged] + self.caps[k + 2:], keep=k)

    def delete_line(self):
        k = self.editing
        if k is None or self.running:
            return
        self._set_caps(self.caps[:k] + self.caps[k + 1:], keep=k if k < len(self.caps) - 1 else k - 1)

    def play_line(self):
        k = self._sel()
        if k is None or not self.video:
            return
        c = self.caps[k]
        start = max(0.0, c["start"] - 0.15)
        dur = c["end"] - start + 0.25
        wav = os.path.join(self.tmp, f"line{int(time.time() * 1000) % 100000}.wav")

        def work():
            try:
                subprocess.run([self.ffmpeg, "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                                "-i", self.video["path"], "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", wav],
                               capture_output=True, timeout=60, **_POPEN_FLAGS)
                if os.path.exists(wav):
                    self.ui(lambda: self.player.play(wav))
            except Exception as e:
                log_error(f"Play: {type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _update_buttons(self):
        busy = self.running
        has_file = self.video is not None
        self.go_btn.config(state="normal" if has_file and not busy else "disabled")
        self.open_subs_btn.config(state="normal" if has_file and not busy else "disabled")
        self.choose_btn.config(state="disabled" if busy else "normal")
        self.save_btn.config(state="normal" if self.caps and not busy else "disabled")
        sel = self.editing is not None and not busy
        for b in self.edit_btns + [self.play_btn]:
            b.config(state="normal" if sel else "disabled")
        state = "normal" if sel else "disabled"
        self.text_w.config(state=state)
        for e in (self.start_e, self.end_e):
            e.config(state=state)

    # ---- preview ------------------------------------------------------------ #
    def _request_preview(self):
        """Show the selected caption on its frame, as it will look burned in."""
        if not self.video or not self.video["info"]["width"]:
            self.preview.config(image="", text="(audio file: no picture)" if self.video else "")
            return
        k = self._sel()
        if k is not None and k < len(self.caps):
            c = dict(self.caps[k])
            t = (c["start"] + c["end"]) / 2
        else:
            c, t = None, min(5.0, self.video["info"]["duration"] / 2)
        with self.preview_cv:
            self.preview_req = (self.video["path"], self.video["info"], t, c, self._style(), self.v["burn"].get())
            self.preview_cv.notify()

    def _preview_worker(self):
        while True:
            with self.preview_cv:
                while self.preview_req is None:
                    self.preview_cv.wait()
                req, self.preview_req = self.preview_req, None
            time.sleep(0.12)  # typing: wait for a pause
            with self.preview_cv:
                if self.preview_req is not None:
                    continue
            try:
                im = self._render_preview(*req)
                self.ui(lambda im=im: self._show_preview(im))
            except Exception as e:
                log_error(f"Preview: {type(e).__name__}: {e}")

    def _render_preview(self, path, info, t, cap, style, burn):
        import io
        w, h = info["width"], info["height"]
        scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        pw, ph = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
        vf = f"scale={pw}:{ph}"
        cwd = None
        if cap is not None:
            shown = dict(cap, start=0, end=36000)
            if burn:
                cwd = prepare_subs_dir([shown], w, h, style)
            else:  # not burning in: show it in a plain player-like style
                cwd = prepare_subs_dir([shown], w, h, dict(font=self.fonts[0], size="Medium", bold=False,
                                                           colour="#ffffff", look="Box behind the text",
                                                           position="Bottom"))
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

    def _show_preview(self, im):
        self.preview_photo = ImageTk.PhotoImage(im)
        self.preview.config(image=self.preview_photo, text="")

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

    def on_close(self):
        if self.running and not messagebox.askyesno("Still working", "Stop and close?"):
            return
        if not self.running and not self._confirm_discard():
            return
        self.cancel()
        self.player.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.root.destroy()


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
    Fast model, read a video's sound, run Whisper on it, burn in captions and add a
    subtitle track, and load drag and drop."""
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
                     position="Bottom")
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


def main():
    if "--self-test" in sys.argv:
        code = self_test()
        _say(f"finished with code {code}")
        quit_now(code)
    root = tk.Tk()
    try:  # nicer native look where available
        ttk.Style().theme_use("vista" if sys.platform == "win32" else "aqua" if sys.platform == "darwin" else "clam")
    except tk.TclError:
        pass
    root.geometry("1120x820")
    root.minsize(1000, 720)
    app = SubtitleApp(root)
    files = [a for a in sys.argv[1:] if os.path.isfile(a)]
    if files:
        root.after(300, lambda: app._on_drop(files))
    root.mainloop()
    quit_now(0)


if __name__ == "__main__":
    main()
