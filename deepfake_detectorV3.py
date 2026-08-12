
def _get_asset_b64(path):
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ''

def _get_audio_b64():
    try:
        with open('netflix_ta_dum.mp3', 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


# ── Standard Library ──────────────────────────────────────────────────
import base64
import csv as _csv_mod
import hashlib
import io
import json
import textwrap
import logging
import os
import re
import sqlite3
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Security Logger (console only — never writes paths/secrets to UI) ─
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_sec_logger = logging.getLogger("deepfake_lab.security")

# ── Third-party ───────────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch
import torchvision.models as tvm
import torchvision.transforms as T
import transformers
from PIL import ExifTags, Image
from torch import nn

pipeline = transformers.pipeline
AutoFeatureExtractor = transformers.AutoFeatureExtractor
AutoModelForImageClassification = transformers.AutoModelForImageClassification
AutoConfig = transformers.AutoConfig
try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# ── Optional: OpenCV for face detection ───────────────────────────────
try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── Optional: librosa for audio waveform ──────────────────────────────
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════
# HISTORY DATABASE  (SQLite — no extra dependencies)
# ═══════════════════════════════════════════════════════════════════════
_DB_DIR = Path.home() / ".deepfake_lab"
_DB_PATH = _DB_DIR / "history.db"
_MAX_HISTORY = 200


def _db_conn():
    _DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db_conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            filename     TEXT    DEFAULT 'unknown',
            thumbnail    TEXT,
            model_name   TEXT,
            model_id     TEXT,
            mode         TEXT    DEFAULT 'single',
            verdict      TEXT,
            confidence   REAL,
            elapsed      REAL,
            results_json TEXT
        )""")
        c.commit()


def save_history(
    filename,
    pil_img,
    model_name,
    model_id,
    mode,
    verdict,
    confidence,
    elapsed,
    all_results,
):
    """Save one inference record; prune to _MAX_HISTORY rows."""
    thumb = pil_img.copy().convert("RGB")
    thumb.thumbnail((100, 100))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=75)
    thumb_b64 = base64.b64encode(buf.getvalue()).decode()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _db_conn() as c:
        c.execute(
            """
            INSERT INTO history
              (timestamp,filename,thumbnail,model_name,model_id,
               mode,verdict,confidence,elapsed,results_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                ts,
                filename,
                thumb_b64,
                model_name,
                model_id,
                mode,
                verdict,
                round(float(confidence), 2),
                round(float(elapsed), 3),
                json.dumps(all_results),
            ),
        )
        c.execute(
            """
            DELETE FROM history WHERE id NOT IN
              (SELECT id FROM history ORDER BY id DESC LIMIT ?)""",
            (_MAX_HISTORY,),
        )
        c.commit()


def get_history():
    with _db_conn() as c:
        rows = c.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def clear_history():
    with _db_conn() as c:
        c.execute("DELETE FROM history")
        c.commit()


def _export_csv(rows):
    buf = io.StringIO()
    fields = [
        "id",
        "timestamp",
        "filename",
        "model_name",
        "model_id",
        "mode",
        "verdict",
        "confidence",
        "elapsed",
    ]
    w = _csv_mod.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


_init_db()

# ═══════════════════════════════════════════════════════════════════════
# SECURITY UTILITIES
# ═══════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100 MB
_MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_IMAGE_DIM = 8000  # px per side
_RATE_LIMIT_MAX = 10  # max analyses per window
_RATE_LIMIT_WINDOW = 60  # seconds

# ── Magic bytes for file-type verification ────────────────────────────
_MAGIC = {
    "jpeg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"],  # RIFF....WEBP
    "mp4": [b"\x00\x00\x00", b"ftyp"],  # ftyp at offset 4
    "avi": [b"RIFF"],  # RIFF....AVI
    "mov": [b"\x00\x00\x00", b"ftyp"],
    "wav": [b"RIFF"],
    "mp3": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
}

# ── MIME whitelist ────────────────────────────────────────────────────
_ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}
_ALLOWED_VIDEO_MIMES = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/avi",
    "application/octet-stream",
}
_ALLOWED_AUDIO_MIMES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "application/octet-stream",
}


def _sanitize_filename(name: str) -> str:
    """
    Strip path-traversal sequences, control chars, and limit length.
    Returns a safe display string — never used as an actual file path.
    """
    if not name:
        return "unknown"
    # Remove directory components
    name = Path(name).name
    # Strip path traversal
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    # Remove control characters and HTML-sensitive chars
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    # Limit length
    return name[:100] or "unknown"


def _safe_error(exc: Exception, context: str = "operation") -> str:
    """
    Log full exception details to console, return a GENERIC user message.
    Never exposes file paths, model internals, or stack traces to the UI.
    """
    _sec_logger.error(
        "Security-filtered error during %s: %s: %s",
        context,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    return (
        f"**An error occurred during {context}.**\n\n"
        "Please check your internet connection and try again. "
        "If the problem persists, try a different model or file."
    )


def validate_image_upload(uploaded_file) -> tuple:
    """
    Validate an uploaded image file.
    Returns (pil_image, file_bytes, error_message).
    On success error_message is None; on failure pil_image/file_bytes are None.
    """
    if uploaded_file is None:
        return None, None, "No file uploaded."

    # 1. MIME type check
    mime = getattr(uploaded_file, "type", "") or ""
    if mime and mime not in _ALLOWED_IMAGE_MIMES:
        _sec_logger.warning("Rejected image upload with MIME: %s", mime)
        return (
            None,
            None,
            f"Unsupported file type: `{_sanitize_filename(mime)}`. Use JPG, PNG, or WEBP.",
        )

    # 2. File size check
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    if len(file_bytes) > _MAX_IMAGE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return (
            None,
            None,
            f"File too large ({size_mb:.1f} MB). Maximum is {_MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )

    # 3. Magic bytes check
    header = file_bytes[:16]
    ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        ext = "jpeg"
    valid_magic = False
    for magic_ext in (ext,):
        for magic in _MAGIC.get(magic_ext, []):
            if header.startswith(magic) or magic in header[:12]:
                valid_magic = True
                break
    if not valid_magic and ext in _MAGIC:
        _sec_logger.warning(
            "Magic bytes mismatch for %s (ext=%s)",
            _sanitize_filename(uploaded_file.name),
            ext,
        )
        return (
            None,
            None,
            "File content does not match its extension. The file may be corrupt or misnamed.",
        )

    # 4. PIL openability + dimension check
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.load()  # Force full decode to catch truncated/corrupt files
    except Exception as e:
        _sec_logger.warning("PIL failed to open uploaded image: %s", e)
        return (
            None,
            None,
            "Could not open this image. The file may be corrupt or in an unsupported format.",
        )

    w, h = pil_img.size
    if w > _MAX_IMAGE_DIM or h > _MAX_IMAGE_DIM:
        return (
            None,
            None,
            (
                f"Image dimensions ({w}×{h}) exceed the maximum ({_MAX_IMAGE_DIM}×{_MAX_IMAGE_DIM}). "
                "Please resize and re-upload."
            ),
        )

    return pil_img, file_bytes, None


def extract_video_frames(video_bytes: bytes, num_frames: int = 6) -> list:
    """Extract `num_frames` evenly spaced frames from video bytes using OpenCV."""
    if not CV2_AVAILABLE:
        return []

    import os
    import tempfile

    frames = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        indices = [min(idx, total_frames - 1) for idx in indices]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pct = (idx / max(1, total_frames - 1)) * 100
                frames.append((pct, pil_img))
        cap.release()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return frames


def validate_video_upload(uploaded_file) -> tuple:
    """
    Validate an uploaded video file.
    Returns (video_bytes, error_message).
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    # 1. MIME type check
    mime = getattr(uploaded_file, "type", "") or ""
    if mime and mime not in _ALLOWED_VIDEO_MIMES:
        _sec_logger.warning("Rejected video upload with MIME: %s", mime)
        return None, "Unsupported file type. Use MP4, AVI, or MOV."

    # 2. File size check
    video_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    if len(video_bytes) > _MAX_VIDEO_BYTES:
        size_mb = len(video_bytes) / (1024 * 1024)
        return (
            None,
            f"File too large ({size_mb:.1f} MB). Maximum is {_MAX_VIDEO_BYTES // (1024 * 1024)} MB.",
        )

    # 3. Magic bytes — verify video header
    header = video_bytes[:16]
    if not (header[:4] == b"RIFF" or b"ftyp" in header[:12]):
        _sec_logger.warning(
            "Video magic bytes mismatch for %s", _sanitize_filename(uploaded_file.name)
        )
        return (
            None,
            "File content does not match a valid video format. The file may be corrupt.",
        )

    # 4. OpenCV openability check
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0:
            cap.release()
            return (
                None,
                "Could not read this video. The file may be corrupt or in an unsupported codec.",
            )
        cap.release()
    except Exception as e:
        _sec_logger.warning("Video validation failed: %s", e)
        return None, "Could not validate this video file."
    finally:
        try:
            Path(tmp.name).unlink()
        except Exception:
            pass

    return video_bytes, None


def validate_audio_upload(uploaded_file) -> tuple:
    """
    Validate an uploaded audio file.
    Returns (audio_bytes, error_message).
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    # 1. MIME type check
    mime = getattr(uploaded_file, "type", "") or ""
    if mime and mime not in _ALLOWED_AUDIO_MIMES:
        _sec_logger.warning("Rejected audio upload with MIME: %s", mime)
        return None, "Unsupported file type. Use WAV or MP3."

    # 2. File size check
    audio_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        size_mb = len(audio_bytes) / (1024 * 1024)
        return (
            None,
            f"File too large ({size_mb:.1f} MB). Maximum is {_MAX_AUDIO_BYTES // (1024 * 1024)} MB.",
        )

    # 3. Magic bytes
    header = audio_bytes[:16]
    valid = (
        header[:4] == b"RIFF"  # WAV
        or header[:3] == b"ID3"  # MP3 with ID3 tag
        or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")  # MP3 sync
    )
    if not valid:
        _sec_logger.warning(
            "Audio magic bytes mismatch for %s", _sanitize_filename(uploaded_file.name)
        )
        return (
            None,
            "File content does not match a valid audio format. The file may be corrupt.",
        )

    return audio_bytes, None


def check_rate_limit() -> tuple:
    """
    Session-based sliding window rate limiter.
    Returns (allowed: bool, wait_seconds: float).
    """
    now = time.time()

    if "_rate_timestamps" not in st.session_state:
        st.session_state["_rate_timestamps"] = []

    # Prune old timestamps outside window
    ts_list = [
        t for t in st.session_state["_rate_timestamps"] if now - t < _RATE_LIMIT_WINDOW
    ]
    st.session_state["_rate_timestamps"] = ts_list

    if len(ts_list) >= _RATE_LIMIT_MAX:
        oldest = min(ts_list)
        wait = _RATE_LIMIT_WINDOW - (now - oldest)
        return False, max(0, wait)

    # Record this request
    ts_list.append(now)
    st.session_state["_rate_timestamps"] = ts_list
    return True, 0.0


# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MULTIMEDIA AUTHENTICITY LAB",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# SPLASH SCREEN / OPENING SEQUENCE
# ═══════════════════════════════════════════════════════════════════════
st.markdown(r"""
<style>
/* ----- PHASE 5 MODAL CSS ----- */
#netflix-modal-backdrop {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.8); z-index: 99999999;
    display: none; justify-content: center; align-items: center;
    opacity: 0; transition: opacity 0.25s ease-out;
}
#netflix-modal {
    width: 90%; max-width: 850px; max-height: 90vh;
    background: #181818; border-radius: 8px;
    overflow-y: auto; overflow-x: hidden;
    position: relative;
    transform: scale(0.9); transition: transform 0.25s ease-out;
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
}
.modal-close-btn {
    position: absolute; top: 20px; right: 20px;
    width: 36px; height: 36px; border-radius: 50%;
    background: #181818; color: #fff; border: 2px solid #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: bold; cursor: pointer; z-index: 10;
    transition: background 0.2s, color 0.2s;
}
.modal-close-btn:hover { background: #fff; color: #181818; }
.modal-banner {
    width: 100%; height: 400px;
    background-image: url('https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=1200&auto=format&fit=crop');
    background-size: cover; background-position: center;
    position: relative;
}
.modal-banner-gradient {
    position: absolute; bottom: 0; left: 0; width: 100%; height: 50%;
    background: linear-gradient(to top, #181818 0%, transparent 100%);
}
.modal-content-wrapper { padding: 0 40px 40px 40px; margin-top: -60px; position: relative; z-index: 2; }
.modal-title {
    font-family: 'Bebas Neue', cursive; font-size: 3.5rem; color: #fff;
    text-shadow: 0 2px 8px rgba(0,0,0,0.8); margin-bottom: 16px; line-height: 1;
}
.modal-actions { display: flex; gap: 10px; margin-bottom: 24px; align-items: center; }
.modal-play-btn {
    background: #fff; color: #000; border: none; border-radius: 4px;
    padding: 10px 24px; font-size: 1.2rem; font-weight: 700; cursor: pointer;
    display: flex; align-items: center; gap: 8px; transition: background 0.2s;
}
.modal-play-btn:hover { background: rgba(255,255,255,0.8); }
.modal-circle-btn {
    width: 40px; height: 40px; border-radius: 50%;
    background: rgba(42,42,42,0.6); color: #fff; border: 2px solid rgba(255,255,255,0.5);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; cursor: pointer; transition: border-color 0.2s, background 0.2s;
}
.modal-circle-btn:hover { border-color: #fff; background: rgba(255,255,255,0.1); }

.modal-body-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 40px; }
.modal-meta-row { display: flex; gap: 12px; margin-bottom: 12px; font-weight: 700; align-items: center; font-size: 15px; color: #fff; }
.modal-match { color: #46d369; }
.modal-year { color: #e5e5e5; font-weight: 400; }
.modal-hd { border: 1px solid rgba(255,255,255,0.4); padding: 0 4px; font-size: 12px; border-radius: 2px; color: #e5e5e5; }
.modal-desc { color: #fff; font-size: 16px; line-height: 1.5; margin-bottom: 20px; }

.modal-right-col { color: #b3b3b3; font-size: 14px; line-height: 1.6; }
.modal-right-col span { color: #777; }
.modal-right-col strong { color: #fff; font-weight: normal; }

/* Custom scrollbar for modal */
#netflix-modal::-webkit-scrollbar { width: 8px; }
#netflix-modal::-webkit-scrollbar-track { background: transparent; }
#netflix-modal::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
#netflix-modal::-webkit-scrollbar-thumb:hover { background: #555; }

</style>
""", unsafe_allow_html=True)


if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown('''
<style>
/* Hiding removed to fix sidebar bug */
.stApp { background-color: #000 !important; }

.splash-container {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: black;
    z-index: 999998;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.splash-logo {
    color: #e50914;
    font-size: 4rem;
    font-weight: bold;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 4px;
}
.splash-prompt {
    color: white;
    margin-top: 2rem;
    font-size: 1.5rem;
    animation: pulse 1.5s infinite;
    font-family: 'Inter', sans-serif;
}
@keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }

/* Invisible Full Screen Button */
div.stButton > button {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 999999 !important;
    opacity: 0 !important;
    cursor: pointer !important;
}
</style>
<div class="splash-container">
    <div class="splash-logo">MULTIMEDIA AUTHENTICITY LAB</div>
    <div class="splash-prompt">CLICK ANYWHERE TO ENTER</div>
</div>
''', unsafe_allow_html=True)
    
    if st.button("invisible_enter_button"):
        st.session_state.entered = True
        st.rerun()
else:
    audio_b64 = _get_audio_b64()
    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>' if audio_b64 else ""
    
    st.markdown(audio_html + '''
<div id="full-ui-container">
<style>
.zoom-container{position:fixed;top:0;left:0;width:100vw;height:100vh;background:black;z-index:999998;display:flex;flex-direction:column;align-items:center;justify-content:center;animation:fadeOutSplash 0.5s forwards 2.8s;pointer-events:none;}
.zoom-logo{color:#e50914;font-size:4rem;font-weight:bold;font-family:'Bebas Neue',sans-serif;letter-spacing:4px;animation:netflixZoom 3s forwards cubic-bezier(0.2,0.8,0.2,1);}
@keyframes netflixZoom{0%{transform:scale(1);opacity:0;}10%{transform:scale(0.9);opacity:1;}100%{transform:scale(5);opacity:0;}}
@keyframes fadeOutSplash{to{opacity:0;visibility:hidden;display:none;}}
#css-netflix-nav{position:fixed;top:0;left:0;width:100%;height:70px;background:linear-gradient(to bottom,rgba(0,0,0,0.9) 0%,rgba(0,0,0,0) 100%);display:flex;align-items:center;justify-content:space-between;padding:0 4%;z-index:1000;pointer-events:none;}
.nav-left{display:flex;align-items:center;gap:40px;}
.nav-logo{color:#e50914;font-size:1.8rem;font-weight:bold;font-family:'Bebas Neue',sans-serif;cursor:pointer;pointer-events:auto;}
.nav-logo .red-letter{color:#e50914;}
.nav-right{display:flex;align-items:center;gap:20px;pointer-events:auto;}
.nav-icon{width:24px;height:24px;fill:#fff;cursor:pointer;}
.profile-avatar{width:32px;height:32px;background:#e50914;border-radius:4px;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;cursor:pointer;}
</style>
<div class="zoom-container"><div class="zoom-logo">MULTIMEDIA AUTHENTICITY LAB</div></div>
<div id="css-netflix-nav">
<div class="nav-left"><div class="nav-logo"><span class="red-letter">D</span>EEPFAKE</div></div>
<div class="nav-right">
<svg class="nav-icon" viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
<svg class="nav-icon" viewBox="0 0 24 24"><path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2zm6-6v-5c0-3.07-1.63-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.64 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2zm-2 1H8v-6c0-2.48 1.51-4.5 4-4.5s4 2.02 4 4.5v6z"/></svg>
<div class="profile-avatar">S</div>
</div>
</div>
</div>
''', unsafe_allow_html=True)




# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Reset & Typography ── */
html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
    background-color: #000000 !important;
    color: #e5e5e5;
}


h1, h2, h3, h4, h5, h6 {
    font-family: 'Bebas Neue', cursive !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
}

/* ── App Background & Top Gradient ── */
.stApp {
    background: #000000;
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; height: 150px;
    background: linear-gradient(to bottom, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0) 100%);
    z-index: 100;
    pointer-events: none;
}


/* Ensure file uploaders are clickable and above any rogue overlays */
div[data-testid="stFileUploader"] {
    position: relative !important;
    z-index: 999999 !important;
    pointer-events: auto !important;
}
div[data-testid="stFileUploader"] * {
    pointer-events: auto !important;
}
/* Ensure entire stApp is clickable except where explicitly none */
.stApp {
    pointer-events: auto !important;
}

/* ── Top Navigation (Tabs) ── */
/* Streamlit tabs structure */
div[data-testid="stTabs"] {
    position: relative;
    z-index: 101;
}
/* Tab list wrapper */
div[data-baseweb="tab-list"] {
    background: #000000 !important;
    padding: 0 2rem;
    display: flex;
    gap: 1.5rem;
    border-bottom: 1px solid #1a1a1a !important;
}
/* Individual tab */
button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #b3b3b3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 1rem 0 !important;
    position: relative;
    overflow: hidden;
    transition: color 0.2s;
}
button[data-baseweb="tab"]:hover {
    color: #ffffff !important;
}
/* Active tab text */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
}
/* Tab active underline slide-in */
button[data-baseweb="tab"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 100%; height: 3px;
    background: #E50914;
    transform: translateX(-100%);
    transition: transform 0.3s ease;
}
button[data-baseweb="tab"][aria-selected="true"]::after {
    transform: translateX(0);
}
button[data-baseweb="tab"]:hover::after {
    transform: translateX(0);
}
/* Hide default underline */
div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ── Custom Logo Mark ── */
/* We inject this using a pseudo-element on the main title or nav container */
div[data-baseweb="tab-list"]::before {
    content: 'MULTIMEDIA AUTHENTICITY LAB';
    font-family: 'Bebas Neue', cursive;
    font-size: 2rem;
    color: #E50914;
    letter-spacing: 3px;
    display: flex;
    align-items: center;
    margin-right: 3rem;
}

/* ── Netflix Flat Cards ── */
.card {
    background: #000000;
    border: 1px solid #1f1f1f;
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover {
    border-color: #E50914;
    box-shadow: 0 0 15px rgba(229, 9, 20, 0.2);
}

/* ── Result Badges (Netflix Style) ── */
.badge-real {
    display: inline-block;
    background: #000000;
    color: #2ECC71;
    padding: 0.55rem 2.2rem;
    border-radius: 4px;
    font-family: 'Bebas Neue', cursive;
    font-size: 2.5rem;
    letter-spacing: 3px;
    border: 2px solid #2ECC71;
    /* Static green glow (will be refined in Phase 4) */
    box-shadow: 0 0 20px rgba(46,204,113,0.2); 
}
.badge-fake {
    display: inline-block;
    background: #000000;
    color: #E50914;
    padding: 0.55rem 2.2rem;
    border-radius: 4px;
    font-family: 'Bebas Neue', cursive;
    font-size: 2.5rem;
    letter-spacing: 3px;
    border: 2px solid #E50914;
    /* Red pulsing glow (will be refined in Phase 4) */
    animation: pulseFakeNetflix 2s infinite;
}
@keyframes pulseFakeNetflix {
    0%, 100% { box-shadow: 0 0 10px rgba(229,9,20,0.3); }
    50%      { box-shadow: 0 0 30px rgba(229,9,20,0.8); }
}

/* ── Main Titles ── */
.main-title {
    font-family: 'Bebas Neue', cursive !important;
    font-size: 4rem;
    color: #ffffff;
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 0.3rem;
    text-shadow: 0 4px 20px rgba(0,0,0,0.8);
}
.sub-title {
    text-align: center;
    color: #a3a3a3;
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 2rem;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.5rem;
    letter-spacing: 2px;
    color: #ffffff;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    display: none; /* Removed gradient line */
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #000000 !important;
    border-right: 1px solid #1a1a1a !important;
}

/* ── Basic structural overrides for other components ── */
p, li, label { color: #e5e5e5 !important; }
hr { border-color: #1a1a1a !important; }

/* ── Confidence table row ── */
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #1a1a1a;
    font-size: 0.95rem;
    color: #a3a3a3;
}
.conf-row:last-child { border-bottom: none; }
.conf-val {
    color: #ffffff;
    font-weight: 600;
}

/* ── Phase 3: Hero Banner ── */
.hero-container {
    position: relative;
    width: calc(100vw - 15px); /* prevent scrollbar shift */
    height: 40vh;
    margin-left: -4rem; /* offset streamlit horizontal padding */
    margin-top: 1rem;  /* offset streamlit vertical padding */
    margin-bottom: 3rem;
    overflow: hidden;
}
.hero-bg {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background-size: cover;
    background-position: center;
    animation: kenburns 20s infinite alternate ease-in-out;
    z-index: 1;
}
@keyframes kenburns {
    0% { transform: scale(1.0); }
    100% { transform: scale(1.08); }
}
.hero-gradient-bottom {
    position: absolute; bottom: 0; left: 0; width: 100%; height: 60%;
    background: linear-gradient(180deg, transparent 60%, #141414 100%);
    z-index: 2;
}
.hero-gradient-left {
    position: absolute; top: 0; left: 0; width: 60%; height: 100%;
    background: linear-gradient(90deg, rgba(20,20,20,0.8) 0%, transparent 50%);
    z-index: 3;
}
.hero-gradient-top {
    position: absolute; top: 0; left: 0; width: 100%; height: 15%;
    background: linear-gradient(180deg, rgba(0,0,0,0.7) 10%, transparent 100%);
    z-index: 4;
}
.hero-content {
    position: absolute;
    bottom: 10%; left: 4%;
    z-index: 5;
    max-width: 550px;
}
.hero-badge-pill {
    display: inline-block;
    background: #E50914; color: #000;
    font-size: 12px; font-weight: 700; font-family: 'Inter', sans-serif;
    padding: 2px 6px; border-radius: 2px;
    margin-bottom: 8px; letter-spacing: 1px;
}
.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 4.5rem;
    color: #fff;
    line-height: 1.1;
    text-shadow: 0 2px 8px rgba(0,0,0,0.8);
    margin-bottom: 8px;
}
.hero-metadata {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 12px;
    font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 500;
}
.meta-match { color: #46d369; font-weight: 700; }
.meta-tag { color: #e5e5e5; border: 1px solid #666; padding: 1px 4px; border-radius: 2px; font-size: 12px; }
.meta-hd { color: #e5e5e5; border: 1px solid #666; padding: 1px 4px; border-radius: 2px; font-size: 12px; }
.hero-desc {
    color: #b3b3b3; font-size: 16px; font-family: 'Inter', sans-serif;
    line-height: 1.4; text-shadow: 0 1px 2px rgba(0,0,0,0.8);
    margin-bottom: 20px;
}
.hero-buttons {
    display: flex; gap: 12px;
}
.hero-btn-primary {
    background: #fff; color: #000;
    font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 700;
    padding: 10px 24px; border-radius: 4px; border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: opacity 0.2s;
}
.hero-btn-primary:hover { opacity: 0.9; }
.hero-btn-secondary {
    background: rgba(109,109,110,0.7); color: #fff;
    font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 700;
    padding: 10px 24px; border-radius: 4px; border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    backdrop-filter: blur(5px);
    transition: background 0.2s;
}
.hero-btn-secondary:hover { background: rgba(109,109,110,0.4); }

/* ── Phase 6: Verdict / Result Reveal ── */
.netflix-result-card {
    background: #141414; border-radius: 4px; padding: 24px;
    position: relative; overflow: hidden; margin-bottom: 20px;
    border: 1px solid #222;
}
.glow-calm { box-shadow: 0 0 40px rgba(70, 211, 105, 0.1); border-color: rgba(70,211,105,0.3); }
.glow-danger { box-shadow: 0 0 40px rgba(229, 9, 20, 0.15); border-color: rgba(229,9,20,0.4); }

.result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.result-model-name { color: #fff; font-weight: 700; font-size: 16px; letter-spacing: 1px; text-transform: uppercase; }
.result-accuracy { color: #888; font-size: 12px; }

.result-verdict-container {
    height: 140px; display: flex; justify-content: center; align-items: center;
    position: relative; background: #000; border-radius: 4px; overflow: hidden; margin-bottom: 24px;
    border: 1px solid #1a1a1a;
}

/* Scanning Line Animation */
.scanning-line {
    position: absolute; top: 0; left: 0; width: 100%; height: 2px;
    background: rgba(255,255,255,0.5); box-shadow: 0 0 10px rgba(255,255,255,0.8);
    animation: scan 2s linear infinite; opacity: 0; z-index: 10;
}
@keyframes scan {
    0% { top: -10%; opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { top: 110%; opacity: 0; }
}

/* Reveal Animation */
.verdict-text {
    font-family: 'Bebas Neue', cursive, sans-serif; font-size: 5rem; letter-spacing: 4px;
    animation: revealVerdict 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    transform: scale(0.5); opacity: 0; z-index: 5; margin: 0; line-height: 1;
}
@keyframes revealVerdict {
    0% { transform: scale(0.5); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

/* Glitch Effect for FAKE */
.glitch-text { position: relative; }
.glitch-text::before, .glitch-text::after {
    content: attr(data-text); position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0.8;
}
.glitch-text::before {
    left: 3px; text-shadow: -2px 0 red; background: transparent; overflow: hidden;
    animation: noise-anim-2 3s infinite linear alternate-reverse; z-index: -1;
}
.glitch-text::after {
    left: -3px; text-shadow: -2px 0 blue; background: transparent; overflow: hidden;
    animation: noise-anim 2s infinite linear alternate-reverse; z-index: -2;
}
@keyframes noise-anim {
    0% { clip-path: inset(10% 0 80% 0); } 20% { clip-path: inset(80% 0 10% 0); } 40% { clip-path: inset(30% 0 40% 0); } 60% { clip-path: inset(60% 0 20% 0); } 80% { clip-path: inset(10% 0 50% 0); } 100% { clip-path: inset(50% 0 30% 0); }
}
@keyframes noise-anim-2 {
    0% { clip-path: inset(20% 0 60% 0); } 20% { clip-path: inset(50% 0 30% 0); } 40% { clip-path: inset(10% 0 80% 0); } 60% { clip-path: inset(80% 0 10% 0); } 80% { clip-path: inset(30% 0 40% 0); } 100% { clip-path: inset(60% 0 20% 0); }
}

/* Sleek Progress Bar Custom */
.sleek-progress-bg { width: 100%; height: 3px; background: #333; border-radius: 2px; margin-top: 10px; overflow: hidden; }
.sleek-progress-fill { height: 100%; border-radius: 2px; box-shadow: 0 0 10px currentColor; transition: width 1.5s cubic-bezier(0.2, 0.8, 0.2, 1); width: 0; }
.result-stats { margin-bottom: 20px; }
.stat-row { display: flex; justify-content: space-between; font-size: 14px; color: #ccc; font-weight: 500; }

.result-details { border-top: 1px solid #222; padding-top: 15px; }
.detail-row { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 10px; color: #a3a3a3; }
.raw-label { color: #555; font-size: 11px; margin-left: 6px; }

/* Global Override for Streamlit Native Progress (e.g. during inference loading) */
.stProgress > div > div > div > div {
    background-color: #E50914 !important; height: 3px !important; box-shadow: 0 0 8px #E50914; border-radius: 2px !important;
}
.stProgress > div > div { height: 3px !important; background-color: #333 !important; border-radius: 2px !important; }


/* ── Verdict Labels Hierarchy ── */
.result-label {
    font-family: 'Bebas Neue', cursive, sans-serif;
    font-size: 3.5rem;
    letter-spacing: 2px;
    margin: 10px 0;
    line-height: 1;
}
.result-label.real-text { color: #46d369; text-shadow: 0 0 10px rgba(70,211,105,0.3); }
.result-label.fake-text { color: #E50914; text-shadow: 0 0 10px rgba(229,9,20,0.3); }

.result-conf {
    font-size: 1.2rem;
    color: #a3a3a3;
    font-weight: 500;
}

.ensemble-label {
    font-family: 'Bebas Neue', cursive, sans-serif;
    font-size: 6rem;
    letter-spacing: 4px;
    margin: 15px 0;
    line-height: 1;
}
.ensemble-label.real-text { color: #46d369; text-shadow: 0 0 20px rgba(70,211,105,0.4); }
.ensemble-label.fake-text { color: #E50914; text-shadow: 0 0 20px rgba(229,9,20,0.4); }

.ensemble-conf {
    font-size: 1.8rem;
    color: #cccccc;
    font-weight: 600;
}

/* ── Phase 7: Color & Type System (Netflix Overrides) ── */
/* Base Font and Background */
.stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #141414 !important;
    color: #e5e5e5 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #000000 !important;
    border-right: 1px solid #1a1a1a !important;
}

/* Buttons (Netflix Style) */
button[data-testid="baseButton-secondary"] {
    background-color: rgba(109, 109, 110, 0.7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    backdrop-filter: blur(5px) !important;
    transition: background-color 0.2s !important;
}
button[data-testid="baseButton-secondary"]:hover {
    background-color: rgba(109, 109, 110, 0.4) !important;
}

button[data-testid="baseButton-primary"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s !important;
}
button[data-testid="baseButton-primary"]:hover {
    opacity: 0.8 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: transparent !important;
    border: 1px solid #333 !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stExpander"] summary:hover {
    color: #E50914 !important;
}
[data-testid="stExpanderDetails"] {
    background-color: rgba(0,0,0,0.5) !important;
    border-top: 1px solid #333 !important;
}

/* File Uploader */
[data-testid="stFileUploadDropzone"] {
    background-color: #1a1a1a !important;
    border: 2px dashed #444 !important;
    border-radius: 4px !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #E50914 !important;
}
[data-testid="stFileUploadDropzone"] section {
    color: #e5e5e5 !important;
}
[data-testid="stFileUploadDropzone"] button {
    background-color: #E50914 !important;
    color: #fff !important;
    font-weight: 700 !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #fff !important;
    border-radius: 4px !important;
}
[data-baseweb="popover"] {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
}
[data-baseweb="menu"] {
    background-color: #1a1a1a !important;
}
[data-baseweb="menu"] li {
    color: #fff !important;
}
[data-baseweb="menu"] li:hover {
    background-color: #333 !important;
}

/* Inputs */
input {
    background-color: #1a1a1a !important;
    color: #fff !important;
    border: 1px solid #333 !important;
    border-radius: 4px !important;
}
input:focus {
    border-color: #E50914 !important;
}

/* Typography Headers */
h1, h2, h3 {
    font-family: 'Bebas Neue', cursive, sans-serif !important;
    color: #ffffff !important;
    letter-spacing: 1px !important;
}

/* ── Phase 8: Scrollbar & Polish ── */
::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
::-webkit-scrollbar-track {
    background: transparent !important;
}
::-webkit-scrollbar-thumb {
    background: #333 !important;
    border-radius: 3px !important;
}
::-webkit-scrollbar-thumb:hover {
    background: #555 !important;
}

/* Global Vignette */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    box-shadow: inset 0 0 150px rgba(0,0,0,0.85);
    pointer-events: none;
    z-index: 9999;
}

/* ── Phase 1: Micro-Interactions & Hover States ── */
.stButton > button {
    transition: all 0.3s ease !important;
    border-radius: 6px !important;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4) !important;
    border-color: #E50914 !important;
    color: #ffffff !important;
}

.hist-card {
    background: rgba(20, 20, 20, 0.6);
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1rem;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}
.hist-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6), 0 0 15px rgba(229, 9, 20, 0.15);
    border-color: rgba(229, 9, 20, 0.5);
    background: rgba(30, 30, 30, 0.8);
}

div[data-testid="stFileUploader"] > section {
    transition: all 0.3s ease;
    border-radius: 8px !important;
    background-color: rgba(20,20,20,0.5) !important;
}
div[data-testid="stFileUploader"] > section:hover {
    background-color: rgba(30,30,30,0.8) !important;
    border-color: #E50914 !important;
    box-shadow: 0 0 15px rgba(229, 9, 20, 0.15) !important;
}

/* ── Phase 1: Custom Loading Animations ── */
@keyframes neonPulse {
    0% { border-color: rgba(229,9,20, 0.1); box-shadow: 0 0 5px rgba(229,9,20,0.1); }
    50% { border-color: rgba(229,9,20, 0.8); box-shadow: 0 0 15px rgba(229,9,20,0.6); }
    100% { border-color: rgba(229,9,20, 0.1); box-shadow: 0 0 5px rgba(229,9,20,0.1); }
}

div[data-testid="stSpinner"] {
    animation: neonPulse 2s infinite;
    border-radius: 8px;
    padding: 1rem;
    background: rgba(20,20,20,0.8);
    border: 1px solid transparent;
}
div[data-testid="stSpinner"] > div > div {
    border-top-color: #E50914 !important;
    border-right-color: transparent !important;
    border-bottom-color: transparent !important;
    border-left-color: transparent !important;
}


/* ── Phase 2: Mobile Responsiveness ── */
@media (max-width: 768px) {
    .badge-real, .badge-fake {
        font-size: 1.5rem !important;
        padding: 0.3rem 1rem !important;
    }
    div[data-baseweb="tab-list"] {
        padding: 0 0.5rem !important;
        gap: 0.5rem !important;
        overflow-x: auto;
    }
    div[data-baseweb="tab-list"]::before {
        font-size: 1.2rem !important;
        margin-right: 1rem !important;
    }
    .hist-card {
        padding: 0.5rem !important;
    }
    .card {
        padding: 0.8rem !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# Each entry: (display_name, hf_model_id, label_map_fn, notes)
# label_map_fn maps the raw HF label → ("REAL" or "FAKE")
# ═══════════════════════════════════════════════════════════════════════

# ── Label normaliser helpers ──────────────────────────────────────────
# Each function converts the raw HF label string → "REAL" or "FAKE".
# Models differ in what they call their classes, so we keep one fn each.


def _generic_map(raw: str) -> str:
    """
    Universal fallback: checks for 'real'/'fake' substrings,
    then LABEL_0 (real) / LABEL_1 (fake) convention.
    """
    r = raw.lower()
    if "real" in r:
        return "REAL"
    if "fake" in r or "deepfake" in r or "generated" in r or "ai" in r:
        return "FAKE"
    # LABEL_N convention — 0 = real, 1 = fake (most AutoTrain models)
    if "label_0" in r or r == "0":
        return "REAL"
    return "FAKE"


def _map_dima806(raw: str) -> str:
    """dima806 — classes: 'Fake' / 'Real'"""
    return "REAL" if "real" in raw.lower() else "FAKE"


def _map_prithiv(raw: str) -> str:
    """prithivMLmods — classes: 'Real' / 'Fake'"""
    return "REAL" if "real" in raw.lower() else "FAKE"


def _map_wvolf(raw: str) -> str:
    """Wvolf ViT — classes: 'Real' / 'Fake'"""
    return "REAL" if "real" in raw.lower() else "FAKE"


def _map_aditya(raw: str) -> str:
    """
    AdityaManojShinde hybrid CNN — classes vary; safe to use generic map.
    Model card notes: 'Real' / 'Fake' output labels.
    """
    return _generic_map(raw)


def _map_fc63(raw: str) -> str:
    """
    fc63 EfficientNetB0 (Keras saved as HF) — 'Real' / 'Fake' or LABEL_N.
    """
    return _generic_map(raw)


def _map_maanvadr(raw: str) -> str:
    """MaanVad3r pure CNN — typically 'Real' / 'Fake'."""
    return _generic_map(raw)


def _map_dataflow(raw: str) -> str:
    """dataflow/redeepfake EfficientNetB4 — typically 'Real' / 'Fake'."""
    return _generic_map(raw)


def _map_exp0221(raw: str) -> str:
    """prithivMLmods/Deepfake-Detection-Exp-02-21
    Classes: 'Deepfake' → FAKE  |  'Real' → REAL
    Highest accuracy model available on HF Hub — 98.84%, F1 0.9884.
    Deepfake precision: 0.9962 (almost zero false negatives).
    """
    return "FAKE" if "deepfake" in raw.lower() else "REAL"


# ── MODEL REGISTRY ────────────────────────────────────────────────────
# Keys are display names shown in the dropdown.
# Fields:
#   model_id    — Hugging Face repo slug
#   label_fn    — normalises raw HF label → "REAL" | "FAKE"
#   description — shown in the sidebar info expander
#   accuracy    — headline metric shown as a badge in the dropdown
#   arch        — short architecture description
#   badge       — emoji star rating for quick visual ranking

MODEL_REGISTRY = {
    "🏆  Exp-02-21 — ViT Deepfake Detector (98.84% acc)": {
        "model_id": "prithivMLmods/Deepfake-Detection-Exp-02-21",
        "label_fn": _map_exp0221,
        "description": (
            "Best-accuracy pipeline-native model on Hugging Face. "
            "Fine-tuned from google/vit-base-patch16-224-in21k on a curated high-quality dataset. "
            "Deepfake class precision: 0.9962 — virtually zero missed fakes. "
            "F1: 0.9884 on 3200-image balanced test set. Fully HF pipeline compatible."
        ),
        "accuracy": "98.84%",
        "arch": "ViT-base-patch16-224",
        "badge": "🏆",
        "revision": "main",
        # 100% standard HF pipeline — no custom loader needed
    },
    "🥇  AdityaManojShinde — Hybrid CNN (98.6% acc)": {
        "model_id": "AdityaManojShinde/deepfake-detector",
        "label_fn": _map_aditya,
        "description": (
            "Highest-accuracy model in the list. "
            "Dual-stream hybrid: EfficientNet-B4 (spatial) + Xception with SRM filters (frequency). "
            "Trained on 140 k real/fake face images."
        ),
        "accuracy": "98.6%",
        "arch": "EfficientNet-B4 + Xception (SRM)",
        "badge": "🥇",
        "revision": "main",
        # Non-standard config.json → must use direct PyTorch loader, not HF pipeline
        "custom_loader": True,
        # id2label used by this specific model (from model card)
        "id2label": {0: "Real", 1: "Fake"},
        # Expected input size
        "input_size": 224,
    },
    "🧠  dima806 — EfficientNet (DeepFake vs Real)": {
        "model_id": "dima806/deepfake_vs_real_image_detection",
        "label_fn": _map_dima806,
        "description": (
            "EfficientNet-based binary classifier trained on 140 k real/fake images. "
            "Solid baseline with well-documented label schema."
        ),
        "accuracy": "~95%",
        "arch": "EfficientNet",
        "badge": "🧠",
        "revision": "main",
    },
    "🔬  prithivMLmods — Deep Fake Detector": {
        "model_id": "prithivMLmods/Deep-Fake-Detector-Model",
        "label_fn": _map_prithiv,
        "description": (
            "ViT-style image classifier fine-tuned specifically for deepfake detection. "
            "Good at catching modern GAN-based fakes."
        ),
        "accuracy": "~93%",
        "arch": "ViT (fine-tuned)",
        "badge": "🔭",
        "revision": "main",
    },
    "🛡️   Wvolf — ViT Deepfake Detection": {
        "model_id": "Wvolf/ViT_Deepfake_Detection",
        "label_fn": _map_wvolf,
        "description": (
            "Vision Transformer fine-tuned for GAN & diffusion-based fakes. "
            "Strong on synthetic face detection."
        ),
        "accuracy": "~92%",
        "arch": "ViT",
        "badge": "🛡️",
        "revision": "main",
    },
    "⚡  fc63 — EfficientNetB0 CNN v2 (AUC 0.88)": {
        "model_id": "fc63/deepfake-detection-cnn_v2",
        "label_fn": _map_fc63,
        "description": (
            "EfficientNetB0 with transfer learning + custom classification head. "
            "Trained on the DFDC dataset. 80% accuracy, AUC-ROC 0.88, F1 0.80. "
            "Good for balanced real-time frame classification."
        ),
        "accuracy": "80% / AUC 0.88",
        "arch": "EfficientNetB0 (TL)",
        "badge": "⚡",
        "revision": "main",
        # Saved as a Keras .keras file — must use TF loader, not HF pipeline
        "custom_loader": True,
        "keras_loader": True,
        "keras_file": "keras_model/best_model.keras",
        # sigmoid > 0.5 → FAKE (opposite convention to AdityaManojShinde)
        "keras_fake_if_gt": 0.5,
    },
    "🔷  MaanVad3r — Custom Pure CNN (71% acc)": {
        "model_id": "MaanVad3r/DeepFake-Detector",
        "label_fn": _map_maanvadr,
        "description": (
            "Lightweight custom CNN: Conv → Pool → FC layers with ReLU/Sigmoid, "
            "dropout & L2 regularisation. Input: 128×128. "
            "Simple and fast — good for experimentation."
        ),
        "accuracy": "71%",
        "arch": "Custom CNN",
        "badge": "🔷",
        "revision": "main",
        # Saved as cnn_model.h5 (Keras HDF5) — must use TF loader
        "custom_loader": True,
        "keras_loader": True,
        "keras_file": "cnn_model.h5",
        "keras_fake_if_gt": 0.5,  # prediction >= 0.5 → FAKE (per model card)
        "keras_input_size": 128,  # model was trained on 128×128
    },
    "🌊  dataflow — redeepfake EfficientNetB4": {
        "model_id": "dataflow/redeepfake",
        "label_fn": _map_dataflow,
        "description": (
            "EfficientNetB4 CNN for general 2-D flat-image deepfake detection. "
            "Metrics not published on the model card."
        ),
        "accuracy": "N/A",
        "arch": "EfficientNetB4",
        "badge": "🌊",
        "revision": "main",
        # Saved as redeepfake_model_v4.h5 (Keras HDF5) — must use TF loader
        "custom_loader": True,
        "keras_loader": True,
        "keras_file": "redeepfake_model_v4.h5",
        "keras_fake_if_gt": 0.5,
        "keras_input_size": 224,  # EfficientNetB4 standard input
    },
    "🌀  Purnachander-Konda — Swin Deepfake (Swin)": {
        "model_id": "Purnachander-Konda/deepfake-detection-swin",
        "label_fn": _generic_map,
        "description": (
            "Swin Transformer architecture (SwinForImageClassification) fine-tuned for deepfake detection. "
            "Adds diverse self-attention vision transformer patterns to the ensemble."
        ),
        "accuracy": "N/A",
        "arch": "Swin Transformer",
        "badge": "🌀",
        "revision": "main",
    },
    "🔶  computervisionpro — ConvNeXtV2 Real/Fake": {
        "model_id": "computervisionpro/convnextv2-real-fake",
        "label_fn": _generic_map,
        "description": (
            "ConvNeXt V2 architecture trained for Real vs Fake classification. "
            "Adds modernized pure-convolutional network diversity to the ensemble."
        ),
        "accuracy": "N/A",
        "arch": "ConvNeXt V2",
        "badge": "🔶",
        "revision": "main",
    },
    "🏋️  umm-maybe — AI Image Detector (ViT-Large, Deep Analysis)": {
        "model_id": "umm-maybe/AI-image-detector",
        "label_fn": _generic_map,
        "description": (
            "Large Vision Transformer (ViT-L/16) fine-tuned specifically for detecting "
            "AI-generated images. ~300MB+ model with high accuracy on modern generative "
            "outputs (Stable Diffusion, DALL·E, Midjourney). Best choice when accuracy "
            "matters more than speed — ideal for detailed forensic analysis."
        ),
        "accuracy": "~98%",
        "arch": "ViT-Large/16 (300MB+)",
        "badge": "🏋️",
        "revision": "main",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# HYBRID DEEPFAKE DETECTOR ARCHITECTURE
# Matches AdityaManojShinde/deepfake-detector exactly:
#   Spatial stream   — EfficientNet-B4 (ImageNet pretrained)
#   Frequency stream — Xception-like model (via timm or torchvision)
#   Fusion head      — Linear(3840→512→1) with Sigmoid output
# Output: single float sigmoid score (>0.5 = REAL, <=0.5 = FAKE)
# ═══════════════════════════════════════════════════════════════════════


class _SRMFilter(nn.Module):
    """Fixed 3-channel SRM high-pass filter (non-trainable)."""

    def __init__(self):
        super().__init__()
        srm = (
            np.array(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=np.float32,
            )
            / 12.0
        )
        kernel = torch.from_numpy(srm).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 3, 1, 1)
        self.register_buffer("weight", kernel)

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight, padding=2, groups=1)


class HybridDeepfakeDetector(nn.Module):
    """
    Dual-stream CNN:
      - Spatial stream:   EfficientNet-B4 → 1792-d feature
      - Frequency stream: Xception (via timm) → 2048-d feature
      - Fusion head:      concat → Linear(3840→512) → Linear(512→1) → Sigmoid
    """

    def __init__(self):
        super().__init__()
        eff = tvm.efficientnet_b4(weights=None)
        eff.classifier = nn.Identity()
        self.spatial_stream = eff

        try:
            import timm

            xcep = timm.create_model("xception", pretrained=False, num_classes=0)
            freq_out = xcep.num_features
            self.frequency_stream = xcep
        except Exception:
            res = tvm.resnet50(weights=None)
            res.fc = nn.Identity()
            freq_out = 2048
            self.frequency_stream = res

        self.srm = _SRMFilter()

        fused_dim = 1792 + freq_out
        self.fusion = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        spatial_feat = self.spatial_stream(x)
        freq_in = self.srm(x)
        freq_in = torch.clamp(freq_in, -1.0, 1.0)
        freq_feat = self.frequency_stream(freq_in)

        if spatial_feat.dim() > 2:
            spatial_feat = spatial_feat.flatten(1)
        if freq_feat.dim() > 2:
            freq_feat = freq_feat.flatten(1)

        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.fusion(fused)


class HybridModelWrapper:
    """
    Pipeline-compatible wrapper around HybridDeepfakeDetector.
    Returns the same [{label, score}] format as HF pipeline.
    sigmoid > 0.5 → REAL, <= 0.5 → FAKE  (per model card)
    """

    _TRANSFORM = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, image: "Image.Image", top_k: int = 2):
        tensor = self._TRANSFORM(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            prob_real = float(self.model(tensor).squeeze())
        prob_fake = 1.0 - prob_real
        results = [
            {"label": "Real", "score": prob_real},
            {"label": "Fake", "score": prob_fake},
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


@st.cache_resource(show_spinner=False)
def _load_direct(model_id: str, id2label: tuple):
    """
    Load AdityaManojShinde/deepfake-detector:
      1. Download deepfake_detector_phase2.pth via hf_hub_download
      2. Instantiate HybridDeepfakeDetector and load state_dict
      3. Return a HybridModelWrapper (pipeline-compatible callable)
    """
    if not HF_HUB_AVAILABLE:
        return None, (
            "**`huggingface_hub` not installed.**  "
            "Run `pip install huggingface_hub` then restart the app."
        )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    try:
        with st.spinner("📥 Downloading model weights (~158 MB) — first run only…"):
            pth_path = hf_hub_download(
                repo_id=model_id,
                filename="deepfake_detector_phase2.pth",
            )
    except Exception as e:
        return None, (_safe_error(e, "downloading model weights"))

    try:
        model = HybridDeepfakeDetector()
        state = torch.load(pth_path, map_location=device, weights_only=False)
        # state might be the full state_dict or wrapped in a dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(
            state, strict=False
        )  # strict=False tolerates minor key mismatches
        model.to(device).eval()
        return HybridModelWrapper(model, device), None
    except Exception as e:
        return None, (_safe_error(e, "loading model weights"))


# ═══════════════════════════════════════════════════════════════════════
# KERAS MODEL LOADER
# For fc63/deepfake-detection-cnn_v2 which is saved as a .keras file
# and requires TensorFlow, not PyTorch/transformers.
# ═══════════════════════════════════════════════════════════════════════


class KerasModelWrapper:
    """
    Pipeline-compatible wrapper around a TF/Keras model.
    Returns the same [{label, score}] format as HF pipeline.
    Convention: sigmoid > fake_threshold → FAKE, else REAL
    Preprocessing: resize to 224×224, normalize to [0,1]
    """

    def __init__(self, model, fake_threshold: float = 0.5, input_size: int = 224):
        self.model = model
        self.fake_threshold = fake_threshold
        self.input_size = input_size

    def __call__(self, image: "Image.Image", top_k: int = 2):
        import numpy as _np

        # Preprocess: resize to model's expected input size, then /255
        img = image.convert("RGB").resize((self.input_size, self.input_size))
        arr = _np.array(img, dtype=_np.float32) / 255.0  # H×W×3
        arr = arr[_np.newaxis, ...]  # 1×H×W×3

        prob_fake = float(self.model.predict(arr, verbose=0)[0][0])
        prob_real = 1.0 - prob_fake

        if prob_fake > self.fake_threshold:
            results = [
                {"label": "Fake", "score": prob_fake},
                {"label": "Real", "score": prob_real},
            ]
        else:
            results = [
                {"label": "Real", "score": prob_real},
                {"label": "Fake", "score": prob_fake},
            ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


@st.cache_resource(show_spinner=False)
def _load_keras(
    model_id: str, keras_file: str, fake_threshold: float, input_size: int = 224
):
    """
    Download a .keras model file from HF Hub and load it with TensorFlow.
    Returns (KerasModelWrapper | None, error_str | None).
    """
    if not HF_HUB_AVAILABLE:
        return None, (
            "**`huggingface_hub` not installed.**  "
            "Run `pip install huggingface_hub` then restart the app."
        )

    try:
        import tensorflow as tf  # noqa: F401  — just check it's present
    except ImportError:
        return None, (
            "**TensorFlow is required for this model.**  "
            "Run `pip install tensorflow` then restart the app."
        )

    try:
        with st.spinner("📥 Downloading Keras model weights — first run only…"):
            keras_path = hf_hub_download(
                repo_id=model_id,
                filename=keras_file,
            )
    except Exception as e:
        return None, (_safe_error(e, "downloading Keras model"))

    try:
        from tensorflow.keras.layers import DepthwiseConv2D as _DWConv2D
        from tensorflow.keras.models import load_model as _load_model

        # ── Compatibility shim ──────────────────────────────────────────
        # Models saved with Keras ≥ 2.13 include 'groups' in DepthwiseConv2D
        # config. Older TF versions raise "Unrecognised keyword argument".
        # We subclass and silently drop the unknown kwarg before init.
        class _PatchedDepthwiseConv2D(_DWConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        model = _load_model(
            keras_path,
            custom_objects={"DepthwiseConv2D": _PatchedDepthwiseConv2D},
        )
        return KerasModelWrapper(
            model, fake_threshold=fake_threshold, input_size=input_size
        ), None
    except Exception as e:
        return None, (_safe_error(e, "loading Keras model"))


@st.cache_resource(show_spinner=False)
def load_pipeline(
    model_id: str,
    custom_loader: bool = False,
    id2label: tuple = (),
    keras_loader: bool = False,
    keras_file: str = "",
    keras_fake_if_gt: float = 0.5,
    keras_input_size: int = 224,
):
    """
    Unified entry point.  Returns (callable | None, error_str | None).

    For models with custom_loader=True we use DirectModelWrapper.
    For all others we use the standard HF pipeline.
    """
    if custom_loader and keras_loader:
        return _load_keras(model_id, keras_file, keras_fake_if_gt, keras_input_size)
    if custom_loader:
        return _load_direct(model_id, id2label)

    try:
        device = 0 if torch.cuda.is_available() else -1
        # Demo Mode: check local cache first
        local_cache = Path("./models_cache") / model_id.replace("/", "--")
        use_local = os.environ.get("DEEPFAKE_DEMO_MODE") == "1" and local_cache.exists()
        pipe = pipeline(
            task="image-classification",
            model=str(local_cache) if use_local else model_id,
            device=device,
            **({"local_files_only": True} if use_local else {}),
        )
        return pipe, None
    except OSError as e:
        return None, _safe_error(e, "loading model (not found on Hub)")
    except Exception as e:
        return None, _safe_error(e, "loading model pipeline")


# ═══════════════════════════════════════════════════════════════════════
# FACE DETECTION HELPER  (requires opencv)
# ═══════════════════════════════════════════════════════════════════════
def detect_and_crop_face(pil_image: Image.Image) -> Image.Image:
    """
    Detect the largest face in the image and return a cropped PIL image.
    Falls back to the original image if no face is detected or cv2 is absent.
    """
    if not CV2_AVAILABLE:
        return pil_image

    img_array = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    try:
        # Use OpenCV's bundled Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
    except (AttributeError, Exception) as e:
        _sec_logger.warning("Face detection failed or unavailable: %s", e)
        return pil_image

    if len(faces) == 0:
        return pil_image  # no face found — use whole image

    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_array.shape[1], x + w + pad)
    y2 = min(img_array.shape[0], y + h + pad)
    cropped = img_array[y1:y2, x1:x2]
    return Image.fromarray(cropped)


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════
def run_inference(pipe, pil_image: Image.Image, label_fn, top_k: int = 2):
    """
    Run the image-classification pipeline and return structured results.

    Returns
    -------
    list[dict]  — sorted by confidence desc, each dict has:
        label (str): "REAL" or "FAKE"
        raw_label (str): original HF label
        confidence (float): 0–100
    """
    raw_results = pipe(pil_image, top_k=top_k)  # list of {label, score}
    processed = []
    for r in raw_results:
        processed.append(
            {
                "label": label_fn(r["label"]),
                "raw_label": r["label"],
                "confidence": round(r["score"] * 100, 2),
            }
        )
    # Sort descending by confidence
    processed.sort(key=lambda x: x["confidence"], reverse=True)
    return processed


def _get_model_weight(acc_string: str) -> float:
    """Extract weight (0 to 1) from the accuracy string, default to 0.75 if N/A."""
    if acc_string == "N/A" or not acc_string:
        return 0.75
    # Try to find a percentage number like "98.84%" or "95%"
    import re

    match = re.search(r"(\d+\.?\d*)", acc_string)
    if match:
        val = float(match.group(1))
        # If the number is like 98.84, convert to 0.9884
        # If it's like 0.88, keep as 0.88
        if val > 1.0:
            return val / 100.0
        return val
    return 0.75


# ═══════════════════════════════════════════════════════════════════════
# SHARED HELPERS — DRY pipeline loading & result rendering
# ═══════════════════════════════════════════════════════════════════════
def _load_model_from_cfg(cfg: dict):
    """Load a model pipeline from a MODEL_REGISTRY config dict.
    Returns (pipe, error_string_or_None)."""
    return load_pipeline(
        cfg["model_id"],
        custom_loader=cfg.get("custom_loader", False),
        id2label=tuple((cfg.get("id2label") or {}).items()),
        keras_loader=cfg.get("keras_loader", False),
        keras_file=cfg.get("keras_file", ""),
        keras_fake_if_gt=cfg.get("keras_fake_if_gt", 0.5),
        keras_input_size=cfg.get("keras_input_size", 224),
    )


def render_analysis_card(col, model_name, results, accuracy=""):
    """Render a Netflix-style result card inside the given Streamlit column.
    Used by Compare, Panel, and Ensemble modes for consistent output."""
    top = results[0]
    is_real = top["label"] == "REAL"
    conf_pct = top["confidence"]

    color = "#46d369" if is_real else "#E50914"
    glitch_class = "" if is_real else "glitch-text"
    pulse_class = "glow-calm" if is_real else "glow-danger"

    html = f'''
    <div class="netflix-result-card {pulse_class}">
        <div class="result-header">
            <span class="result-model-name">{model_name}</span>
            <span class="result-accuracy">{accuracy}</span>
        </div>

        <div class="result-verdict-container">
            <div class="scanning-line"></div>
            <div class="verdict-text {glitch_class}" style="color: {color};" data-text="{top["label"]}">{top["label"]}</div>
        </div>

        <div class="card" style="margin: 0.5rem 1rem; padding: 0.75rem; background: rgba(255,255,255,0.03); border-color: {color}40; border-left: 3px solid {color}; text-align: left;">
            <div style="font-size: 0.75rem; color: #d1d5db; line-height: 1.4;">
                {generate_explanation(top["label"], conf_pct, "single")}
            </div>
        </div>

        <div class="result-stats">
            <div class="stat-row">
                <span>Confidence</span>
                <span style="color: {color}; font-weight: bold;">{conf_pct}%</span>
            </div>
            <div class="sleek-progress-bg">
                <div class="sleek-progress-fill" style="width: {conf_pct}%; background-color: {color};"></div>
            </div>
        </div>

        <div class="result-details">
    '''

    for r in results:
        r_color = "#46d369" if r["label"] == "REAL" else "#E50914"
        html += f"""
            <div class="detail-row">
                <span>{r["label"]} <span class="raw-label">({r["raw_label"]})</span></span>
                <span style="color: {r_color}; font-weight: bold;">{r["confidence"]}%</span>
            </div>
        """

    html += """
        </div>
    </div>
    """

    with col:
        st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# DIGITAL FORENSICS HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _get_exif_data(pil_img):
    """Extract EXIF metadata as a readable dict."""
    exif_raw = pil_img.getexif()
    if not exif_raw:
        return {}
    readable = {}
    for tag_id, value in exif_raw.items():
        tag_name = ExifTags.TAGS.get(tag_id, f"Unknown-{tag_id}")
        # Skip very long binary blobs
        if isinstance(value, bytes) and len(value) > 200:
            readable[tag_name] = f"<binary {len(value)} bytes>"
        else:
            readable[tag_name] = str(value)
    return readable


def render_forensics_panel(pil_img, file_bytes):
    """Render the Digital Forensics expander with EXIF, hash, and AI flags."""
    with st.expander("🔍 Digital Forensics", expanded=False):
        # File hash
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        file_size_kb = len(file_bytes) / 1024

        # Compression estimate
        comp_label = "N/A (Not JPEG)"
        if pil_img.format in ("JPEG", "MPO"):
            q_table = pil_img.info.get("quantization")
            if q_table and 0 in q_table:
                # heuristic to estimate compression from luma table
                q_sum = sum(q_table[0])
                if q_sum > 4000:
                    comp_label = "High Compression (Low Quality)"
                elif q_sum > 2000:
                    comp_label = "Medium Compression"
                else:
                    comp_label = "Low Compression (High Quality)"
            else:
                comp_label = "Unknown (No Q-Table)"

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(
                f"""
            <div class="card" style="padding:1rem;">
                <div style="font-size:0.68rem;color:#d1d5db;letter-spacing:0.12em;
                     text-transform:uppercase;margin-bottom:0.5rem;">File Info</div>
                <div class="conf-row"><span>📐 Dimensions</span>
                    <span class="conf-val">{pil_img.width} × {pil_img.height} px</span></div>
                <div class="conf-row"><span>🎨 Color Mode</span>
                    <span class="conf-val">{pil_img.mode}</span></div>
                <div class="conf-row"><span>📦 File Size</span>
                    <span class="conf-val">{file_size_kb:.1f} KB</span></div>
                <div class="conf-row"><span>🗂️ Format</span>
                    <span class="conf-val">{pil_img.format or "N/A"}</span></div>
                <div class="conf-row"><span>🗜️ Compression</span>
                    <span class="conf-val">{comp_label}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_f2:
            st.markdown(
                f"""
            <div class="card" style="padding:1rem;">
                <div style="font-size:0.68rem;color:#d1d5db;letter-spacing:0.12em;
                     text-transform:uppercase;margin-bottom:0.5rem;">Integrity Hash</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                     color:#a78bfa;word-break:break-all;line-height:1.6;">
                    SHA-256:<br>{sha256}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # EXIF analysis
        exif = _get_exif_data(pil_img)

        if not exif:
            st.markdown(
                """
            <div class="card" style="padding:1rem;border-color:rgba(248,113,113,0.3);">
                <div style="color:#f87171;font-size:0.85rem;font-weight:600;">
                    ⚠️ No EXIF data — common in AI-generated images
                </div>
                <div style="color:#d1d5db;font-size:0.75rem;margin-top:0.3rem;">
                    Authentic photographs from cameras typically contain EXIF metadata
                    (camera model, date, GPS, etc.). AI-generated images and heavily
                    processed images often have stripped metadata.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            software = exif.get("Software", None)
            if software:
                st.markdown(
                    f"""
                <div class="card" style="padding:1rem;border-color:rgba(251,191,36,0.3);">
                    <div style="color:#fbbf24;font-size:0.85rem;font-weight:600;">
                        🛠️ Software Tag Detected: {software}
                    </div>
                    <div style="color:#d1d5db;font-size:0.75rem;margin-top:0.3rem;">
                        This image was processed or edited with the above software.
                        Editing tools leave traces in EXIF metadata.
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            elif any(k in exif for k in ("Make", "Model", "DateTimeOriginal")):
                st.markdown(
                    """
                <div class="card" style="padding:1rem;border-color:rgba(52,211,153,0.3);">
                    <div style="color:#34d399;font-size:0.85rem;font-weight:600;">
                        ✅ EXIF data present
                    </div>
                    <div style="color:#d1d5db;font-size:0.75rem;margin-top:0.3rem;">
                        Consistent with an unedited camera-original photo (not proof of authenticity, but a positive signal).
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
            <div style="font-size:0.68rem;color:#d1d5db;letter-spacing:0.12em;
                 text-transform:uppercase;margin:0.5rem 0;">EXIF Fields</div>
            """,
                unsafe_allow_html=True,
            )
            # Display key EXIF fields
            display_tags = [
                "Make",
                "Model",
                "DateTime",
                "DateTimeOriginal",
                "Software",
                "ExifImageWidth",
                "ExifImageHeight",
                "FocalLength",
                "ISOSpeedRatings",
                "ExposureTime",
                "FNumber",
                "GPSInfo",
            ]
            for tag in display_tags:
                if tag in exif:
                    st.markdown(
                        f'<div class="conf-row"><span>{tag}</span>'
                        f'<span class="conf-val">{exif[tag][:60]}</span></div>',
                        unsafe_allow_html=True,
                    )

        # Error Level Analysis (ELA)
        st.markdown(
            '<div style="font-size:0.68rem;color:#d1d5db;letter-spacing:0.12em;text-transform:uppercase;margin:1.5rem 0 0.5rem;">Error Level Analysis (ELA)</div>',
            unsafe_allow_html=True,
        )
        try:
            import io

            from PIL import ImageChops, ImageEnhance

            temp_io = io.BytesIO()
            pil_img.convert("RGB").save(temp_io, "JPEG", quality=90)
            temp_io.seek(0)
            resaved = Image.open(temp_io).convert("RGB")

            ela_image = ImageChops.difference(pil_img.convert("RGB"), resaved)
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema]) if extrema else 1
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            st.image(
                ela_image,
                use_container_width=True,
                caption="ELA Heatmap (Bright areas indicate potential manipulation or differing compression levels)",
            )
        except Exception as e:
            st.toast(f"ELA Failed: {e}", icon="❌")


# ═══════════════════════════════════════════════════════════════════════
# EXPLANATION TEMPLATES
# ═══════════════════════════════════════════════════════════════════════


def generate_explanation(
    verdict: str, confidence: float, mode: str = "single", num_models: int = 1
) -> str:
    """Generate dynamic explanations based on verdict, confidence, and analysis mode."""
    if mode == "ensemble":
        if verdict == "FAKE":
            return (
                f"**High-Confidence Ensemble FAKE** — Across {num_models} models, the weighted average confidence is {confidence:.1f}%. The consensus strongly indicates synthetic manipulation."
                if confidence > 80
                else f"**Moderate Ensemble FAKE** — The ensemble average is {confidence:.1f}%. While some models detect manipulation, there is slight disagreement."
            )
        else:
            return (
                f"**Strong Consensus REAL** — The ensemble average is {confidence:.1f}%. The models agree the media is authentic."
                if confidence > 80
                else f"**Moderate Consensus REAL** — The ensemble leans towards authentic with {confidence:.1f}% confidence."
            )

    if mode == "panel":
        if verdict == "FAKE":
            return f"**Panel FAKE** — The majority of the {num_models} models agree this is manipulated ({confidence:.1f}% avg)."
        else:
            return f"**Panel REAL** — The models predominantly classify this as authentic ({confidence:.1f}% avg)."

    if mode == "video":
        if verdict == "FAKE":
            return f"**Video FAKE** — Temporal inconsistencies detected across frames with {confidence:.1f}% peak confidence."
        else:
            return f"**Video REAL** — Frame-by-frame analysis found natural continuity ({confidence:.1f}% confidence)."

    # Default (single)
    if verdict == "FAKE":
        if confidence >= 90:
            return "**High confidence FAKE** — flagged for GAN/diffusion artifacts like unnatural texture or edge anomalies."
        elif confidence >= 70:
            return "**Moderate confidence FAKE** — subtle artifacts detected. Cross-checking recommended."
        else:
            return (
                "**Low confidence FAKE** — weak signals of manipulation. Inconclusive."
            )
    else:
        if confidence >= 90:
            return "**High confidence REAL** — consistent with natural, unedited camera captures."
        elif confidence >= 70:
            return "**Moderate confidence REAL** — generally authentic but slight compression artifacts exist."
        else:
            return "**Low confidence REAL** — borderline result, potential noise or filters confusing the model."


def get_explanation_text(verdict: str, confidence: float) -> str:
    """Return natural-language reasoning based on verdict + confidence tier."""
    if verdict == "FAKE":
        if confidence >= 90:
            return (
                "**High confidence FAKE detected** — the model flagged inconsistencies "
                "typical of GAN-based or diffusion-model generation. Common artifacts "
                "include unnatural skin texture, irregular lighting gradients, edge "
                "blending anomalies, and subtle asymmetry in facial features."
            )
        elif confidence >= 70:
            return (
                "**Moderate confidence FAKE** — subtle artifacts detected that suggest "
                "synthetic manipulation. The model identified minor blending artifacts, "
                "slight color inconsistencies, or unnatural texture patterns. Consider "
                "cross-checking with another model for a second opinion."
            )
        else:
            return (
                "**Low confidence FAKE** — the image shows weak signals of manipulation. "
                "The result is inconclusive; the detected anomalies could also appear in "
                "heavily compressed or filtered authentic photos. Manual inspection or "
                "multi-model panel analysis is recommended."
            )
    else:  # REAL
        if confidence >= 90:
            return (
                "**High confidence REAL** — the image shows natural characteristics "
                "consistent with authentic photographs. The model found consistent "
                "lighting, organic textures, natural color gradients, and no traces "
                "of synthetic generation artifacts."
            )
        elif confidence >= 70:
            return (
                "**Moderate confidence REAL** — the image appears authentic but has "
                "some ambiguous features. The model finds no strong manipulation "
                "signatures, though minor compression artifacts or filters may be present."
            )
        else:
            return (
                "**Low confidence REAL** — the model leans toward authentic but cannot "
                "rule out manipulation. The image may have undergone heavy post-processing "
                "that obscures clear signals. Consider analyzing with additional models."
            )


# ═══════════════════════════════════════════════════════════════════════
# VIDEO FRAME EXTRACTION
# ════════════════════════════════════════════════════════════════════════
_AUDIO_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"


@st.cache_resource(show_spinner=False)
def load_audio_pipeline():
    """Load the audio deepfake detection model. Returns (pipe, error_str)."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            task="audio-classification",
            model=_AUDIO_MODEL_ID,
            device=device,
        )
        return pipe, None
    except Exception as e:
        return None, _safe_error(e, "loading audio model")


# ═══════════════════════════════════════════════════════════════════════
# PLOTLY CIRCULAR GAUGE
# ═══════════════════════════════════════════════════════════════════════
def make_gauge(value: float, label: str, is_real: bool) -> go.Figure:
    """
    Render a half-donut gauge with the confidence score.
    Green for REAL, Red for FAKE.
    """
    accent = "#34d399" if is_real else "#f87171"
    track = "#d1d5db"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "font": {"size": 34, "color": accent, "family": "Space Mono"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickcolor": "rgba(0,0,0,0)",
                    "tickfont": {"color": "#9ca3af", "size": 10},
                },
                "bar": {"color": accent, "thickness": 0.28},
                "bgcolor": track,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, value], "color": "rgba(0,0,0,0)"},
                    {"range": [value, 100], "color": track},
                ],
                "threshold": {
                    "line": {"color": accent, "width": 4},
                    "thickness": 0.85,
                    "value": value,
                },
            },
            title={
                "text": f"<b>{label}</b><br><span style='font-size:11px;color:#d1d5db'>Confidence</span>",
                "font": {"size": 15, "color": "#e5e7eb", "family": "Space Mono"},
            },
        )
    )

    fig.update_layout(
        height=240,
        margin=dict(t=60, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center;margin-bottom:1.5rem;'>
        <span style='font-size:2.4rem;'>🔍</span>
        <p style='font-family:Space Mono,monospace;font-size:0.75rem;
                  letter-spacing:0.2em;color:#9ca3af;margin:0.3rem 0 0;'>
            DEEPFAKE LAB
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-label">Model Selection</p>', unsafe_allow_html=True)
    chosen_model_name = st.selectbox(
        label="model",
        options=list(MODEL_REGISTRY.keys()),
        label_visibility="collapsed",
    )
    model_cfg = MODEL_REGISTRY[chosen_model_name]

    with st.expander("ℹ️ Model Info", expanded=True):
        st.markdown(f"""
**ID:** `{model_cfg["model_id"]}`

**Architecture:** {model_cfg["arch"]}

**Accuracy:** {model_cfg["accuracy"]}

**About:** {model_cfg["description"]}

🔗 [View on Hugging Face](https://huggingface.co/{model_cfg["model_id"]})
        """)

    st.markdown("---")

    # Face-detect toggle

    st.sidebar.markdown("### ⚙️ Global Settings")

    demo_mode = st.sidebar.toggle(
        "🎬 Demo Mode (Offline-Safe)",
        value=False,
        help="When enabled, loads models from local cache (./models_cache/) first. Use 'python download_models.py' to pre-cache.",
    )
    if demo_mode:
        os.environ["DEEPFAKE_DEMO_MODE"] = "1"
        missing = [
            n
            for n, c in MODEL_REGISTRY.items()
            if not (Path("./models_cache") / c["model_id"].replace("/", "--")).exists()
        ]
        if missing:
            st.sidebar.caption(
                f"⚠️ {len(missing)} model(s) not cached locally — will fall back to live download."
            )
    else:
        os.environ.pop("DEEPFAKE_DEMO_MODE", None)

    use_face_crop = st.sidebar.toggle(
        "🧑 Auto Face Crop (OpenCV)",
        value=False,
        help="Detect & crop the largest face before inference. Improves accuracy for portrait shots.",
        disabled=not CV2_AVAILABLE,
    )
    if not CV2_AVAILABLE:
        st.caption("Install `opencv-python-headless` to enable face crop.")

    # ── Analysis mode selector ────────────────────────────────────────
    st.sidebar.markdown("### 📊 Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "mode",
        [
            "🔬 Single Model",
            "⚖️ Compare 2 Models",
            "🧪 4-Model Panel",
            "🌐 Full Ensemble (All Models)",
        ],
        index=3,  # Default to Full Ensemble
        label_visibility="collapsed",
        help="Single: one model. Compare: 2 models side-by-side. Panel: all 4 with verdict. Ensemble: All models weighted average.",
    )

    compare_mode = analysis_mode == "⚖️ Compare 2 Models"
    ensemble_mode = analysis_mode == "🌐 Full Ensemble (All Models)"
    quad_mode = analysis_mode == "🧪 4-Model Panel"
    all_model_names = list(MODEL_REGISTRY.keys())

    if compare_mode:
        compare_model_name = st.selectbox(
            "Second model",
            [m for m in all_model_names if m != chosen_model_name],
        )

    if quad_mode:
        remaining = [m for m in all_model_names if m != chosen_model_name]
        st.caption("Select 3 more models for the panel:")
        quad_model_2 = st.selectbox("Model 2", remaining, index=0, key="q2")
        remaining2 = [m for m in remaining if m != quad_model_2]
        quad_model_3 = st.selectbox("Model 3", remaining2, index=0, key="q3")
        remaining3 = [m for m in remaining2 if m != quad_model_3]
        quad_model_4 = st.selectbox("Model 4", remaining3, index=0, key="q4")

    st.markdown("---")
    st.markdown(
        """
    <div style='font-size:0.75rem;color:#9ca3af;line-height:1.7;'>
        <b style='color:#9ca3af;'>How to run</b><br>
        <code style='color:#7b61ff;'>pip install streamlit transformers torch torchvision Pillow plotly opencv-python-headless librosa</code><br><br>
        <code style='color:#7b61ff;'>streamlit run deepfake_detector.py</code>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════
# MAIN CONTENT — TABBED LAYOUT
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 & 5: CONTENT ROWS AND DETAIL MODAL (Netflix Style)
# ═══════════════════════════════════════════════════════════════════════

THUMBNAILS = {
    "prithivMLmods/Deepfake-Detection-Exp-02-21": f"data:image/png;base64,{_get_asset_b64('assets/asset_1.png')}",
    "AdityaManojShinde/deepfake-detector": f"data:image/png;base64,{_get_asset_b64('assets/asset_2.png')}",
    "dima806/deepfake_vs_real_image_detection": f"data:image/png;base64,{_get_asset_b64('assets/asset_3.png')}",
    "prithivMLmods/Deep-Fake-Detector-Model": f"data:image/png;base64,{_get_asset_b64('assets/asset_4.png')}",
    "Wvolf/ViT_Deepfake_Detection": f"data:image/png;base64,{_get_asset_b64('assets/asset_5.png')}",
    "fc63/deepfake-detection-cnn_v2": f"data:image/png;base64,{_get_asset_b64('assets/asset_6.png')}",
    "MaanVad3r/DeepFake-Detector": f"data:image/png;base64,{_get_asset_b64('assets/asset_7.png')}",
    "dataflow/redeepfake": f"data:image/png;base64,{_get_asset_b64('assets/asset_8.png')}",
    "Purnachander-Konda/deepfake-detection-swin": f"data:image/png;base64,{_get_asset_b64('assets/asset_9.png')}",
    "computervisionpro/convnextv2-real-fake": f"data:image/png;base64,{_get_asset_b64('assets/asset_10.png')}",
    "umm-maybe/AI-image-detector": f"data:image/png;base64,{_get_asset_b64('assets/asset_11.png')}",
}

_model_data = []
for name, cfg in MODEL_REGISTRY.items():
    thumb_b64 = THUMBNAILS.get(
        cfg["model_id"], THUMBNAILS["prithivMLmods/Deepfake-Detection-Exp-02-21"]
    )

    short = name.split("—")[1].strip() if "—" in name else name
    _model_data.append(
        {
            "short": short,
            "badge": cfg["badge"],
            "acc": cfg["accuracy"],
            "arch": cfg["arch"],
            "desc": cfg["description"],
            "thumb": thumb_b64,
        }
    )

_model_data_json = json.dumps(_model_data)

_cards_html = ""
_modals_html = ""
for idx, (name, cfg) in enumerate(MODEL_REGISTRY.items()):
    short = name.split("-")[1].strip() if "-" in name else name
    thumb_b64 = THUMBNAILS.get(cfg["model_id"], THUMBNAILS["prithivMLmods/Deepfake-Detection-Exp-02-21"])
    bg_img = thumb_b64
    desc = cfg["description"]
    m_id = f"m{idx}"
    acc = cfg.get("accuracy", "90%")
    arch = cfg.get("arch", "Unknown")
    
    _cards_html += f"""
        <div class="netflix-card" onclick="document.getElementById('modal-{m_id}').style.display='flex';">
            <img src="{bg_img}" alt="{short}">
            <div class="card-info">
                <div class="card-title">{short}</div>
                <div class="card-desc">{desc[:90]}...</div>
            </div>
        </div>
    """
    _modals_html += f"""
    <div id="modal-{m_id}" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:99999999; background:rgba(0,0,0,0.8); justify-content:center; align-items:center;" onclick="this.style.display='none';">
        <div class="netflix-modal-box" onclick="event.stopPropagation();">
            <button class="modal-close-btn" onclick="document.getElementById('modal-{m_id}').style.display='none';">X</button>
            <div class="modal-banner" style="background-image: url('{bg_img}');">
                <div class="modal-banner-gradient"></div>
            </div>
            <div class="modal-content-wrapper">
                <div class="modal-title">{short}</div>
                <div class="modal-actions">
                    <button class="modal-play-btn" onclick="document.getElementById('modal-{m_id}').style.display='none'; const fileInput = document.querySelector('input[type=file]'); if(fileInput) fileInput.click(); else alert('Please upload a file in the main UI.');">
                        <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Analyze
                    </button>
                </div>
                <div class="modal-body-grid">
                    <div>
                        <div class="modal-meta-row">
                            <span class="modal-match">{acc} Match</span>
                            <span class="modal-year">2026</span>
                            <span class="modal-hd">HD</span>
                        </div>
                        <div class="modal-desc">{desc}</div>
                    </div>
                    <div class="modal-right-col">
                        <div><span>Architecture:</span> <strong>{arch}</strong></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

html_str = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .netflix-row-container {{ padding: 20px 4%; font-family: 'Inter', sans-serif; margin-top: 10px; }}
    .row-title {{ color: #e5e5e5; font-size: 1.4vw; font-weight: 500; margin-bottom: 0.5vw; display: flex; align-items: center; gap: 8px; cursor: pointer; }}
    .row-chevron {{ color: #54b9c5; font-size: 1.2vw; opacity: 0; transition: opacity 0.3s, transform 0.3s; display: inline-block; }}
    .row-title:hover .row-chevron {{ opacity: 1; transform: translateX(5px); }}
    
    .row-scroll-wrapper {{ position: relative; }}
    .row-cards {{
        display: flex; gap: 8px; overflow-x: auto; padding: 20px 0;
        scrollbar-width: none; -ms-overflow-style: none; scroll-behavior: smooth;
    }}
    .row-cards::-webkit-scrollbar {{ display: none; }}
    
    .row-arrow {{
        position: absolute; top: 0; bottom: 0; width: 4%; background: rgba(0,0,0,0.5);
        color: #fff; border: none; font-size: 2vw; cursor: pointer; z-index: 10;
        opacity: 0; transition: opacity 0.3s, background 0.3s;
        display: flex; align-items: center; justify-content: center;
    }}
    .row-scroll-wrapper:hover .row-arrow {{ opacity: 1; }}
    .row-arrow:hover {{ background: rgba(0,0,0,0.7); font-size: 2.5vw; }}
    .left-arrow {{ left: -4%; border-top-right-radius: 4px; border-bottom-right-radius: 4px; }}
    .right-arrow {{ right: -4%; border-top-left-radius: 4px; border-bottom-left-radius: 4px; }}
    
    .netflix-card {{
        flex: 0 0 16.66666667%; min-width: 250px; position: relative;
        border-radius: 4px; overflow: hidden; cursor: pointer;
        transition: transform 0.3s cubic-bezier(0.2,0.8,0.2,1), box-shadow 0.3s;
        background: #181818;
    }}
    .netflix-card img {{ width: 100%; height: 140px; object-fit: cover; transition: transform 0.3s; }}
    .netflix-card:hover {{
        transform: scale(1.15) translateY(-10px);
        z-index: 20;
        box-shadow: 0 10px 20px rgba(0,0,0,0.8);
    }}
    .card-info {{ padding: 12px; position: absolute; bottom: 0; left: 0; width: 100%; background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, transparent 100%); }}
    .card-title {{ font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 4px; }}
    .card-desc {{ font-size: 10px; color: #a3a3a3; line-height: 1.3; }}
    
    /* Modal Styles */
    .netflix-modal-box {{
        width: 90%; max-width: 850px; max-height: 90vh;
        background: #181818; border-radius: 8px;
        overflow-y: auto; overflow-x: hidden; position: relative;
        box-shadow: 0 0 40px rgba(0,0,0,0.5); font-family: 'Inter', sans-serif;
    }}
    .modal-close-btn {{
        position: absolute; top: 20px; right: 20px; width: 36px; height: 36px; border-radius: 50%;
        background: #181818; color: #fff; border: 2px solid #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; font-weight: bold; cursor: pointer; z-index: 10;
        transition: background 0.2s, color 0.2s;
    }}
    .modal-close-btn:hover {{ background: #fff; color: #181818; }}
    .modal-banner {{ width: 100%; height: 400px; background-size: cover; background-position: center; position: relative; }}
    .modal-banner-gradient {{ position: absolute; bottom: 0; left: 0; width: 100%; height: 50%; background: linear-gradient(to top, #181818 0%, transparent 100%); }}
    .modal-content-wrapper {{ padding: 0 40px 40px 40px; margin-top: -60px; position: relative; z-index: 2; }}
    .modal-title {{ font-family: 'Bebas Neue', cursive; font-size: 3.5rem; color: #fff; text-shadow: 0 2px 8px rgba(0,0,0,0.8); margin-bottom: 16px; line-height: 1; }}
    .modal-actions {{ display: flex; gap: 10px; margin-bottom: 24px; align-items: center; }}
    .modal-play-btn {{ background: #fff; color: #000; border: none; border-radius: 4px; padding: 10px 24px; font-size: 1.2rem; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: background 0.2s; }}
    .modal-play-btn:hover {{ background: rgba(255,255,255,0.8); }}
    .modal-body-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 40px; }}
    .modal-meta-row {{ display: flex; gap: 12px; margin-bottom: 12px; font-weight: 700; align-items: center; font-size: 15px; color: #fff; }}
    .modal-match {{ color: #46d369; }}
    .modal-year {{ color: #e5e5e5; font-weight: 400; }}
    .modal-hd {{ border: 1px solid rgba(255,255,255,0.4); padding: 0 4px; font-size: 12px; border-radius: 2px; color: #e5e5e5; }}
    .modal-desc {{ color: #fff; font-size: 16px; line-height: 1.5; margin-bottom: 20px; }}
    .modal-right-col {{ color: #b3b3b3; font-size: 14px; line-height: 1.6; }}
    .modal-right-col span {{ color: #777; }}
    .modal-right-col strong {{ color: #fff; font-weight: normal; }}
    </style>

    <div class="netflix-row-container" id="row-detector">
        <h2 class="row-title">Choose Your Detector <span class="row-chevron">></span></h2>
        <div class="row-scroll-wrapper">
            <button class="row-arrow left-arrow" onclick="document.querySelector('.row-cards').scrollBy({{left: -600, behavior: 'smooth'}})">&#10094;</button>
            <div class="row-cards">
                {_cards_html}
            </div>
            <button class="row-arrow right-arrow" onclick="document.querySelector('.row-cards').scrollBy({{left: 600, behavior: 'smooth'}})">&#10095;</button>
        </div>
    </div>
    {_modals_html}
    """
html_str = "".join(line.strip() for line in html_str.splitlines())
st.markdown(html_str, unsafe_allow_html=True)



tab_image, tab_video, tab_audio, tab_url, tab_history = st.tabs(
    ["🖼️ Image", "🎬 Video", "🎵 Audio", "🔗 URL Scanner", "📊 History"]
)

with tab_image:
    import requests

    class UploadedFileMock:
        def __init__(self, name, bytes_data):
            self.name = name
            self.type = "image/jpeg"
            self.bytes_data = bytes_data

        def read(self):
            return self.bytes_data

        def seek(self, arg):
            pass

    st.markdown('<p class="section-label">Input Image</p>', unsafe_allow_html=True)

    input_method = st.radio(
        "Method",
        ["Upload File", "Paste URL"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_file = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            label="drop an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            help="Supported formats: JPG, JPEG, PNG, WEBP",
            key="img_uploader",
        )
    else:
        img_url = st.text_input(
            "Paste Image URL",
            placeholder="https://example.com/image.jpg",
            label_visibility="collapsed",
        )
        if img_url:
            try:
                with st.spinner("Fetching image..."):
                    response = requests.get(
                        img_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    response.raise_for_status()

                file_name = img_url.split("/")[-1]
                if "?" in file_name:
                    file_name = file_name.split("?")[0]
                if not file_name or "." not in file_name:
                    file_name = "downloaded_image.jpg"

                uploaded_file = UploadedFileMock(file_name, response.content)
            except Exception as e:
                st.toast(f"❌ Failed to fetch image: {e}", icon="❌")

    if uploaded_file is not None:
        # Pass through the validation pipeline as requested
        val_img, file_bytes, val_err = validate_image_upload(uploaded_file)
        if val_err:
            st.toast(f"❌ {val_err}", icon="❌")
            st.stop()

        safe_filename = _sanitize_filename(uploaded_file.name)

        #  Layout: image preview + controls
        preview_col, ctrl_col = st.columns([1.1, 0.9], gap="large")

        with preview_col:
            try:
                pil_img = Image.open(io.BytesIO(file_bytes))
                # Keep format info before converting
                _img_format = pil_img.format
                pil_img_with_exif = pil_img.copy()  # keep original for EXIF
                pil_img = pil_img.convert("RGB")
            except Exception:
                st.toast(
                    "⚠️ The uploaded file appears to be corrupted or is not a valid image format.",
                    icon="❌",
                )
                st.stop()
            st.markdown('<p class="section-label">Preview</p>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.caption(
                f"📐 {pil_img.width} × {pil_img.height} px  •  {uploaded_file.type}"
            )

            # ── Digital Forensics Panel ────────────────────────────────
            render_forensics_panel(pil_img_with_exif, file_bytes)

            #  Web Verification Panel
            with st.expander("🌐 Web Verification (Reverse Search)", expanded=False):
                st.markdown(
                    "<p style='font-size:0.85rem;color:#9ca3af;'>Search the internet for visually similar instances of this image to help corroborate its authenticity.</p>",
                    unsafe_allow_html=True,
                )
                if st.button("Search Google Lens", key="btn_lens_search"):
                    with st.spinner("Querying Google Lens..."):
                        try:
                            import os
                            import tempfile

                            from googlelens import GoogleLens

                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".jpg"
                            ) as tmp:
                                pil_img.convert("RGB").save(tmp.name)
                                tmp_path = tmp.name

                            lens = GoogleLens()
                            result = lens.search_by_file(tmp_path)

                            if result and result.get("related_images"):
                                st.success(
                                    f"Found {len(result['related_images'])} visually similar matches online!"
                                )
                                for i, match in enumerate(result["related_images"][:3]):
                                    st.markdown(
                                        f"**Match {i + 1}:** [{match.get('title', 'Link')}]({match.get('url', '#')})"
                                    )
                            else:
                                st.toast(
                                    "No visually similar images found. This could mean it is entirely unique/AI-generated.",
                                    icon="⚠️",
                                )

                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                        except Exception as e:
                            st.toast(f"Search failed: {e}", icon="❌")

        with ctrl_col:
            st.markdown(
                '<p class="section-label">Analysis Controls</p>', unsafe_allow_html=True
            )

            if ensemble_mode:
                card_html = f"""
                <div class="card">
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                        <span style='font-size:0.75rem;color:#d1d5db;letter-spacing:0.12em;text-transform:uppercase;'>Active Mode</span>
                        <span style='font-size:0.72rem;background:rgba(123,97,255,0.15);color:#a78bfa;
                                     padding:0.15rem 0.6rem;border-radius:20px;border:1px solid rgba(123,97,255,0.3);'>
                            Maximum Accuracy
                        </span>
                    </div>
                    <div style='font-family:Space Mono,monospace;font-size:0.85rem;color:#a78bfa;margin:0.35rem 0 0.25rem;'>
                        🌐 Full Ensemble
                    </div>
                    <div style='font-size:0.75rem;color:#9ca3af;'>
                        🏗️ Utilizing {len(MODEL_REGISTRY)} independent models
                    </div>
                </div>
                """
            elif quad_mode:
                card_html = """
                <div class="card">
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                        <span style='font-size:0.75rem;color:#d1d5db;letter-spacing:0.12em;text-transform:uppercase;'>Active Mode</span>
                    </div>
                    <div style='font-family:Space Mono,monospace;font-size:0.85rem;color:#a78bfa;margin:0.35rem 0 0.25rem;'>
                        🧪 4-Model Panel
                    </div>
                    <div style='font-size:0.75rem;color:#9ca3af;'>
                        🏗️ Top 4 Models side-by-side
                    </div>
                </div>
                """
            else:
                card_html = f"""
                <div class="card">
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                        <span style='font-size:0.75rem;color:#d1d5db;letter-spacing:0.12em;text-transform:uppercase;'>Active Model</span>
                        <span style='font-size:0.72rem;background:rgba(123,97,255,0.15);color:#a78bfa;
                                     padding:0.15rem 0.6rem;border-radius:20px;border:1px solid rgba(123,97,255,0.3);'>
                            {model_cfg.get("accuracy", "N/A")} acc
                        </span>
                    </div>
                    <div style='font-family:Space Mono,monospace;font-size:0.82rem;color:#a78bfa;margin:0.35rem 0 0.25rem;word-break:break-all;'>
                        {model_cfg["model_id"]}
                    </div>
                    <div style='font-size:0.75rem;color:#9ca3af;'>
                        🏗️ {model_cfg.get("arch", "Unknown")}
                    </div>
                </div>
                """
            st.markdown(card_html, unsafe_allow_html=True)

            analyze_btn = st.button(
                "🔬 Analyze Image", use_container_width=True, key="img_analyze"
            )

        # ── Inference ──────────────────────────────────────────────────
        if analyze_btn:
            if "first_run_done" not in st.session_state:
                st.toast(
                    "🔥 First run detected: Warming up models. This may take a few extra seconds."
                )
                st.session_state.first_run_done = True

            # -- Prepare image --
            inference_img = pil_img.copy()
            if use_face_crop:
                with st.spinner("🧑 Detecting face…"):
                    cropped = detect_and_crop_face(pil_img)
                    if cropped is not pil_img:
                        inference_img = cropped
                        st.success("✅ Face detected and cropped for analysis.")
                    else:
                        st.info("ℹ️ No face detected — using full image.")

            # ── Helper: render one model card (delegates to module-level function)
            def render_model_result(col, model_name, results, accuracy=""):
                render_analysis_card(col, model_name, results, accuracy)

            # ── SINGLE MODEL MODE ──────────────────────────────────────
            if not compare_mode and not quad_mode:
                with st.spinner("⚙️ Loading model & running inference…"):
                    t0 = time.time()
                    pipe, pipe_err = _load_model_from_cfg(model_cfg)
                    if pipe_err:
                        st.toast(
                            "❌ Could not load the selected model. Please try a different model.",
                            icon="❌",
                        )
                        st.stop()
                    results = run_inference(
                        pipe, inference_img, model_cfg["label_fn"], top_k=2
                    )
                    elapsed = time.time() - t0

                # ── Save to history
                try:
                    save_history(
                        filename=safe_filename,
                        pil_img=pil_img,
                        model_name=chosen_model_name,
                        model_id=model_cfg["model_id"],
                        mode="single",
                        verdict=results[0]["label"],
                        confidence=results[0]["confidence"],
                        elapsed=elapsed,
                        all_results=results,
                    )
                except Exception:
                    pass

                top = results[0]
                is_real = top["label"] == "REAL"
                badge_class = "badge-real" if is_real else "badge-fake"
                icon = "✅" if is_real else "⚠️"
                elapsed_s = f"{elapsed:.2f}s"

                st.markdown("---")
                st.markdown(
                    '<p class="section-label">Result</p>', unsafe_allow_html=True
                )
                res_left, res_right = st.columns([1, 1], gap="large")

                with res_left:
                    st.markdown(
                        f'<div style="text-align:center;padding:1.2rem 0;">'
                        f'<div style="font-size:2.8rem;margin-bottom:0.4rem;">{icon}</div>'
                        f'<div class="{badge_class}">{top["label"]}</div>'
                        f'<p style="color:#d1d5db;font-size:0.8rem;margin-top:0.8rem;">'
                        f"Inference in {elapsed_s}</p></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Confidence — {top['confidence']}%**")
                    st.progress(int(top["confidence"]))
                    st.markdown(
                        '<p class="section-label" style="margin-top:1rem;">Top-2 Predictions</p>',
                        unsafe_allow_html=True,
                    )
                    for r in results:
                        icon2 = "🟢" if r["label"] == "REAL" else "🔴"
                        st.markdown(
                            f'<div class="conf-row"><span>{icon2} {r["label"]} '
                            f'<span style="font-size:0.7rem;color:#9ca3af;"> ({r["raw_label"]})</span></span>'
                            f'<span class="conf-val">{r["confidence"]}%</span></div>',
                            unsafe_allow_html=True,
                        )

                with res_right:
                    fig = make_gauge(top["confidence"], top["label"], is_real)
                    st.plotly_chart(
                        fig, use_container_width=True, config={"displayModeBar": False}
                    )

            # ── 2-MODEL COMPARE MODE ───────────────────────────────────
            elif compare_mode:
                cfg_b = MODEL_REGISTRY[compare_model_name]
                with st.spinner("⚙️ Loading both models…"):
                    t0 = time.time()
                    pipe_a, err_a = _load_model_from_cfg(model_cfg)
                    pipe_b, err_b = _load_model_from_cfg(cfg_b)
                    if err_a:
                        st.toast(
                            "❌ Model A failed to load. Please try a different model.",
                            icon="❌",
                        )
                        st.stop()
                    if err_b:
                        st.toast(
                            "❌ Model B failed to load. Please try a different model.",
                            icon="❌",
                        )
                        st.stop()
                    res_a = run_inference(
                        pipe_a, inference_img, model_cfg["label_fn"], top_k=2
                    )
                    res_b = run_inference(
                        pipe_b, inference_img, cfg_b["label_fn"], top_k=2
                    )
                    elapsed = time.time() - t0

                # ── Save to history
                try:
                    save_history(
                        filename=safe_filename,
                        pil_img=pil_img,
                        model_name=f"{chosen_model_name.split('—')[1].strip() if '—' in chosen_model_name else chosen_model_name} vs {compare_model_name.split('—')[1].strip() if '—' in compare_model_name else compare_model_name}",
                        model_id=model_cfg["model_id"],
                        mode="compare",
                        verdict=res_a[0]["label"],
                        confidence=res_a[0]["confidence"],
                        elapsed=elapsed,
                        all_results={"model_a": res_a, "model_b": res_b},
                    )
                except Exception:
                    pass

                st.markdown("---")
                elapsed_str = f"{elapsed:.2f}s"
                st.markdown(
                    f'<p class="section-label">Comparison Results — {elapsed_str} total</p>',
                    unsafe_allow_html=True,
                )
                col_a, col_b = st.columns(2, gap="large")
                render_model_result(
                    col_a,
                    chosen_model_name.split("—")[1].strip(),
                    res_a,
                    model_cfg.get("accuracy", ""),
                )
                render_model_result(
                    col_b,
                    compare_model_name.split("—")[1].strip(),
                    res_b,
                    cfg_b.get("accuracy", ""),
                )

                agree = res_a[0]["label"] == res_b[0]["label"]
                if agree:
                    st.success(f"🤝 Both models agree: **{res_a[0]['label']}**")
                else:
                    st.toast(
                        "⚡ Models disagree — consider the one with higher confidence.",
                        icon="⚠️",
                    )

            # ── 4-MODEL PANEL MODE ─────────────────────────────────────
            elif quad_mode:
                panel_names = [
                    chosen_model_name,
                    quad_model_2,
                    quad_model_3,
                    quad_model_4,
                ]
                panel_cfgs = [MODEL_REGISTRY[n] for n in panel_names]

                with st.spinner("⚙️ Loading 4 models — may take a moment on first run…"):
                    t0 = time.time()
                    panel_pipes = []
                    for cfg in panel_cfgs:
                        p, err = _load_model_from_cfg(cfg)
                        if err:
                            st.toast(
                                "❌ One or more models failed to load. Please try different models.",
                                icon="❌",
                            )
                            st.stop()
                        panel_pipes.append(p)

                    panel_results = [
                        run_inference(pipe, inference_img, cfg["label_fn"], top_k=2)
                        for pipe, cfg in zip(panel_pipes, panel_cfgs)
                    ]
                    elapsed = time.time() - t0

                # ── Save to history (verdict decided after this block)
                _quad_save = dict(
                    filename=safe_filename,
                    pil_img=pil_img,
                    model_name=" | ".join(
                        n.split("—")[1].strip()[:12] if "—" in n else n[:12]
                        for n in panel_names
                    ),
                    model_id=" | ".join(c["model_id"] for c in panel_cfgs),
                    mode="4-panel",
                    elapsed=elapsed,
                    all_results=[
                        {"model": n, "results": r}
                        for n, r in zip(panel_names, panel_results)
                    ],
                )

                # ── 2×2 model card grid ────────────────────────────────
                st.markdown("---")
                elapsed_str = f"{elapsed:.2f}s"
                st.markdown(
                    f'<p class="section-label">4-Model Panel — {elapsed_str} total</p>',
                    unsafe_allow_html=True,
                )

                row1 = st.columns(2, gap="large")
                row2 = st.columns(2, gap="large")
                all_cols = list(row1) + list(row2)

                for i, (col, name, res, cfg) in enumerate(
                    zip(all_cols, panel_names, panel_results, panel_cfgs)
                ):
                    short = name.split("—")[1].strip() if "—" in name else name
                    render_model_result(
                        col, f"M{i + 1}: {short}", res, cfg.get("accuracy", "")
                    )

                # ── OVERALL VERDICT ────────────────────────────────────
                st.markdown("---")

                tops = [r[0] for r in panel_results]
                labels_all = [t["label"] for t in tops]
                confs_all = [t["confidence"] for t in tops]

                fake_votes = labels_all.count("FAKE")
                real_votes = labels_all.count("REAL")

                # Weighted fake confidence across all 4 models
                weighted_fake = (
                    sum(
                        c if l == "FAKE" else (100.0 - c)
                        for l, c in zip(labels_all, confs_all)
                    )
                    / 4.0
                )

                # Verdict: 3+ votes wins; tie → weighted conf decides
                if fake_votes >= 3:
                    final_v = "FAKE"
                elif real_votes >= 3:
                    final_v = "REAL"
                else:
                    final_v = "FAKE" if weighted_fake >= 50 else "REAL"

                # ── Save to history now that verdict is known
                try:
                    _quad_conf = (
                        weighted_fake if final_v == "FAKE" else (100.0 - weighted_fake)
                    )
                    save_history(
                        verdict=final_v,
                        confidence=round(_quad_conf, 2),
                        **_quad_save,
                    )
                except Exception:
                    pass

                v_real = final_v == "REAL"
                v_color = "#34d399" if v_real else "#f87171"
                v_bg = "rgba(52,211,153,0.08)" if v_real else "rgba(248,113,113,0.08)"
                v_border = "#34d399" if v_real else "#f87171"
                v_icon = "✅" if v_real else "🚨"
                v_badge = "badge-real" if v_real else "badge-fake"

                strength = (
                    "Unanimous"
                    if abs(fake_votes - real_votes) == 4
                    else "Strong Consensus"
                    if abs(fake_votes - real_votes) == 3
                    else "Majority"
                    if abs(fake_votes - real_votes) == 2
                    else "Split Decision"
                )
                interp = (
                    "All 4 models detect a deepfake."
                    if fake_votes == 4
                    else "All 4 models confirm authentic."
                    if real_votes == 4
                    else "3 of 4 models say FAKE — likely deepfake."
                    if fake_votes == 3
                    else "3 of 4 models say REAL — likely authentic."
                    if real_votes == 3
                    else "Models are split 2–2. Manual inspection advised."
                )
                vote_dots = ("🟥 " * fake_votes) + ("🟩 " * real_votes)
                wf_str = f"{weighted_fake:.1f}%"

                # Horizontal bar chart
                short_names = [
                    (n.split("—")[1].strip()[:20] if "—" in n else n[:20])
                    for n in panel_names
                ]
                bar_colors = [
                    "#f87171" if l == "FAKE" else "#34d399" for l in labels_all
                ]
                bar_labels = [f"{l}  {c:.1f}%" for l, c in zip(labels_all, confs_all)]

                bar_fig = go.Figure(
                    go.Bar(
                        x=confs_all,
                        y=[f"M{i + 1}: {s}" for i, s in enumerate(short_names)],
                        orientation="h",
                        marker_color=bar_colors,
                        text=bar_labels,
                        textposition="outside",
                        textfont=dict(color="#e5e7eb", size=11, family="Space Mono"),
                    )
                )
                bar_fig.update_layout(
                    height=200,
                    margin=dict(t=10, b=10, l=8, r=90),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        range=[0, 135],
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                    ),
                    yaxis=dict(showgrid=False, tickfont=dict(color="#9ca3af", size=10)),
                    font_color="#e5e7eb",
                )

                # Donut vote chart
                donut_fig = go.Figure(
                    go.Pie(
                        labels=["FAKE", "REAL"],
                        values=[max(fake_votes, 0.01), max(real_votes, 0.01)],
                        hole=0.62,
                        marker_colors=["#f87171", "#34d399"],
                        textinfo="none",
                    )
                )
                ann_color = "#f87171" if fake_votes >= real_votes else "#34d399"
                ann_text = (
                    f"<b>{fake_votes}/4</b><br>FAKE"
                    if fake_votes >= real_votes
                    else f"<b>{real_votes}/4</b><br>REAL"
                )
                donut_fig.update_layout(
                    height=200,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(
                        font=dict(color="#9ca3af", size=10),
                        orientation="h",
                        x=0.05,
                        y=-0.15,
                    ),
                    annotations=[
                        dict(
                            text=ann_text,
                            x=0.5,
                            y=0.5,
                            font_size=13,
                            font_color=ann_color,
                            font_family="Space Mono",
                            showarrow=False,
                        )
                    ],
                )

                vcol_main, vcol_bar, vcol_donut = st.columns(
                    [1.2, 1.8, 1.0], gap="large"
                )

                with vcol_main:
                    st.markdown(
                        f"<div style='border:1px solid {v_border};border-radius:16px;"
                        f"background:{v_bg};padding:1.4rem 1.2rem;text-align:center;"
                        f"box-shadow:0 0 28px {v_border}33;'>"
                        f"<div style='font-size:0.68rem;letter-spacing:0.18em;text-transform:uppercase;"
                        f"color:#d1d5db;margin-bottom:0.5rem;font-family:Space Mono,monospace;'>Overall Verdict</div>"
                        f"<div style='font-size:2.6rem;margin-bottom:0.3rem;'>{v_icon}</div>"
                        f"<div class='{v_badge}' style='font-size:1.3rem;'>{final_v}</div>"
                        f"<div class='card' style='margin: 0.8rem 0; padding: 0.75rem; background: rgba(255,255,255,0.03); border-color: {v_color}40; border-left: 3px solid {v_color}; text-align: left;'>"
                        f"<div style='font-size: 0.75rem; color: #d1d5db; line-height: 1.4;'>"
                        f"{{generate_explanation(final_v, weighted_fake if final_v == 'FAKE' else (100.0 - weighted_fake), 'panel', 4)}}"
                        f"</div></div>"
                        f"<div style='margin-top:0.8rem;font-size:0.78rem;color:{v_color};"
                        f"font-family:Space Mono,monospace;font-weight:700;'>{strength}</div>"
                        f"<div style='font-size:0.72rem;color:#d1d5db;margin-top:0.3rem;'>"
                        f"Weighted fake conf: {wf_str}</div>"
                        f"<div style='margin-top:0.9rem;font-size:1.1rem;letter-spacing:0.12em;'>"
                        f"{vote_dots}</div>"
                        f"<div style='margin-top:0.7rem;font-size:0.72rem;color:#9ca3af;"
                        f"line-height:1.5;padding:0 0.3rem;'>{interp}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with vcol_bar:
                    st.markdown(
                        '<p class="section-label">Per-Model Confidence</p>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        bar_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                with vcol_donut:
                    st.markdown(
                        '<p class="section-label">Vote Split</p>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        donut_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            # ── FULL ENSEMBLE MODE ─────────────────────────────────────
            elif ensemble_mode:
                all_cfgs = list(MODEL_REGISTRY.values())
                all_names = list(MODEL_REGISTRY.keys())

                progress_bar = st.progress(0)
                status_text = st.empty()

                ensemble_results = []
                t0 = time.time()

                for i, (name, cfg) in enumerate(zip(all_names, all_cfgs)):
                    short_name = name.split("—")[1].strip() if "—" in name else name
                    status_text.text(
                        f"⚙️ Running model {i + 1}/{len(all_names)}: {short_name}..."
                    )

                    p, err = _load_model_from_cfg(cfg)

                    if not err and p:
                        res = run_inference(p, inference_img, cfg["label_fn"], top_k=2)
                        ensemble_results.append(
                            {
                                "name": name,
                                "short": short_name,
                                "results": res,
                                "weight": _get_model_weight(cfg.get("accuracy", "N/A")),
                            }
                        )
                    else:
                        # Fallback if model fails to load
                        ensemble_results.append(
                            {
                                "name": name,
                                "short": short_name,
                                "results": [
                                    {
                                        "label": "ERROR",
                                        "confidence": 0,
                                        "raw_label": "error",
                                    }
                                ],
                                "weight": 0,
                            }
                        )

                    progress_bar.progress((i + 1) / len(all_names))

                status_text.empty()
                progress_bar.empty()
                elapsed = time.time() - t0

                # ── ALGORITHM: Weighted Average ─────────────────────────
                total_weight = sum(
                    r["weight"] for r in ensemble_results if r["weight"] > 0
                )
                weighted_fake_score = 0.0

                valid_models = 0
                for r in ensemble_results:
                    if r["weight"] > 0 and r["results"][0]["label"] != "ERROR":
                        valid_models += 1
                        top = r["results"][0]
                        conf = (
                            top["confidence"]
                            if top["label"] == "FAKE"
                            else (100.0 - top["confidence"])
                        )
                        weighted_fake_score += conf * r["weight"]

                if total_weight > 0:
                    weighted_fake_score /= total_weight
                else:
                    weighted_fake_score = 50.0

                final_v = "FAKE" if weighted_fake_score > 50 else "REAL"

                # Save to history
                try:
                    _ens_conf = (
                        weighted_fake_score
                        if final_v == "FAKE"
                        else (100.0 - weighted_fake_score)
                    )
                    save_history(
                        filename=safe_filename,
                        pil_img=pil_img,
                        model_name="Full Ensemble",
                        model_id="all",
                        mode="ensemble",
                        verdict=final_v,
                        confidence=round(_ens_conf, 2),
                        elapsed=elapsed,
                        all_results=ensemble_results,
                    )
                except Exception:
                    pass

                # ── UI RENDERING ─────────────────────────────────────────
                st.markdown("---")
                elapsed_str = f"{elapsed:.2f}s"
                st.markdown(
                    f'<p class="section-label">Full Ensemble ({valid_models} Models) — {elapsed_str} total</p>',
                    unsafe_allow_html=True,
                )

                v_real = final_v == "REAL"
                v_color = "#34d399" if v_real else "#f87171"
                v_bg = "rgba(52,211,153,0.08)" if v_real else "rgba(248,113,113,0.08)"
                v_border = "#34d399" if v_real else "#f87171"
                v_icon = "✅" if v_real else "🚨"
                v_badge = "badge-real" if v_real else "badge-fake"

                _ens_conf = (
                    weighted_fake_score
                    if final_v == "FAKE"
                    else (100.0 - weighted_fake_score)
                )

                # Hero Block
                st.markdown(
                    f"<div style='border:1px solid {v_border};border-radius:16px;"
                    f"background:{v_bg};padding:2.5rem 2rem;text-align:center;"
                    f"box-shadow:0 0 40px {v_border}33; margin-bottom: 2rem;'>"
                    f"<div style='font-size:0.8rem;letter-spacing:0.2em;text-transform:uppercase;"
                    f"color:#d1d5db;margin-bottom:1rem;font-family:Space Mono,monospace;'>Weighted Consensus Verdict</div>"
                    f"<div style='font-size:4rem;margin-bottom:0.5rem;'>{v_icon}</div>"
                    f"<div class='{v_badge}' style='font-size:2rem; padding: 0.5rem 2rem;'>{final_v}</div>"
                    f"<div class='card' style='margin: 1.5rem 0 0.5rem 0; padding: 1rem; background: rgba(255,255,255,0.03); border-color: {v_color}40; border-left: 4px solid {v_color}; text-align: left;'>"
                    f"<div style='font-size: 0.85rem; color: #d1d5db; line-height: 1.5;'>"
                    f"{{generate_explanation(final_v, _ens_conf, 'ensemble', valid_models)}}"
                    f"</div></div>"
                    f"<div style='margin-top:1.5rem;font-size:1.1rem;color:{v_color};"
                    f"font-family:Space Mono,monospace;font-weight:700;'>{_ens_conf:.1f}% Aggregate Confidence</div>"
                    f"<div style='font-size:0.85rem;color:#9ca3af;margin-top:0.5rem;'>"
                    f"Calculated from {valid_models} models using weighted average algorithm.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Grid of individual models
                st.markdown(
                    '<p class="section-label">Individual Model Breakdown</p>',
                    unsafe_allow_html=True,
                )

                # Build HTML for grid
                grid_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1rem;">'

                for r in ensemble_results:
                    if r["results"][0]["label"] == "ERROR":
                        continue

                    top = r["results"][0]
                    is_real = top["label"] == "REAL"
                    m_color = "#34d399" if is_real else "#f87171"
                    m_bg = (
                        "rgba(52,211,153,0.05)" if is_real else "rgba(248,113,113,0.05)"
                    )

                    grid_html += f"""
                    <div style="border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 1rem; background: #18181b;">
                        <div style="font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.5rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{r["short"]}</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-weight: bold; color: {m_color};">{top["label"]}</span>
                            <span style="font-family: Space Mono; color: #d1d5db; font-size: 0.9rem;">{top["confidence"]}%</span>
                        </div>
                        <div style="font-size: 0.65rem; color: #d1d5db; display: flex; justify-content: space-between;">
                            <span>Weight: {r["weight"]:.2f}</span>
                        </div>
                        <div style="width: 100%; height: 4px; background: #9ca3af; border-radius: 2px; margin-top: 0.4rem; overflow: hidden;">
                            <div style="width: {top["confidence"]}%; height: 100%; background: {m_color};"></div>
                        </div>
                    </div>
                    """

                grid_html += "</div>"
                st.markdown(grid_html, unsafe_allow_html=True)

    else:
        # ── Empty state ────────────────────────────────────────────────
        st.markdown(
            """
        <div style='text-align:center;padding:4rem 1rem;'>
            <div style='font-size:4rem;margin-bottom:1rem;'>📁</div>
            <p style='color:#9ca3af;font-size:1rem;'>
                Upload an image using the uploader above to begin analysis.
            </p>
            <p style='color:#d1d5db;font-size:0.82rem;'>
                Supported: JPG · JPEG · PNG · WEBP
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════════



# TAB 2: VIDEO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown('<p class="section-label">Upload Video</p>', unsafe_allow_html=True)

    if not CV2_AVAILABLE:
        st.toast(
            "⚠️ OpenCV is required for video analysis. Install with: `pip install opencv-python-headless`",
            icon="⚠️",
        )
    else:
        video_file = st.file_uploader(
            label="drop a video",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
            help="Supported formats: MP4, AVI, MOV",
            key="vid_uploader",
        )

        if video_file is not None:
            video_bytes, val_err = validate_video_upload(video_file)
            if val_err:
                st.toast(f"⚠️ {val_err}", icon="⚠️")
                st.stop()
            safe_vid_filename = _sanitize_filename(video_file.name)
            st.video(video_bytes)
            st.caption(
                f"📁 {video_file.name}  •  {len(video_bytes) / 1024 / 1024:.1f} MB"
            )

            analyze_vid_btn = st.button(
                "🎬 Analyze Video Frames", use_container_width=True, key="vid_analyze"
            )

            if analyze_vid_btn:
                allowed, wait_sec = check_rate_limit()
                if not allowed:
                    st.toast(
                        f"⏱️ Rate limit reached. Please wait {wait_sec:.1f}s before analyzing again.",
                        icon="⚠️",
                    )
                    st.stop()
                with st.spinner("🎞️ Extracting frames from video…"):
                    frames = extract_video_frames(video_bytes, num_frames=6)

                if not frames:
                    st.toast(
                        "❌ Could not extract frames from this video. The file may be corrupt or in an unsupported codec.",
                        icon="❌",
                    )
                else:
                    st.markdown(
                        f'<p class="section-label">Extracted {len(frames)} Frames</p>',
                        unsafe_allow_html=True,
                    )

                    # Load model once
                    with st.spinner("⚙️ Loading model…"):
                        pipe, pipe_err = _load_model_from_cfg(model_cfg)
                        if pipe_err:
                            st.toast(
                                "❌ Could not load the selected model. Please try a different model.",
                                icon="❌",
                            )
                            st.stop()

                    # Run inference on each frame
                    frame_results = []
                    progress_bar = st.progress(0, text="Analyzing frames…")
                    t0 = time.time()
                    for i, (frame_idx, frame_img) in enumerate(frames):
                        inf_img = frame_img.copy()
                        if use_face_crop:
                            cropped = detect_and_crop_face(frame_img)
                            if cropped is not frame_img:
                                inf_img = cropped
                        res = run_inference(
                            pipe, inf_img, model_cfg["label_fn"], top_k=2
                        )
                        frame_results.append((frame_idx, frame_img, res))
                        progress_bar.progress(
                            (i + 1) / len(frames), text=f"Frame {i + 1}/{len(frames)}"
                        )
                    elapsed = time.time() - t0
                    progress_bar.empty()

                    st.markdown(
                        f"<p style='color:#d1d5db;font-size:0.8rem;'>⏱️ Analyzed {len(frames)} frames in {elapsed:.2f}s</p>",
                        unsafe_allow_html=True,
                    )

                    # ── Frame-by-frame results grid (6 columns) ────────
                    st.markdown(
                        '<p class="section-label">Frame-by-Frame Results</p>',
                        unsafe_allow_html=True,
                    )
                    _VCOLS = 6
                    for row_start in range(0, len(frame_results), _VCOLS):
                        row_chunk = frame_results[row_start : row_start + _VCOLS]
                        cols = st.columns(_VCOLS)
                        for col, (fidx, fimg, fres) in zip(cols, row_chunk):
                            top = fres[0]
                            is_real = top["label"] == "REAL"
                            badge = "badge-real" if is_real else "badge-fake"
                            icon = "✅" if is_real else "⚠️"
                            with col:
                                st.image(fimg, use_container_width=True)
                                st.markdown(
                                    f"<div class='card' style='padding:0.6rem;text-align:center;'>"
                                    f"<div style='font-size:0.6rem;color:#d1d5db;margin-bottom:0.2rem;'>"
                                    f"{fidx:.0f}%</div>"
                                    f"<div>{icon} <span class='{badge}' style='font-size:0.65rem;"
                                    f"padding:0.2rem 0.4rem;'>{top['label']}</span></div>"
                                    f"<div style='font-family:Space Mono,monospace;font-size:0.7rem;"
                                    f"color:#e5e7eb;margin-top:0.2rem;'>{top['confidence']}%</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # ── Overall video verdict ──────────────────────────
                    st.markdown("---")
                    fake_count = sum(
                        1 for _, _, r in frame_results if r[0]["label"] == "FAKE"
                    )

                    if fake_count >= 4:
                        video_verdict = "FAKE"
                        v_conf = (fake_count / len(frames)) * 100
                    elif fake_count <= 2:
                        video_verdict = "REAL"
                        v_conf = ((len(frames) - fake_count) / len(frames)) * 100
                    else:
                        video_verdict = "INCONCLUSIVE"
                        v_conf = 50.0

                    v_is_real = video_verdict == "REAL"

                    st.markdown(
                        '<p class="section-label" style="margin-bottom:1rem;">Overall Video Verdict</p>',
                        unsafe_allow_html=True,
                    )

                    if video_verdict == "INCONCLUSIVE":
                        color = "#fbbf24"
                        pulse = ""
                        glitch = ""
                        expl = "INCONCLUSIVE — Mixed frame results, manual review recommended."
                    else:
                        color = "#46d369" if v_is_real else "#E50914"
                        pulse = "glow-calm" if v_is_real else "glow-danger"
                        glitch = "" if v_is_real else "glitch-text"
                        expl = (
                            f"{fake_count} out of {len(frames)} frames flagged FAKE.<br><br>"
                            + generate_explanation(video_verdict, v_conf, mode="video")
                        )

                    html = f'''
                    <div class="netflix-result-card {pulse}" style="margin: 0 auto; max-width: 500px;">
                        <div class="result-header">
                            <span class="result-model-name">Temporal Consistency Analysis</span>
                            <span class="result-accuracy">{len(frames)} Frames Analyzed</span>
                        </div>
                        
                        <div class="result-verdict-container">
                            <div class="scanning-line"></div>
                            <div class="verdict-text {glitch}" style="color: {color};" data-text="{video_verdict}">{video_verdict}</div>
                        </div>
                        
                        <div class="card" style="margin: 0.5rem 1rem; padding: 0.75rem; background: rgba(255,255,255,0.03); border-color: {color}40; border-left: 3px solid {color}; text-align: left;">
                            <div style="font-size: 0.75rem; color: #d1d5db; line-height: 1.4;">
                                {expl}
                            </div>
                        </div>
                        
                        <div class="result-stats">
                            <div class="stat-row">
                                <span>Verdict Confidence</span>
                                <span style="color: {color}; font-weight: bold;">{v_conf:.0f}%</span>
                            </div>
                            <div class="sleek-progress-bg">
                                <div class="sleek-progress-fill" style="width: {min(100.0, float(v_conf)):.0f}%; background-color: {color};"></div>
                            </div>
                        </div>
                    </div>
                    '''
                    st.markdown(html, unsafe_allow_html=True)

                    # ── Timeline chart ─────────────────────────────────
                    st.markdown(
                        '<p class="section-label" style="margin-top:2rem;">Fake Confidence Timeline</p>',
                        unsafe_allow_html=True,
                    )

                    timeline_x = []
                    timeline_y = []
                    timeline_colors = []
                    for fidx, fimg, fres in frame_results:
                        top = fres[0]
                        timeline_x.append(fidx)
                        if top["label"] == "FAKE":
                            timeline_y.append(top["confidence"])
                        else:
                            timeline_y.append(100.0 - top["confidence"])
                        timeline_colors.append(
                            "#f87171" if top["label"] == "FAKE" else "#34d399"
                        )

                    tl_fig = go.Figure()
                    tl_fig.add_trace(
                        go.Scatter(
                            x=timeline_x,
                            y=timeline_y,
                            mode="lines+markers",
                            line=dict(color="#E50914", width=3, shape="spline"),
                            fill="tozeroy",
                            fillcolor="rgba(229, 9, 20, 0.2)",
                            marker=dict(
                                size=14,
                                color=timeline_colors,
                                line=dict(width=2, color="#ffffff"),
                            ),
                            hovertemplate="<b>Frame: %{x:.1f}%</b><br>Confidence: %{y:.1f}%<extra></extra>",
                        )
                    )
                    tl_fig.add_hline(
                        y=50,
                        line_dash="dash",
                        line_color="#d1d5db",
                        annotation_text="FAKE Threshold (50%)",
                        annotation_position="bottom right",
                    )
                    tl_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(
                            title="Video Position (%)",
                            gridcolor="#9ca3af",
                            showgrid=True,
                            range=[0, 100],
                        ),
                        yaxis=dict(
                            title="Confidence (%)",
                            range=[0, 105],
                            gridcolor="#9ca3af",
                            showgrid=True,
                        ),
                        height=250,
                    )
                    st.plotly_chart(
                        tl_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                    # ── Save to history
                    try:
                        # Use first frame as thumbnail
                        save_history(
                            filename=video_file.name,
                            pil_img=frame_results[0][1],
                            model_name=chosen_model_name,
                            model_id=model_cfg["model_id"],
                            mode="video",
                            verdict=video_verdict,
                            confidence=round(
                                sum(y for y in timeline_y) / len(timeline_y), 2
                            ),
                            elapsed=elapsed,
                            all_results=[
                                {
                                    "frame": fi,
                                    "verdict": r[0]["label"],
                                    "confidence": r[0]["confidence"],
                                }
                                for fi, _, r in frame_results
                            ],
                        )
                    except Exception:
                        pass

        else:
            st.markdown(
                """
            <div style='text-align:center;padding:4rem 1rem;'>
                <div style='font-size:4rem;margin-bottom:1rem;'>🎬</div>
                <p style='color:#9ca3af;font-size:1rem;'>
                    Upload a video to analyze individual frames for deepfake artifacts.
                </p>
                <p style='color:#d1d5db;font-size:0.82rem;'>
                    Supported: MP4 · AVI · MOV
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: AUDIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
with tab_audio:
    st.markdown('<p class="section-label">Upload Audio</p>', unsafe_allow_html=True)

    audio_file = st.file_uploader(
        label="drop an audio file",
        type=["wav", "mp3"],
        label_visibility="collapsed",
        help="Supported formats: WAV, MP3",
        key="audio_uploader",
    )

    if audio_file is not None:
        audio_bytes, val_err = validate_audio_upload(audio_file)
        if val_err:
            st.toast(f"⚠️ {val_err}", icon="⚠️")
            st.stop()
        safe_aud_filename = _sanitize_filename(safe_aud_filename)
        audio_file.seek(0)
        st.audio(audio_bytes, format=f"audio/{audio_file.name.split('.')[-1]}")
        st.caption(f"📁 {audio_file.name}  •  {len(audio_bytes) / 1024:.1f} KB")

        analyze_audio_btn = st.button(
            "🎵 Analyze Audio", use_container_width=True, key="audio_analyze"
        )

        if analyze_audio_btn:
            # ── Load audio model ───────────────────────────────────────
            with st.spinner("⚙️ Loading audio deepfake detection model…"):
                audio_pipe, audio_err = load_audio_pipeline()
            if audio_err:
                st.toast(
                    "❌ Could not load the audio model. Please check your connection and try again.",
                    icon="❌",
                )
                st.stop()

            # ── Save to temp file for pipeline ─────────────────────────
            with st.spinner("🔊 Analyzing audio…"):
                t0 = time.time()
                tmp_audio = tempfile.NamedTemporaryFile(
                    suffix=f".{audio_file.name.split('.')[-1]}", delete=False
                )
                tmp_audio.write(audio_bytes)
                tmp_audio.flush()
                tmp_audio_path = tmp_audio.name
                tmp_audio.close()

                try:
                    raw_results = audio_pipe(tmp_audio_path, top_k=2)
                finally:
                    try:
                        Path(tmp_audio_path).unlink()
                    except Exception:
                        pass
                elapsed = time.time() - t0

            # ── Normalize labels ───────────────────────────────────────
            audio_results = []
            for r in raw_results:
                lbl_raw = r["label"].lower()
                if "fake" in lbl_raw or "spoof" in lbl_raw or "deepfake" in lbl_raw:
                    norm = "FAKE"
                elif "real" in lbl_raw or "bonafide" in lbl_raw or "genuine" in lbl_raw:
                    norm = "REAL"
                else:
                    norm = "FAKE" if "1" in lbl_raw else "REAL"
                audio_results.append(
                    {
                        "label": norm,
                        "raw_label": r["label"],
                        "confidence": round(r["score"] * 100, 2),
                    }
                )
            audio_results.sort(key=lambda x: x["confidence"], reverse=True)

            top = audio_results[0]
            is_real = top["label"] == "REAL"
            badge_class = "badge-real" if is_real else "badge-fake"
            icon = "✅" if is_real else "⚠️"

            st.markdown("---")
            st.markdown(
                '<p class="section-label">Audio Analysis Result</p>',
                unsafe_allow_html=True,
            )
            res_l, res_r = st.columns([1, 1], gap="large")

            with res_l:
                st.markdown(
                    f'<div style="text-align:center;padding:1.2rem 0;">'
                    f'<div style="font-size:2.8rem;margin-bottom:0.4rem;">{icon}</div>'
                    f'<div class="{badge_class}">{top["label"]}</div>'
                    f'<p style="color:#d1d5db;font-size:0.8rem;margin-top:0.8rem;">'
                    f"Inference in {elapsed:.2f}s</p></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Confidence — {top['confidence']}%**")
                st.progress(int(top["confidence"]))
                for r in audio_results:
                    icon2 = "🟢" if r["label"] == "REAL" else "🔴"
                    st.markdown(
                        f'<div class="conf-row"><span>{icon2} {r["label"]} '
                        f'<span style="font-size:0.7rem;color:#9ca3af;"> ({r["raw_label"]})</span></span>'
                        f'<span class="conf-val">{r["confidence"]}%</span></div>',
                        unsafe_allow_html=True,
                    )

            with res_r:
                fig = make_gauge(top["confidence"], top["label"], is_real)
                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )

            # ── Explanation ────────────────────────────────────────────
            _a_expl = (
                generate_explanation(top["label"], top["confidence"], "single")
                .replace("image", "audio")
                .replace("photograph", "recording")
                .replace("skin texture", "voice patterns")
                .replace("edge blending", "spectral artifacts")
                .replace("lighting", "pitch")
                .replace("color", "tonal")
            )
            _a_color = "#34d399" if is_real else "#f87171"
            _a_bg = "rgba(52,211,153,0.06)" if is_real else "rgba(248,113,113,0.06)"
            _a_border = "rgba(52,211,153,0.25)" if is_real else "rgba(248,113,113,0.25)"
            st.markdown(
                f"<div class='card' style='border-color:{_a_border};background:{_a_bg};'>"
                f"<p class='section-label' style='margin-bottom:0.4rem;'>🧠 Analysis Explanation</p>"
                f"<div style='color:#d1d5db;font-size:0.82rem;line-height:1.7;'>"
                f"{_a_expl}</div></div>",
                unsafe_allow_html=True,
            )

            # ── Waveform visualization ─────────────────────────────────
            if LIBROSA_AVAILABLE:
                st.markdown(
                    '<p class="section-label" style="margin-top:1rem;">Waveform</p>',
                    unsafe_allow_html=True,
                )
                try:
                    # Re-write temp file for librosa
                    tmp_audio2 = tempfile.NamedTemporaryFile(
                        suffix=f".{audio_file.name.split('.')[-1]}", delete=False
                    )
                    tmp_audio2.write(audio_bytes)
                    tmp_audio2.flush()
                    tmp_audio2_path = tmp_audio2.name
                    tmp_audio2.close()

                    y, sr = librosa.load(tmp_audio2_path, sr=None, mono=True)
                    Path(tmp_audio2_path).unlink(missing_ok=True)

                    # Downsample for plotting (max 5000 points)
                    max_pts = 5000
                    if len(y) > max_pts:
                        step = len(y) // max_pts
                        y_plot = y[::step]
                    else:
                        y_plot = y
                    t_axis = [
                        i / sr for i in range(0, len(y), max(1, len(y) // len(y_plot)))
                    ]
                    t_axis = t_axis[: len(y_plot)]

                    wf_fig = go.Figure()
                    wf_fig.add_trace(
                        go.Scatter(
                            x=t_axis,
                            y=y_plot,
                            mode="lines",
                            line=dict(color="#7b61ff", width=1),
                            fill="tozeroy",
                            fillcolor="rgba(123,97,255,0.15)",
                            hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.4f}<extra></extra>",
                        )
                    )
                    wf_fig.update_layout(
                        height=200,
                        margin=dict(t=10, b=30, l=50, r=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(
                            title="Time (s)",
                            showgrid=False,
                            tickfont=dict(color="#9ca3af"),
                        ),
                        yaxis=dict(
                            title="Amplitude",
                            showgrid=True,
                            gridcolor="rgba(255,255,255,0.05)",
                            tickfont=dict(color="#9ca3af"),
                        ),
                        font_color="#e5e7eb",
                    )
                    st.plotly_chart(
                        wf_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                except Exception as e:
                    _sec_logger.warning("Waveform render failed: %s", e)
                    st.caption("⚠️ Could not render waveform visualization.")
            else:
                st.caption(
                    "💡 Install `librosa` for waveform visualization: `pip install librosa`"
                )

            # ── Save to history ────────────────────────────────────────
            try:
                # Create a simple placeholder image for audio history
                _aud_thumb = Image.new("RGB", (100, 100), color=(30, 30, 50))
                save_history(
                    filename=audio_file.name,
                    pil_img=_aud_thumb,
                    model_name=f"Audio: {_AUDIO_MODEL_ID.split('/')[-1]}",
                    model_id=_AUDIO_MODEL_ID,
                    mode="audio",
                    verdict=top["label"],
                    confidence=top["confidence"],
                    elapsed=elapsed,
                    all_results=audio_results,
                )
            except Exception:
                pass

    else:
        st.markdown(
            """
        <div style='text-align:center;padding:4rem 1rem;'>
            <div style='font-size:4rem;margin-bottom:1rem;'>🎵</div>
            <p style='color:#9ca3af;font-size:1rem;'>
                Upload an audio file to detect voice deepfakes.
            </p>
            <p style='color:#d1d5db;font-size:0.82rem;'>
                Supported: WAV · MP3
            </p>
            <p style='color:#d1d5db;font-size:0.72rem;margin-top:0.5rem;'>
                Model: MelodyMachine/Deepfake-audio-detection-V2
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: URL SCANNER
# ═══════════════════════════════════════════════════════════════════════
with tab_url:
    st.markdown(
        """
    <div style="background:#181818; padding:3rem 2rem; border-radius:12px; border:1px solid #333; text-align:center; margin-top:1rem;">
        <h2 style="font-family:'Bebas Neue', cursive; font-size:3rem; color:#fff; margin-bottom:0.5rem; letter-spacing:2px;">URL Multimedia Authenticity Scanner</h2>
        <p style="color:#9ca3af; font-size:1.1rem; max-width:600px; margin:0 auto 2rem;">Paste a URL from Twitter, YouTube, or any news site. Our system will extract the media and run the 5-model ensemble on it.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    url_input = st.text_input("Media URL", placeholder="https://twitter.com/...")

    if st.button("🔗 Scan URL", use_container_width=True):
        if not url_input:
            st.toast("Please enter a URL first.", icon="⚠️")
        elif not url_input.startswith("http"):
            st.toast("Invalid URL format. Must start with http/https.", icon="⚠️")
        else:
            import requests
            import io
            from PIL import Image
            
            allowed, wait_sec = check_rate_limit()
            if not allowed:
                st.toast(f"⏱️ Rate limit reached. Please wait {wait_sec:.1f}s before analyzing again.", icon="⚠️")
            else:
                with st.spinner("🌐 Extracting media from URL..."):
                    try:
                        resp = requests.get(url_input, timeout=10)
                        resp.raise_for_status()
                        content_type = resp.headers.get("Content-Type", "")
                        if not content_type.startswith("image/"):
                            st.error(f"❌ URL does not point to a direct image (Content-Type: {content_type}). Please provide a direct image link.")
                            st.stop()
                        
                        image_bytes = resp.content
                        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    except Exception as e:
                        st.error(f"❌ Failed to fetch image: {e}")
                        st.stop()
                        
                with st.spinner(f"🔬 Running {chosen_model_name}..."):
                    model_cfg = MODEL_REGISTRY[chosen_model_name]
                    pipe, pipe_err = _load_model_from_cfg(model_cfg)
                    if pipe_err:
                        st.error("❌ Failed to load model.")
                        st.stop()
                    
                    t0 = time.time()
                    res = run_inference(pipe, pil_img, model_cfg["label_fn"], top_k=2)
                    elapsed = time.time() - t0
                
                st.success(f"Analysis Complete in {elapsed:.2f}s!")
                top = res[0]
                is_real = top["label"] == "REAL"
                badge_class = "badge-real" if is_real else "badge-fake"
                icon = "✅" if is_real else "🚨"
                verdict_text = "AUTHENTIC MEDIA" if is_real else "SYNTHETIC MEDIA DETECTED"
                border_color = "#34d399" if is_real else "#f87171"
                bg_color = "rgba(52,211,153,0.08)" if is_real else "rgba(248,113,113,0.08)"

                st.image(pil_img, caption="Analyzed Image", use_container_width=True)
                st.markdown(
                    f"""
                <div style='border:1px solid {border_color}; border-radius:16px; background:{bg_color}; padding:1.4rem; text-align:center; box-shadow:0 0 28px {border_color}33; margin-top:1rem;'>
                    <div style='font-size:0.68rem; letter-spacing:0.18em; text-transform:uppercase; color:#d1d5db; margin-bottom:0.5rem; font-family:Space Mono,monospace;'>URL Analysis Verdict</div>
                    <div style='font-size:2.6rem; margin-bottom:0.3rem;'>{icon}</div>
                    <div class='{badge_class}' style='font-size:1.3rem;'>{verdict_text} ({top['confidence']:.1f}%)</div>
                    <div style='margin-top:0.6rem; font-size:0.78rem; color:#9ca3af;'>Model: {chosen_model_name}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # render_forensics_expander("video")  # Not implemented for video

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: HISTORY
# ═══════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown(
        '<p class="sub-title">Last 200 Analyzed Items · SQLite Database</p>',
        unsafe_allow_html=True,
    )

    _hrows = get_history()
    _htotal = len(_hrows)

    if _htotal > 0:
        _hfake = sum(1 for r in _hrows if r["verdict"] == "FAKE")
        _hreal = _htotal - _hfake
        _havg = round(sum(r["confidence"] for r in _hrows) / _htotal, 1)
        from collections import Counter as _Ctr

        _hmodel = _Ctr(r["model_name"] for r in _hrows).most_common(1)[0][0]
        _hms = _hmodel.split("—")[1].strip()[:18] if "—" in _hmodel else _hmodel[:18]

        sc1, sc2, sc3, sc4 = st.columns(4)
        for _col, _num, _lbl, _col_css in [
            (sc1, _htotal, "Total Tested", "#a78bfa"),
            (sc2, _hfake, "Fakes Found", "#f87171"),
            (sc3, _hreal, "Real Confirmed", "#34d399"),
            (sc4, f"{_havg}%", "Avg Confidence", "#00d4ff"),
        ]:
            _col.markdown(
                f"<div class='stat-box'>"
                f"<div class='stat-num' style='color:{_col_css};'>{_num}</div>"
                f"<div class='stat-label'>{_lbl}</div></div>",
                unsafe_allow_html=True,
            )

        st.caption(f"🤖 Most used model: **{_hms}**")
        st.markdown("---")

        # ── Filters
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            _hvf = st.selectbox("Verdict", ["All", "REAL", "FAKE"], key="hvf")
        with fc2:
            _hmf = st.selectbox(
                "Mode",
                ["All", "single", "compare", "4-panel", "video", "audio"],
                key="hmf",
            )
        with fc3:
            _hsr = st.text_input("Search filename", placeholder="filter…", key="hsr")

        _hfilt = _hrows
        if _hvf != "All":
            _hfilt = [r for r in _hfilt if r["verdict"] == _hvf]
        if _hmf != "All":
            _hfilt = [r for r in _hfilt if r["mode"] == _hmf]
        if _hsr:
            _hfilt = [
                r for r in _hfilt if _hsr.lower() in (r["filename"] or "").lower()
            ]

        # ── Actions
        ac1, ac2, _ = st.columns([1, 1.3, 3])
        with ac1:
            if st.button("🗑️ Clear All", key="hca"):
                clear_history()
                st.rerun()
        with ac2:
            st.download_button(
                "📥 Export CSV",
                _export_csv(_hrows),
                "deepfake_history.csv",
                "text/csv",
                key="hxp",
            )

        st.markdown(
            f"<p style='color:#9ca3af;font-size:0.8rem;'>"
            f"Showing <b>{len(_hfilt)}</b> of {_htotal} records "
            f"(max {_MAX_HISTORY})</p>",
            unsafe_allow_html=True,
        )

        # ── Card grid (4 per row)
        _NC = 4
        for _i in range(0, len(_hfilt), _NC):
            _chunk = _hfilt[_i : _i + _NC]
            _gcols = st.columns(_NC)
            for _gc, _row in zip(_gcols, _chunk):
                with _gc:
                    if _row.get("thumbnail"):
                        try:
                            st.image(
                                base64.b64decode(_row["thumbnail"]),
                                use_container_width=True,
                            )
                        except Exception:
                            pass
                    _v = _row.get("verdict") or "?"
                    _bc = "hv-real" if _v == "REAL" else "hv-fake"
                    _ico = "✅" if _v == "REAL" else "⚠️"
                    _mi = {
                        "single": "🔬",
                        "compare": "⚖️",
                        "4-panel": "🧪",
                        "video": "🎬",
                        "audio": "🎵",
                    }.get(_row.get("mode", ""), "🔬")
                    _mn = _row.get("model_name", "")
                    _ms2 = _mn.split("—")[-1].strip()[:16] if "—" in _mn else _mn[:16]
                    st.markdown(
                        f"<div class='hist-card' tabindex='0' aria-label='History Record'>"
                        f"<span class='{_bc}'>{_ico} {_v}</span>"
                        f"<div style='font-size:0.7rem;color:#e5e7eb;margin-top:0.3rem;"
                        f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
                        f"📁 {(_row.get('filename') or 'unknown')[:18]}</div>"
                        f"<div style='font-size:0.67rem;color:#d1d5db;margin-top:0.15rem;'>"
                        f"{_mi} {_row.get('mode', '')} · {_row.get('confidence', 0)}%</div>"
                        f"<div style='font-size:0.65rem;color:#9ca3af;margin-top:0.1rem;"
                        f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
                        f"🤖 {_ms2}</div>"
                        f"<div style='font-size:0.63rem;color:#9ca3af;margin-top:0.1rem;'>"
                        f"🕒 {_row.get('timestamp', '')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            """
        <div style='text-align:center;padding:4rem 1rem;'>
            <div style='font-size:4rem;margin-bottom:1rem;'>📂</div>
            <p style='color:#9ca3af;font-size:1rem;'>No history yet.</p>
            <p style='color:#d1d5db;font-size:0.85rem;'>Analyze some images, videos, or audio first, then come back here.</p>
        </div>""",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
<style>
/* Hide default Streamlit footer and main menu */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
/* header {visibility: hidden;} */
</style>

<div style='text-align:center;padding:1.5rem 0 0.5rem;'>
    <span style='font-family:Space Mono,monospace;font-size:0.7rem;
                 color:#6b7280;letter-spacing:0.15em;'>
        MULTIMEDIA AUTHENTICITY LAB · IMAGE · VIDEO · AUDIO · CNN-BASED · HUGGING FACE<br>
        <span style="font-size: 0.65rem; font-weight: 500; letter-spacing: 0.05em; margin-top: 10px; display: inline-block;">© 2026 TheSweetDuo</span>
    </span>
</div>
""",
    unsafe_allow_html=True,
)
