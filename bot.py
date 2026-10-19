#!/usr/bin/env python3
"""
RepDuel: Telegram-бот, где ИИ-судья считает отжимания по видео.

Что умеет:
  * считать отжимания по видео (MediaPipe Pose, вид сбоку);
  * дуэли с друзьями по ссылке;
  * рейтинг недели и за всё время;
  * уровни, достижения, скины карточек;
  * красивую карточку результата (картинка 4:5 для TikTok/Reels/Stories);
  * русский и английский язык (по языку Telegram пользователя).

Всё меню (профиль, режимы, дуэли, рейтинг, скины) живёт в Mini App, бот отдаёт ему данные через API.
Для приложения положи cloudflared.exe рядом с bot.py: бот сам откроет туннель и возьмёт адрес.
Запуск:   python bot.py
Самотест: python bot.py --selftest   (без Telegram и без видео)
Токен:    переменная окружения BOT_TOKEN (выдаёт @BotFather)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import re
import io
import logging
import math
import os
import random
import shutil
import sqlite3
import string
import sys
import tempfile
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("REPDUEL_DB", str(BASE_DIR / "repduel.db"))
MODEL_DIR = BASE_DIR / "models"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

MAX_VIDEO_SECONDS = 120          # длиннее не принимаем
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024   # лимит Telegram Bot API на скачивание
MAX_CONCURRENT = 2               # сколько видео анализируем одновременно
ANALYSIS_FPS = 15                # на скольких кадрах в секунду считаем
FRAME_WIDTH = 480                # кадр уменьшаем для скорости

log = logging.getLogger("repduel")


# ════════════════════════════════════════════════════════════════
#  1. СЧЁТЧИК ПОВТОРЕНИЙ
# ════════════════════════════════════════════════════════════════

class VideoError(Exception):
    """code: open | too_long | no_pose"""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class RepResult:
    reps: int
    partial: int
    duration: float
    valid_ratio: float
    samples: int


def joint_angle(a, b, c):
    """Угол в точке b между отрезками b→a и b→c, в градусах."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n < 1e-9:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


class RepCounter:
    """
    Автомат по углу сустава.
    Полный повтор: угол упал до down и вернулся выше up (и, если нужно, в нижней
    точке выполнено доп. условие: например подбородок выше перекладины).
    Неполный: угол ушёл ниже descent и вернулся выше up, но условие не выполнено.
    """

    def __init__(self, down, descent, up, gap, need_extra=False):
        self.down, self.descent, self.up, self.gap = down, descent, up, gap
        self.need_extra = need_extra
        self.state = "up"
        self.min_angle = 180.0
        self.bottom_ok = False
        self.reps = 0
        self.partial = 0
        self.last_rep_t = -10.0

    def update(self, angle: float, t: float, extra_ok: bool = True):
        if self.state == "up":
            if angle < self.descent:
                self.state = "down"
                self.min_angle = angle
                self.bottom_ok = angle <= self.down and extra_ok
        else:
            self.min_angle = min(self.min_angle, angle)
            if angle <= self.down and extra_ok:
                self.bottom_ok = True
            if angle > self.up:
                if t - self.last_rep_t >= self.gap:
                    deep = self.min_angle <= self.down
                    if deep and (self.bottom_ok or not self.need_extra):
                        self.reps += 1
                    else:
                        self.partial += 1
                    self.last_rep_t = t
                self.state = "up"


# индексы точек тела в MediaPipe Pose
NOSE = 0
SIDES = {
    "L": dict(sh=11, el=13, wr=15, hip=23, kn=25, an=27),
    "R": dict(sh=12, el=14, wr=16, hip=24, kn=26, an=28),
}

# joint: какие три точки задают угол; orient: поза тела в кадре;
# down/descent/up: пороги угла; gap: минимум секунд между повторениями;
# chin: нужно ли, чтобы нос был выше кистей в верхней точке (подтягивания)
EXERCISES = {
    "pushup": dict(icon="💪", name=("Отжимания", "Push-ups"), card="PUSH-UPS",
                   joint=("sh", "el", "wr"), orient="horizontal",
                   down=100.0, descent=135.0, up=150.0, gap=0.35, chin=False),
    "squat":  dict(icon="🦵", name=("Приседания", "Squats"), card="SQUATS",
                   joint=("hip", "kn", "an"), orient="vertical",
                   down=100.0, descent=140.0, up=160.0, gap=0.70, chin=False),
    "pullup": dict(icon="🧗", name=("Подтягивания", "Pull-ups"), card="PULL-UPS",
                   joint=("sh", "el", "wr"), orient="vertical",
                   down=75.0, descent=120.0, up=155.0, gap=0.80, chin=True),
}


def ex_name(ex, lang):
    return EXERCISES[ex]["name"][0 if lang == "ru" else 1]


class PoseBackend:
    """
    Достаёт 33 точки тела из RGB-кадра: список (x, y, visibility) или None.
    Сначала пробуем «старый» API mediapipe.solutions (модель внутри пакета,
    версия mediapipe==0.10.14). Если его нет, берём Tasks API и один раз
    скачиваем модель pose_landmarker_lite.task.
    """

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.mode = None
        self._ts = 0
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mode = "legacy"
        else:
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
            model = self._ensure_model()
            opts = vision.PoseLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.VIDEO, num_poses=1)
            self.pose = vision.PoseLandmarker.create_from_options(opts)
            self.mode = "tasks"

    @staticmethod
    def _ensure_model() -> Path:
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / "pose_landmarker_lite.task"
        if not path.exists() or path.stat().st_size < 1_000_000:
            log.info("Скачиваю модель позы (один раз)...")
            urllib.request.urlretrieve(MODEL_URL, path)
        return path

    def landmarks(self, rgb, dt_ms: int):
        if self.mode == "legacy":
            res = self.pose.process(rgb)
            if not res.pose_landmarks:
                return None
            return [(p.x, p.y, p.visibility) for p in res.pose_landmarks.landmark]
        self._ts += max(1, dt_ms)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.pose.detect_for_video(img, self._ts)
        if not res.pose_landmarks:
            return None
        return [(p.x, p.y, p.visibility) for p in res.pose_landmarks[0]]

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass


def count_from_landmarks(frames, fps: float, duration: float, exercise: str = "pushup") -> RepResult:
    """
    frames: список (t_секунды, landmarks | None, ширина, высота) уже после
    выборки по частоте. Вынесено отдельно, чтобы тестировать без видео.
    """
    cfg = EXERCISES[exercise]
    counter = RepCounter(cfg["down"], cfg["descent"], cfg["up"], cfg["gap"], cfg["chin"])
    window = deque(maxlen=3)
    valid = 0
    for t, lm, w, h in frames:
        if lm is None:
            continue
        best = None
        for side in SIDES.values():
            idx = [side[k] for k in cfg["joint"]]
            vis = min(lm[i][2] for i in idx)
            if best is None or vis > best[0]:
                best = (vis, side, idx)
        vis, side, idx = best
        if vis < 0.5:
            continue
        px = lambda i: (lm[i][0] * w, lm[i][1] * h)
        # поза тела: 0° = горизонтально (отжимания), 90° = вертикально (присед, турник)
        far = side["an"] if lm[side["an"]][2] >= 0.3 else side["hip"]
        dx = abs(px(side["sh"])[0] - px(far)[0])
        dy = abs(px(side["sh"])[1] - px(far)[1])
        tilt = math.degrees(math.atan2(dy, max(dx, 1e-6)))
        if cfg["orient"] == "horizontal" and tilt > 40:
            continue
        if cfg["orient"] == "vertical" and tilt < 50:
            continue
        ang = joint_angle(px(idx[0]), px(idx[1]), px(idx[2]))
        if ang is None:
            continue
        extra_ok = True
        if cfg["chin"] and lm[NOSE][2] >= 0.3:
            extra_ok = px(NOSE)[1] < px(side["wr"])[1]   # нос выше кисти
        window.append(ang)
        valid += 1
        counter.update(sum(window) / len(window), t, extra_ok)
    sampled = max(1, len(frames))
    ratio = valid / sampled
    if valid < 10 or ratio < 0.25:
        raise VideoError("no_pose")
    return RepResult(counter.reps, counter.partial, duration, ratio, sampled)


def analyze_video(path: str, exercise: str = "pushup", backend: PoseBackend | None = None) -> RepResult:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("open")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 240:
        fps = 30.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = nframes / fps if nframes > 0 else 0.0
    if duration > MAX_VIDEO_SECONDS + 1:
        cap.release()
        raise VideoError("too_long")
    step = max(1, round(fps / ANALYSIS_FPS))
    own_backend = backend is None
    if own_backend:
        backend = PoseBackend()
    frames = []
    last_t = 0.0
    try:
        i = 0
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    if w > FRAME_WIDTH:
                        scale = FRAME_WIDTH / w
                        frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))
                        h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    t = i / fps
                    lm = backend.landmarks(rgb, int(1000 * step / fps))
                    frames.append((t, lm, w, h))
                    last_t = t
            i += 1
            if i / fps > MAX_VIDEO_SECONDS + 1:
                raise VideoError("too_long")
    finally:
        cap.release()
        if own_backend:
            backend.close()
    return count_from_landmarks(frames, fps, duration or last_t, exercise)


# ════════════════════════════════════════════════════════════════
#  2. БАЗА ДАННЫХ
# ════════════════════════════════════════════════════════════════

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def q(sql, args=(), one=False):
    c = _conn()
    try:
        rows = c.execute(sql, args).fetchall()
    finally:
        c.close()
    if one:
        return rows[0] if rows else None
    return rows


def x(sql, args=()):
    c = _conn()
    try:
        cur = c.execute(sql, args)
        c.commit()
        return cur.rowcount
    finally:
        c.close()


def init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY, name TEXT, username TEXT, lang TEXT,
        skin TEXT DEFAULT 'classic', created REAL);
    CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, reps INTEGER,
        partial INTEGER, duration REAL, ts REAL, duel TEXT, ex TEXT DEFAULT 'pushup');
    CREATE TABLE IF NOT EXISTS duels(
        code TEXT PRIMARY KEY, creator INTEGER, opponent INTEGER,
        c_reps INTEGER, o_reps INTEGER, status TEXT, created REAL, ex TEXT DEFAULT 'pushup');
    CREATE TABLE IF NOT EXISTS ach(
        user_id INTEGER, key TEXT, ts REAL, PRIMARY KEY(user_id, key));
    """)
    for tbl in ("results", "duels"):   # миграция старых баз
        cols = [r[1] for r in c.execute(f"PRAGMA table_info({tbl})")]
        if "ex" not in cols:
            c.execute(f"ALTER TABLE {tbl} ADD COLUMN ex TEXT DEFAULT 'pushup'")
    c.commit()
    c.close()


def lang_of(code):
    return "ru" if (code or "").lower()[:2] in ("ru", "uk", "be", "kk") else "en"


def ensure_user(tg_user) -> str:
    lang = lang_of(tg_user.language_code)
    name = tg_user.full_name or tg_user.username or str(tg_user.id)
    x("""INSERT INTO users(id,name,username,lang,created) VALUES(?,?,?,?,?)
         ON CONFLICT(id) DO UPDATE SET name=excluded.name,
         username=excluded.username, lang=excluded.lang""",
      (tg_user.id, name, tg_user.username, lang, time.time()))
    return lang


def user_stats(uid):
    r = q("""SELECT COALESCE(SUM(reps),0) total, COALESCE(MAX(reps),0) best,
             COUNT(*) sets FROM results WHERE user_id=?""", (uid,), one=True)
    return int(r["total"]), int(r["best"]), int(r["sets"])


def user_best(uid, ex) -> int:
    r = q("SELECT COALESCE(MAX(reps),0) b FROM results WHERE user_id=? AND ex=?", (uid, ex), one=True)
    return int(r["b"])


def level_info(total: int):
    lvl = int(math.sqrt(total / 10)) + 1
    lo, hi = 10 * (lvl - 1) ** 2, 10 * lvl ** 2
    return lvl, (total - lo) / (hi - lo)


# ════════════════════════════════════════════════════════════════
#  3. ИГРОВАЯ ЧАСТЬ: СКИНЫ И ДОСТИЖЕНИЯ
# ════════════════════════════════════════════════════════════════

SKINS = {
    "classic": dict(unlock=0, name=("Классика", "Classic"),
                    c1=(22, 26, 52), c2=(58, 92, 176), accent=(120, 205, 255)),
    "neon":    dict(unlock=50, name=("Неон", "Neon"),
                    c1=(10, 6, 30), c2=(120, 20, 170), accent=(0, 255, 200)),
    "gold":    dict(unlock=200, name=("Золото", "Gold"),
                    c1=(26, 18, 4), c2=(150, 108, 18), accent=(255, 214, 90)),
    "inferno": dict(unlock=500, name=("Инферно", "Inferno"),
                    c1=(28, 4, 4), c2=(190, 32, 10), accent=(255, 170, 60)),
}

ACH = {
    "first":    ("🎬", "Первый подход", "First set"),
    "set10":    ("🔟", "10 за подход", "10 in one set"),
    "set25":    ("💪", "25 за подход", "25 in one set"),
    "set50":    ("🔥", "50 за подход", "50 in one set"),
    "total100": ("💯", "100 повторений всего", "100 reps total"),
    "total500": ("🚀", "500 повторений всего", "500 reps total"),
    "duel_win": ("⚔️", "Первая победа в дуэли", "First duel win"),
    "duel3":    ("🏆", "3 победы в дуэлях", "3 duel wins"),
}


def unlock_achievements(uid, reps, total, duel_wins):
    wanted = ["first"]
    if reps >= 10: wanted.append("set10")
    if reps >= 25: wanted.append("set25")
    if reps >= 50: wanted.append("set50")
    if total >= 100: wanted.append("total100")
    if total >= 500: wanted.append("total500")
    if duel_wins >= 1: wanted.append("duel_win")
    if duel_wins >= 3: wanted.append("duel3")
    new = []
    for key in wanted:
        if x("INSERT OR IGNORE INTO ach(user_id,key,ts) VALUES(?,?,?)",
             (uid, key, time.time())):
            new.append(key)
    return new


def duel_wins(uid) -> int:
    r = q("""SELECT COUNT(*) n FROM duels WHERE status='done' AND
             ((creator=? AND c_reps>o_reps) OR (opponent=? AND o_reps>c_reps))""",
          (uid, uid), one=True)
    return int(r["n"])


# ════════════════════════════════════════════════════════════════
#  4. КАРТОЧКА РЕЗУЛЬТАТА (картинка 1080x1350)
# ════════════════════════════════════════════════════════════════

_FONT_BOLD = ["fonts/Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
              "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_font_cache: dict = {}
HAS_TTF = False

_TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                     ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l",
                      "m", "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch",
                      "sh", "shch", "", "y", "", "e", "yu", "ya"]))


def _font(size):
    global HAS_TTF
    if size in _font_cache:
        return _font_cache[size]
    f = None
    for cand in _FONT_BOLD:
        try:
            f = ImageFont.truetype(str(BASE_DIR / cand) if cand.startswith("fonts/") else cand, size)
            HAS_TTF = True
            break
        except Exception:
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size)
        except TypeError:
            f = ImageFont.load_default()
    _font_cache[size] = f
    return f


def _safe_name(name: str) -> str:
    _font(40)
    if HAS_TTF:
        return name
    out = "".join(_TRANSLIT.get(ch.lower(), ch) for ch in name)
    return "".join(ch for ch in out if ord(ch) < 128).strip() or "Athlete"


def _spaced(draw, xy, text, font, fill, spacing, align="center"):
    """Текст с разрядкой. align: center (xy = центр) или left (xy = левый край)."""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x0 = xy[0] - total / 2 if align == "center" else xy[0]
    for ch, w in zip(text, widths):
        draw.text((x0, xy[1]), ch, font=font, fill=fill, anchor="lm")
        x0 += w + spacing


def render_card(name, reps, duration, quality, level, skin_id, bot_username="", ex="pushup") -> io.BytesIO:
    W, H = 1080, 1350
    sk = SKINS.get(skin_id, SKINS["classic"])
    c1, c2, ac = sk["c1"], sk["c2"], sk["accent"]

    img = Image.new("RGB", (W, H), c1)
    d = ImageDraw.Draw(img)
    for y in range(H):
        k = y / (H - 1)
        d.line([(0, y), (W, y)], fill=tuple(int(c1[i] + (c2[i] - c1[i]) * k) for i in range(3)))
    img = img.convert("RGBA")

    deco = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dd = ImageDraw.Draw(deco)
    dd.ellipse((-300, -300, 620, 620), fill=ac + (34,))
    dd.ellipse((560, 800, 1460, 1700), fill=ac + (28,))
    for i in range(7):
        x0 = 120 + i * 150
        dd.polygon([(x0, 1350), (x0 + 60, 1350), (x0 + 360, 0), (x0 + 300, 0)], fill=(255, 255, 255, 7))
    img = Image.alpha_composite(img, deco)

    # свечение за числом
    big = _font(560 if reps < 100 else 430)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((W / 2, 560), str(reps), font=big, fill=ac + (210,), anchor="mm")
    glow = glow.filter(ImageFilter.GaussianBlur(34))
    img = Image.alpha_composite(img, glow)

    d = ImageDraw.Draw(img)
    # шапка
    d.rounded_rectangle((60, 78, 74, 122), radius=7, fill=ac)
    _spaced(d, (96, 100), "REPDUEL", _font(44), (255, 255, 255), 8, align="left")
    # число
    d.text((W / 2 + 6, 566), str(reps), font=big, fill=(0, 0, 0, 90), anchor="mm")
    d.text((W / 2, 560), str(reps), font=big, fill=(255, 255, 255), anchor="mm")
    _spaced(d, (W / 2, 860), EXERCISES[ex]["card"], _font(64), ac, 14)

    # три плашки
    chips = [("TIME", f"{duration:.0f}s"), ("FORM", f"{quality}%"), ("LEVEL", str(level))]
    chip_w, gap = 300, 30
    x0 = (W - (3 * chip_w + 2 * gap)) / 2
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for i, _ in enumerate(chips):
        a = x0 + i * (chip_w + gap)
        ld.rounded_rectangle((a, 940, a + chip_w, 1100), radius=32,
                             fill=(255, 255, 255, 26), outline=ac + (150,), width=3)
    img = Image.alpha_composite(img, layer)
    d = ImageDraw.Draw(img)
    for i, (label, value) in enumerate(chips):
        cx = x0 + i * (chip_w + gap) + chip_w / 2
        d.text((cx, 980), label, font=_font(30), fill=ac, anchor="mm")
        d.text((cx, 1043), value, font=_font(64), fill=(255, 255, 255), anchor="mm")

    # низ
    d.text((W / 2, 1170), _safe_name(name)[:22], font=_font(60), fill=(255, 255, 255), anchor="mm")
    _spaced(d, (W / 2, 1235), "AI-JUDGED DUELS", _font(32), (255, 255, 255), 5)
    if bot_username:
        d.text((W / 2, 1290), "@" + bot_username, font=_font(36), fill=ac, anchor="mm")

    out = io.BytesIO()
    img.convert("RGB").save(out, "PNG", optimize=True)
    out.seek(0)
    out.name = "repduel.png"
    return out


# ════════════════════════════════════════════════════════════════
#  5. ТЕКСТЫ (ru / en)
# ════════════════════════════════════════════════════════════════

T = {
    "ru": {
        "welcome": ("🏋️ <b>RepDuel</b>: упражнения с ИИ-судьёй\n\n"
                    "Выбери упражнение и нажми «📹 Считать вживую»: камера откроется прямо здесь. Я считаю повторения честно, "
                    "без «поверь на слово».\n\n"
                    "💪🦵🧗 отжимания, приседания, подтягивания вживую\n"
                    "⚔️ дуэли с друзьями по ссылке\n"
                    "🏆 рейтинг недели\n"
                    "🎨 скины карточек за повторения\n\n"
                    "Выбери действие 👇"),
        "howto": ("🎥 <b>Как снимать</b>\n"
                  "• телефон сбоку, на уровне пола, метров 2 от тебя\n"
                  "• в кадре всё тело, от рук до ног\n"
                  "• хороший свет, других людей в кадре нет\n"
                  "• видео до {max} секунд, файл до 20 МБ\n\n"
                  "<b>Полный повтор</b>: локти согнуты примерно до 90°, затем полностью "
                  "выпрямлены. Неполные не засчитываются."),
        "howto_squat": ("🎥 <b>Как снимать приседания</b>\n"
                        "• телефон сбоку на уровне пояса, метров 2–3\n"
                        "• в кадре всё тело, от головы до стоп\n"
                        "• хороший свет, без других людей\n"
                        "• видео до {max} секунд, файл до 20 МБ\n\n"
                        "<b>Полный повтор</b>: бёдра опускаются примерно до колен "
                        "(угол в колене около 90°), затем полностью встаёшь."),
        "howto_pullup": ("🎥 <b>Как снимать подтягивания</b>\n"
                         "• камера сбоку или чуть спереди, метров 2–3\n"
                         "• видно перекладину и всё тело\n"
                         "• хороший свет, без других людей\n"
                         "• видео до {max} секунд, файл до 20 МБ\n\n"
                         "<b>Полный повтор</b>: из виса на прямых руках подбородок выше "
                         "уровня кистей, затем снова полностью выпрямленные руки."),
        "analyzing": "🔍 ИИ-судья смотрит видео…",
        "err_open": "😕 Не смог открыть видео. Пришли ещё раз (лучше mp4).",
        "err_no_pose": "😕 Не вижу упражнения. Сними так, чтобы всё тело было в кадре.",
        "err_too_long": "⏱ Видео слишком длинное. Максимум {max} секунд.",
        "err_too_big": "📦 Файл слишком большой (максимум 20 МБ). Сожми видео или сними короче.",
        "err_generic": "⚠️ Что-то пошло не так. Попробуй ещё раз.",
        "result": ("🏁 <b>Результат засчитан</b>\n"
                   "━━━━━━━━━━━━━━\n"
                   "{icon} {exname}: <b>{reps}</b>\n"
                   "🎯 Чистота: {qbar} <b>{q}%</b>\n"
                   "⏱ {dur:.0f} с · {pace:.1f} повт/с\n"
                   "⚠️ Неполных: {partial}\n"
                   "━━━━━━━━━━━━━━\n"
                   "⭐ Уровень <b>{lvl}</b> {lbar}"),
        "pb": "🔥 <b>Новый личный рекорд!</b>",
        "ach": "🏅 Достижение: {icon} <b>{name}</b>",
        "skin_new": "🎨 Открыт скин: <b>{name}</b>",
        "duel_ask": ("⚔️ <b>Дуэль создана</b> · {icon} {exname}\n"
                     "Нажми «📹 Считать вживую» под полем ввода и выполни подход. Потом отправишь вызов другу."),
        "duel_share": ("⚔️ <b>Вызов готов!</b>\n"
                       "Твой результат: <b>{n}</b>. Отправь ссылку другу, пусть попробует побить."),
        "duel_share_text": "Бросаю вызов: побей мои {n} ({exname})! ИИ-судья считает честно.",
        "duel_btn_share": "📤 Отправить другу",
        "duel_challenge": ("⚔️ <b>{name}</b> бросает тебе вызов!\n"
                           "Надо побить: <b>{n}</b> ({icon} {exname})."),
        "duel_accept": "✅ Принять вызов",
        "duel_join": "Принято! Нажми «📹 Считать вживую» под полем ввода и выполни подход.",
        "duel_not_ready": "Вызов ещё не готов: автор ещё не выполнил подход.",
        "duel_own": "Это твоя дуэль. Отправь ссылку другу:",
        "duel_gone": "Такой дуэли нет или она уже завершена.",
        "duel_done": "🏁 <b>Дуэль завершена</b>\n━━━━━━━━━━━━━━\n{a}: <b>{ra}</b>\n{b}: <b>{rb}</b>\n━━━━━━━━━━━━━━\n{verdict}",
        "duel_win": "🏆 Победил(а): <b>{name}</b>",
        "duel_draw": "🤝 Ничья!",
        "top_week": "🏆 <b>Рейтинг недели</b>\nлучший подход",
        "top_all": "🏆 <b>Рейтинг за всё время</b>\nлучший подход",
        "top_empty": "Пока пусто. Будь первым! 💪",
        "profile": ("👤 <b>{name}</b>\n"
                    "⭐ Уровень <b>{lvl}</b> {lbar}\n"
                    "💪 Всего: <b>{total}</b> · рекорд: <b>{best}</b> · подходов: {sets}\n"
                    "🎨 Скин: <b>{skin}</b>\n"
                    "🏅 Достижения: {ach}"),
        "no_ach": "пока нет",
        "skins_title": "🎨 <b>Скины карточек</b>\nОткрываются за общее число повторений.",
        "skin_locked": "🔒 {name}: нужно {n}",
        "skin_set": "✅ Скин выбран",
        "b_duel": "⚔️ Дуэль", "b_top": "🏆 Рейтинг", "b_skins": "🎨 Скины",
        "b_me": "👤 Профиль", "b_howto": "🎥 Как снимать",
        "b_week": "📅 Неделя", "b_all": "♾ Всё время",
        "send_video": "Нажми «📹 Считать вживую» под полем ввода: камера откроется прямо в Telegram 📹",
        "live_only": "📹 Видео по ссылке не принимаю: счёт идёт только вживую. Нажми «📹 Считать вживую» под полем ввода.",
        "live_only_app": "📹 Видео не принимаю: счёт идёт только вживую в приложении. Открой его кнопкой 👇",
        "app_welcome": ("🏋️ <b>RepDuel</b>: упражнения с ИИ-судьёй\n\n"
                        "Всё теперь в приложении: профиль, режимы, дуэли с друзьями, рейтинг и скины. "
                        "Камера считает повторения вживую.\n\nЖми кнопку 👇"),
        "app_open": "🏋️ Это теперь в приложении 👇",
        "app_btn": "🏋️ Открыть RepDuel",
    },
    "en": {
        "welcome": ("🏋️ <b>RepDuel</b>: workouts judged by AI\n\n"
                    "Pick an exercise and tap “📹 Count live”: the camera opens right here. I count your reps "
                    "honestly, no \"trust me bro\".\n\n"
                    "💪🦵🧗 push-ups, squats, pull-ups live\n"
                    "⚔️ duels with friends via link\n"
                    "🏆 weekly leaderboard\n"
                    "🎨 card skins unlocked by reps\n\n"
                    "Pick an action 👇"),
        "howto": ("🎥 <b>How to film</b>\n"
                  "• phone on the floor, from the side, about 2 m away\n"
                  "• your whole body in frame, hands to feet\n"
                  "• good light, nobody else in frame\n"
                  "• up to {max} seconds, file up to 20 MB\n\n"
                  "<b>Full rep</b>: elbows bend to about 90°, then lock out fully. "
                  "Partial reps don't count."),
        "howto_squat": ("🎥 <b>How to film squats</b>\n"
                        "• phone from the side at waist height, 2–3 m away\n"
                        "• whole body in frame, head to feet\n"
                        "• good light, nobody else in frame\n"
                        "• up to {max} seconds, file up to 20 MB\n\n"
                        "<b>Full rep</b>: hips go down to about knee level "
                        "(knee angle around 90°), then stand up fully."),
        "howto_pullup": ("🎥 <b>How to film pull-ups</b>\n"
                         "• camera from the side or slightly front, 2–3 m away\n"
                         "• the bar and your whole body in frame\n"
                         "• good light, nobody else in frame\n"
                         "• up to {max} seconds, file up to 20 MB\n\n"
                         "<b>Full rep</b>: from a dead hang, chin above wrist level, "
                         "then arms fully straight again."),
        "analyzing": "🔍 The AI judge is watching…",
        "err_open": "😕 Couldn't open the video. Send it again (mp4 works best).",
        "err_no_pose": "😕 I can't see the exercise. Film with your whole body in frame.",
        "err_too_long": "⏱ Video is too long. Max {max} seconds.",
        "err_too_big": "📦 File is too large (max 20 MB). Compress it or film a shorter set.",
        "err_generic": "⚠️ Something went wrong. Try again.",
        "result": ("🏁 <b>Result confirmed</b>\n"
                   "━━━━━━━━━━━━━━\n"
                   "{icon} {exname}: <b>{reps}</b>\n"
                   "🎯 Form: {qbar} <b>{q}%</b>\n"
                   "⏱ {dur:.0f}s · {pace:.1f} reps/s\n"
                   "⚠️ Partial: {partial}\n"
                   "━━━━━━━━━━━━━━\n"
                   "⭐ Level <b>{lvl}</b> {lbar}"),
        "pb": "🔥 <b>New personal best!</b>",
        "ach": "🏅 Achievement: {icon} <b>{name}</b>",
        "skin_new": "🎨 Skin unlocked: <b>{name}</b>",
        "duel_ask": ("⚔️ <b>Duel created</b> · {icon} {exname}\n"
                     "Tap “📹 Count live” below the input field and do your set. Then you'll send the challenge to a friend."),
        "duel_share": ("⚔️ <b>Challenge ready!</b>\n"
                       "Your score: <b>{n}</b>. Send the link to a friend and let them try to beat it."),
        "duel_share_text": "I challenge you: beat my {n} ({exname})! The AI judge counts honestly.",
        "duel_btn_share": "📤 Send to a friend",
        "duel_challenge": ("⚔️ <b>{name}</b> challenges you!\n"
                           "Beat <b>{n}</b> ({icon} {exname})."),
        "duel_accept": "✅ Accept challenge",
        "duel_join": "Accepted! Tap “📹 Count live” below the input field and do your set.",
        "duel_not_ready": "This challenge isn't ready: the author hasn't done their set yet.",
        "duel_own": "This is your duel. Send the link to a friend:",
        "duel_gone": "No such duel, or it's already finished.",
        "duel_done": "🏁 <b>Duel finished</b>\n━━━━━━━━━━━━━━\n{a}: <b>{ra}</b>\n{b}: <b>{rb}</b>\n━━━━━━━━━━━━━━\n{verdict}",
        "duel_win": "🏆 Winner: <b>{name}</b>",
        "duel_draw": "🤝 It's a draw!",
        "top_week": "🏆 <b>Weekly leaderboard</b>\nbest set",
        "top_all": "🏆 <b>All-time leaderboard</b>\nbest set",
        "top_empty": "Nothing here yet. Be the first! 💪",
        "profile": ("👤 <b>{name}</b>\n"
                    "⭐ Level <b>{lvl}</b> {lbar}\n"
                    "💪 Total: <b>{total}</b> · best: <b>{best}</b> · sets: {sets}\n"
                    "🎨 Skin: <b>{skin}</b>\n"
                    "🏅 Achievements: {ach}"),
        "no_ach": "none yet",
        "skins_title": "🎨 <b>Card skins</b>\nUnlocked by total reps.",
        "skin_locked": "🔒 {name}: need {n}",
        "skin_set": "✅ Skin selected",
        "b_duel": "⚔️ Duel", "b_top": "🏆 Top", "b_skins": "🎨 Skins",
        "b_me": "👤 Profile", "b_howto": "🎥 How to film",
        "b_week": "📅 Week", "b_all": "♾ All time",
        "send_video": "Tap “📹 Count live” below the input field: the camera opens right in Telegram 📹",
        "live_only": "📹 I don't accept videos: counting works live only. Tap “📹 Count live” below the input field.",
        "live_only_app": "📹 I don't accept videos: counting works live only, inside the app. Open it with the button 👇",
        "app_welcome": ("🏋️ <b>RepDuel</b>: workouts judged by AI\n\n"
                        "Everything lives in the app now: profile, modes, duels with friends, leaderboard and skins. "
                        "The camera counts your reps live.\n\nTap the button 👇"),
        "app_open": "🏋️ This lives in the app now 👇",
        "app_btn": "🏋️ Open RepDuel",
    },
}


def t(lang, key, **kw):
    s = T[lang].get(key) or T["en"][key]
    return s.format(**kw) if kw else s


def bar(p, n=10):
    p = max(0.0, min(1.0, p))
    k = round(p * n)
    return "█" * k + "░" * (n - k)


def esc(s):
    return html.escape(str(s or ""))


def skin_name(skin_id, lang):
    return SKINS[skin_id]["name"][0 if lang == "ru" else 1]


# ════════════════════════════════════════════════════════════════
#  6. TELEGRAM
# ════════════════════════════════════════════════════════════════

from telegram import (InlineKeyboardButton, InlineKeyboardMarkup, Update, BotCommand,  # noqa: E402
                      KeyboardButton, MenuButtonWebApp, ReplyKeyboardMarkup, WebAppInfo)
from telegram.constants import ParseMode  # noqa: E402
from telegram.error import BadRequest, TelegramError  # noqa: E402
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,  # noqa: E402
                          ContextTypes, MessageHandler, filters)

_sem = None


def sem():
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENT)
    return _sem


def howto_text(lang, ex):
    key = "howto" if ex == "pushup" else f"howto_{ex}"
    text = t(lang, key, max=MAX_VIDEO_SECONDS)
    if WEBAPP_URL and not ALLOW_VIDEO:  # живой режим: строки про файл видео не нужны
        text = "\n".join(l for l in text.split("\n") if "20 МБ" not in l and "20 MB" not in l)
    return text


def cur_ex(ctx):
    ex = ctx.user_data.get("ex", "pushup")
    return ex if ex in EXERCISES else "pushup"


def kb_main(lang):
    if API_URL:  # всё меню теперь в приложении
        return InlineKeyboardMarkup([[open_button(lang)]])
    ex_row = [InlineKeyboardButton(f"{e['icon']} {ex_name(k, lang)}", callback_data=f"ex:{k}")
              for k, e in EXERCISES.items()]
    return InlineKeyboardMarkup([
        ex_row,
        [InlineKeyboardButton(t(lang, "b_duel"), callback_data="duel_new"),
         InlineKeyboardButton(t(lang, "b_top"), callback_data="top:week")],
        [InlineKeyboardButton(t(lang, "b_skins"), callback_data="skins"),
         InlineKeyboardButton(t(lang, "b_me"), callback_data="me")],
        [InlineKeyboardButton(t(lang, "b_howto"), callback_data="howto")],
    ])


def top_view(lang, scope, ex="pushup"):
    since = time.time() - 7 * 86400 if scope == "week" else 0
    rows = q("""SELECT u.name name, MAX(r.reps) best, SUM(r.reps) total
                FROM results r JOIN users u ON u.id=r.user_id
                WHERE r.ts>? AND r.ex=? GROUP BY r.user_id
                ORDER BY best DESC, total DESC LIMIT 10""", (since, ex))
    text = (t(lang, "top_week" if scope == "week" else "top_all")
            + f"\n{EXERCISES[ex]['icon']} <b>{ex_name(ex, lang)}</b>\n━━━━━━━━━━━━━━\n")
    if not rows:
        text += t(lang, "top_empty")
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(rows):
        mark = medals[i] if i < 3 else f"{i + 1}."
        text += f"{mark} {esc(r['name'])}: <b>{r['best']}</b>\n"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(("• " if scope == "week" else "") + t(lang, "b_week"), callback_data=f"top:week:{ex}"),
         InlineKeyboardButton(("• " if scope == "all" else "") + t(lang, "b_all"), callback_data=f"top:all:{ex}")],
        [InlineKeyboardButton(("• " if k == ex else "") + e["icon"], callback_data=f"top:{scope}:{k}")
         for k, e in EXERCISES.items()],
    ])
    return text, kb


def profile_view(lang, uid):
    u = q("SELECT * FROM users WHERE id=?", (uid,), one=True)
    total, _best, sets = user_stats(uid)
    lvl, prog = level_info(total)
    bests = " ".join(f"{e['icon']}{user_best(uid, k)}" for k, e in EXERCISES.items())
    keys = [r["key"] for r in q("SELECT key FROM ach WHERE user_id=?", (uid,))]
    ach = " ".join(ACH[k][0] for k in keys if k in ACH) or t(lang, "no_ach")
    return t(lang, "profile", name=esc(u["name"]), lvl=lvl, lbar=bar(prog), total=total,
             best=bests, sets=sets, skin=skin_name(u["skin"], lang), ach=ach)


def skins_view(lang, uid):
    u = q("SELECT skin FROM users WHERE id=?", (uid,), one=True)
    total, _, _ = user_stats(uid)
    rows = []
    for sid, sk in SKINS.items():
        nm = skin_name(sid, lang)
        if total >= sk["unlock"]:
            label = ("✅ " if u["skin"] == sid else "🎨 ") + nm
            rows.append([InlineKeyboardButton(label, callback_data=f"skin:{sid}")])
        else:
            rows.append([InlineKeyboardButton(
                t(lang, "skin_locked", name=nm, n=sk["unlock"]), callback_data="noop")])
    return t(lang, "skins_title"), InlineKeyboardMarkup(rows)


def new_code():
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(6))


def share_markup(lang, bot_username, code, n, ex="pushup"):
    link = f"https://t.me/{bot_username}?start=duel_{code}"
    text = t(lang, "duel_share_text", n=n, exname=ex_name(ex, lang))
    share = f"https://t.me/share/url?url={quote(link)}&text={quote(text)}"
    return InlineKeyboardMarkup([[InlineKeyboardButton(t(lang, "duel_btn_share"), url=share)]]), link


# ── команды ────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    lang = ensure_user(user)
    ctx.user_data["mode"] = {"kind": "solo"}
    args = ctx.args or []
    if args and args[0].startswith("duel_"):
        return await show_duel_link(update, ctx, lang, args[0][5:])
    if API_URL:
        return await update.effective_message.reply_text(
            t(lang, "app_welcome"), parse_mode=ParseMode.HTML, reply_markup=kb_main(lang))
    await update.effective_message.reply_text(
        t(lang, "welcome"), parse_mode=ParseMode.HTML, reply_markup=kb_main(lang))
    live = kb_live(lang, cur_ex(ctx))
    if live:  # reply-кнопка «Считать вживую» появится под полем ввода
        await update.effective_message.reply_text(t(lang, "send_video"), reply_markup=live)


async def show_duel_link(update, ctx, lang, code):
    msg, user = update.effective_message, update.effective_user
    d = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
    if not d or d["status"] == "done":
        return await msg.reply_text(t(lang, "duel_gone"))
    if d["creator"] == user.id and API_URL and d["status"] == "waiting":
        return await msg.reply_text(t(lang, "app_open"),
                                    reply_markup=InlineKeyboardMarkup([[open_button(lang, duel=code, tab="duels")]]))
    if d["creator"] == user.id:
        markup, link = share_markup(lang, ctx.bot.username, code, d["c_reps"] or 0, d["ex"] or "pushup")
        return await msg.reply_text(t(lang, "duel_own") + "\n" + link, reply_markup=markup)
    if d["status"] != "open":
        return await msg.reply_text(t(lang, "duel_not_ready"))
    creator = q("SELECT name FROM users WHERE id=?", (d["creator"],), one=True)
    if API_URL:
        kb = InlineKeyboardMarkup([[open_button(lang, duel=code, label=t(lang, "duel_accept"))]])
    else:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(t(lang, "duel_accept"), callback_data=f"join:{code}")]])
    dex = d["ex"] or "pushup"
    await msg.reply_text(t(lang, "duel_challenge", name=esc(creator["name"]), n=d["c_reps"],
                           icon=EXERCISES[dex]["icon"], exname=ex_name(dex, lang)),
                         parse_mode=ParseMode.HTML, reply_markup=kb)


async def cmd_help(update, ctx):
    lang = ensure_user(update.effective_user)
    await update.effective_message.reply_text(
        howto_text(lang, cur_ex(ctx)), parse_mode=ParseMode.HTML, reply_markup=kb_main(lang))


async def start_duel(update, ctx, lang):
    code = new_code()
    ex = cur_ex(ctx)
    x("INSERT INTO duels(code,creator,status,created,ex) VALUES(?,?,?,?,?)",
      (code, update.effective_user.id, "waiting", time.time(), ex))
    ctx.user_data["mode"] = {"kind": "duel_create", "code": code}
    await update.effective_chat.send_message(
        t(lang, "duel_ask", icon=EXERCISES[ex]["icon"], exname=ex_name(ex, lang)) + "\n\n" + howto_text(lang, ex),
        parse_mode=ParseMode.HTML, reply_markup=kb_live(lang, ex, code))


async def open_in_app(update, lang, tab=None, ex=None):
    kb = InlineKeyboardMarkup([[open_button(lang, ex=ex, tab=tab)]])
    await update.effective_message.reply_text(t(lang, "app_open"), reply_markup=kb)


async def cmd_duel(update, ctx):
    lang = ensure_user(update.effective_user)
    if API_URL:
        return await open_in_app(update, lang, tab="duels")
    await start_duel(update, ctx, lang)


async def cmd_top(update, ctx):
    lang = ensure_user(update.effective_user)
    if API_URL:
        return await open_in_app(update, lang, tab="top")
    text, kb = top_view(lang, "week", cur_ex(ctx))
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)


async def cmd_me(update, ctx):
    lang = ensure_user(update.effective_user)
    if API_URL:
        return await open_in_app(update, lang, tab="profile")
    await update.effective_message.reply_text(
        profile_view(lang, update.effective_user.id), parse_mode=ParseMode.HTML, reply_markup=kb_main(lang))


async def _choose_ex(update, ctx, ex):
    lang = ensure_user(update.effective_user)
    if API_URL:
        return await open_in_app(update, lang, tab="modes", ex=ex)
    ctx.user_data["ex"] = ex
    ctx.user_data["mode"] = {"kind": "solo"}
    await update.effective_message.reply_text(
        howto_text(lang, ex) + "\n\n" + t(lang, "send_video"),
        parse_mode=ParseMode.HTML, reply_markup=kb_live(lang, ex))


async def cmd_pushups(update, ctx):
    await _choose_ex(update, ctx, "pushup")


async def cmd_squats(update, ctx):
    await _choose_ex(update, ctx, "squat")


async def cmd_pullups(update, ctx):
    await _choose_ex(update, ctx, "pullup")


async def cmd_skins(update, ctx):
    lang = ensure_user(update.effective_user)
    if API_URL:
        return await open_in_app(update, lang, tab="skins")
    text, kb = skins_view(lang, update.effective_user.id)
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)


# ── кнопки ─────────────────────────────────────────────────────

async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cq = update.callback_query
    user = update.effective_user
    lang = ensure_user(user)
    data = cq.data or ""
    await cq.answer()
    msg = cq.message
    try:
        if data == "noop":
            return
        if data == "duel_new":
            return await start_duel(update, ctx, lang)
        if data.startswith("ex:"):
            ex = data[3:]
            if ex in EXERCISES:
                ctx.user_data["ex"] = ex
                ctx.user_data["mode"] = {"kind": "solo"}
                return await msg.reply_text(howto_text(lang, ex) + "\n\n" + t(lang, "send_video"),
                                            parse_mode=ParseMode.HTML, reply_markup=kb_live(lang, ex))
            return
        if data == "howto":
            return await msg.reply_text(howto_text(lang, cur_ex(ctx)), parse_mode=ParseMode.HTML)
        if data == "me":
            return await msg.reply_text(profile_view(lang, user.id), parse_mode=ParseMode.HTML,
                                        reply_markup=kb_main(lang))
        if data.startswith("top:"):
            parts = data.split(":")
            top_ex = parts[2] if len(parts) > 2 and parts[2] in EXERCISES else cur_ex(ctx)
            text, kb = top_view(lang, parts[1], top_ex)
            try:
                return await msg.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)
            except BadRequest:  # то же самое содержимое
                return
        if data == "skins":
            text, kb = skins_view(lang, user.id)
            return await msg.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)
        if data.startswith("skin:"):
            sid = data[5:]
            total, _, _ = user_stats(user.id)
            if sid in SKINS and total >= SKINS[sid]["unlock"]:
                x("UPDATE users SET skin=? WHERE id=?", (sid, user.id))
                text, kb = skins_view(lang, user.id)
                try:
                    await msg.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)
                except BadRequest:
                    pass
                return
        if data.startswith("join:"):
            code = data[5:]
            d = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
            if not d or d["status"] != "open" or d["creator"] == user.id:
                return await msg.reply_text(t(lang, "duel_gone"))
            ctx.user_data["mode"] = {"kind": "duel_join", "code": code}
            return await msg.reply_text(t(lang, "duel_join") + "\n\n" + howto_text(lang, d["ex"] or "pushup"),
                                        parse_mode=ParseMode.HTML,
                                        reply_markup=kb_live(lang, d["ex"] or "pushup", code))
    except TelegramError as e:
        log.warning("button error: %s", e)


# ── видео ──────────────────────────────────────────────────────

async def on_media(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, user = update.effective_message, update.effective_user
    lang = ensure_user(user)
    if API_URL and not ALLOW_VIDEO:
        return await msg.reply_text(t(lang, "live_only_app"), reply_markup=kb_main(lang))
    if WEBAPP_URL and not ALLOW_VIDEO:
        # только живой режим: видео не принимаем, возвращаем кнопку Mini App
        mode = ctx.user_data.get("mode") or {"kind": "solo"}
        code = mode.get("code") if mode["kind"] in ("duel_create", "duel_join") else None
        ex = cur_ex(ctx)
        if code:
            d = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
            if d:
                ex = d["ex"] or "pushup"
            else:
                code = None
        return await msg.reply_text(t(lang, "live_only"), reply_markup=kb_live(lang, ex, code or ""))
    media = msg.video or msg.video_note
    if media is None and msg.document and (msg.document.mime_type or "").startswith("video/"):
        media = msg.document
    if media is None:
        return
    if media.file_size and media.file_size > MAX_DOWNLOAD_BYTES:
        return await msg.reply_text(t(lang, "err_too_big"))
    if getattr(media, "duration", None) and media.duration > MAX_VIDEO_SECONDS:
        return await msg.reply_text(t(lang, "err_too_long", max=MAX_VIDEO_SECONDS))

    mode = ctx.user_data.get("mode") or {"kind": "solo"}
    code = mode.get("code") if mode["kind"] in ("duel_create", "duel_join") else None
    ex = cur_ex(ctx)
    if code:
        d = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
        if not d or (mode["kind"] == "duel_join" and d["status"] != "open"):
            ctx.user_data["mode"] = {"kind": "solo"}
            return await msg.reply_text(t(lang, "duel_gone"))
        ex = d["ex"] or "pushup"

    status = await msg.reply_text(t(lang, "analyzing"))
    tmp = tempfile.mkdtemp(prefix="repduel_")
    path = os.path.join(tmp, "video.mp4")
    try:
        tg_file = await media.get_file()
        await tg_file.download_to_drive(path)
        async with sem():
            res: RepResult = await asyncio.to_thread(analyze_video, path, ex)
    except VideoError as e:
        key = {"open": "err_open", "too_long": "err_too_long", "no_pose": "err_no_pose"}.get(e.code, "err_generic")
        return await status.edit_text(t(lang, key, max=MAX_VIDEO_SECONDS))
    except BadRequest as e:
        key = "err_too_big" if "too big" in str(e).lower() else "err_generic"
        return await status.edit_text(t(lang, key))
    except Exception:
        log.exception("analysis failed")
        return await status.edit_text(t(lang, "err_generic"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    await finish(update, ctx, res, ex, mode, code, status)


def commit_result(uid, res: RepResult, ex: str, kind: str, code):
    """Записывает подход в базу и считает всё, что нужно показать (общая часть для бота и приложения).
    Бросает ValueError("duel_gone"), если дуэль уже недоступна."""
    total_before, _b, _s = user_stats(uid)
    best_before = user_best(uid, ex)
    if kind == "duel_create":
        if not x("UPDATE duels SET c_reps=?, status='open' WHERE code=? AND creator=? AND status='waiting'",
                 (res.reps, code, uid)):
            raise ValueError("duel_gone")
    elif kind == "duel_join":
        if not x("UPDATE duels SET opponent=?, o_reps=?, status='done' "
                 "WHERE code=? AND status='open' AND creator!=?", (uid, res.reps, code, uid)):
            raise ValueError("duel_gone")
    x("INSERT INTO results(user_id,reps,partial,duration,ts,duel,ex) VALUES(?,?,?,?,?,?,?)",
      (uid, res.reps, res.partial, res.duration, time.time(), code, ex))
    total, best, _ = user_stats(uid)
    lvl, prog = level_info(total)
    attempts = res.reps + res.partial
    quality = round(100 * res.reps / attempts) if attempts else 100
    skin_id = q("SELECT skin FROM users WHERE id=?", (uid,), one=True)["skin"]
    new_ach = unlock_achievements(uid, res.reps, total, duel_wins(uid))
    new_skins = [sid for sid, sk in SKINS.items() if total_before < sk["unlock"] <= total and sk["unlock"] > 0]
    return dict(reps=res.reps, partial=res.partial, duration=res.duration, ex=ex, quality=quality,
                lvl=lvl, prog=prog, total=total, skin=skin_id, new_ach=new_ach, new_skins=new_skins,
                pb=(best_before > 0 and res.reps > best_before),
                duel_finished=(kind == "duel_join"))


async def build_result_message(bot, lang, name, info, kind, code):
    """Карточка, подпись и кнопки для сообщения с результатом."""
    ex = info["ex"]
    caption = t(lang, "result", icon=EXERCISES[ex]["icon"], exname=ex_name(ex, lang), reps=info["reps"],
                qbar=bar(info["quality"] / 100), q=info["quality"], dur=info["duration"],
                pace=(info["reps"] / info["duration"] if info["duration"] else 0), partial=info["partial"],
                lvl=info["lvl"], lbar=bar(info["prog"]))
    if info["pb"]:
        caption = t(lang, "pb") + "\n\n" + caption
    for k in info["new_ach"]:
        caption += "\n" + t(lang, "ach", icon=ACH[k][0], name=ACH[k][1 if lang == "ru" else 2])
    for sid in info["new_skins"]:
        caption += "\n" + t(lang, "skin_new", name=skin_name(sid, lang))
    card = await asyncio.to_thread(render_card, name or "Athlete", info["reps"], info["duration"],
                                   info["quality"], info["lvl"], info["skin"], bot.username or "", ex)
    markup = kb_main(lang)
    if kind == "duel_create":
        markup, _link = share_markup(lang, bot.username, code, info["reps"], ex)
        caption += "\n\n" + t(lang, "duel_share", n=info["reps"])
    return card, caption, markup


async def notify_duel_done(bot, code):
    """Итог дуэли обоим участникам."""
    d = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
    if not d or not d["opponent"]:
        return
    a = q("SELECT name,lang FROM users WHERE id=?", (d["creator"],), one=True)
    b = q("SELECT name,lang FROM users WHERE id=?", (d["opponent"],), one=True)
    for uid, person in ((d["creator"], a), (d["opponent"], b)):
        l = person["lang"] or "en"
        if d["c_reps"] == d["o_reps"]:
            verdict = t(l, "duel_draw")
        else:
            winner = a if d["c_reps"] > d["o_reps"] else b
            verdict = t(l, "duel_win", name=esc(winner["name"]))
        text = t(l, "duel_done", a=esc(a["name"]), ra=d["c_reps"], b=esc(b["name"]), rb=d["o_reps"],
                 verdict=verdict)
        try:
            await bot.send_message(uid, text, parse_mode=ParseMode.HTML)
        except TelegramError as e:
            log.warning("cannot notify %s: %s", uid, e)


async def finish(update, ctx, res: RepResult, ex: str, mode: dict, code, status=None):
    msg, user = update.effective_message, update.effective_user
    lang = ensure_user(user)
    try:
        info = commit_result(user.id, res, ex, mode["kind"], code)
    except ValueError:
        ctx.user_data["mode"] = {"kind": "solo"}
        return await msg.reply_text(t(lang, "duel_gone"))
    ctx.user_data["mode"] = {"kind": "solo"}
    card, caption, markup = await build_result_message(ctx.bot, lang, user.full_name, info, mode["kind"], code)
    if status:
        try:
            await status.delete()
        except TelegramError:
            pass
    await msg.reply_photo(photo=card, caption=caption, parse_mode=ParseMode.HTML, reply_markup=markup)
    if info["duel_finished"]:
        await notify_duel_done(ctx.bot, code)


# ── Mini App (счёт вживую) ─────────────────────────────────────
# Адрес Mini App: впиши сюда свою ссылку на GitHub Pages, например
# "https://ник.github.io/репо/index.html". Либо задай переменную окружения WEBAPP_URL.
WEBAPP_URL = os.environ.get("WEBAPP_URL", "").strip() or "https://yllubully7.github.io/duel/index.html"


# True = разрешить ещё и видео-файлы (запасной путь). False = только живой режим в Mini App.
ALLOW_VIDEO = False


def kb_live(lang, ex, duel_code=""):
    """Reply-клавиатура: только из неё Mini App может вернуть данные через sendData."""
    if not WEBAPP_URL:
        return None
    url = f"{WEBAPP_URL}?ex={ex}" + (f"&duel={duel_code}" if duel_code else "")
    label = "📹 Считать вживую" if lang == "ru" else "📹 Count live"
    return ReplyKeyboardMarkup([[KeyboardButton(label, web_app=WebAppInfo(url=url))]],
                               resize_keyboard=True)


async def on_webapp_data(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    try:
        d = json.loads(msg.web_app_data.data)
        ex = d["ex"] if d["ex"] in EXERCISES else "pushup"
        reps = max(0, min(int(d["reps"]), 500))
        partial = max(0, min(int(d["partial"]), 500))
        duration = max(1.0, min(float(d["duration"]), 600.0))
        valid_ratio = float(d.get("valid_ratio", 1))
        if valid_ratio < 0.25:
            raise ValueError("low quality")
        # антиабьюз: не быстрее 0.3 с на повтор
        if reps / duration > 3.0:
            raise ValueError("too fast")
    except Exception:
        return await msg.reply_text("⚠️")
    res = RepResult(reps, partial, duration, valid_ratio, 0)

    mode = ctx.user_data.get("mode") or {"kind": "solo"}
    code = d.get("duel") or None
    if code and mode.get("code") != code:
        # дуэль открыта из ссылки: определяем роль по базе
        row = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
        uid = update.effective_user.id
        if row and row["creator"] == uid and row["status"] == "waiting":
            mode = {"kind": "duel_create", "code": code}
        elif row and row["status"] == "open" and row["creator"] != uid:
            mode = {"kind": "duel_join", "code": code}
        else:
            mode, code = {"kind": "solo"}, None
    elif not code and mode.get("kind") in ("duel_create", "duel_join"):
        code = mode.get("code")
    if code and mode.get("kind") in ("duel_create", "duel_join"):
        # упражнение в дуэли задаёт сама дуэль, а не клиент
        row = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
        if not row or (mode["kind"] == "duel_join" and row["status"] != "open"):
            ctx.user_data["mode"] = {"kind": "solo"}
            return await msg.reply_text(t(ensure_user(update.effective_user), "duel_gone"))
        ex = row["ex"] or "pushup"
    else:
        code = None
    await finish(update, ctx, res, ex, mode, code)


# ════════════════════════════════════════════════════════════════
#  Приложение: API для Mini App + туннель
# ════════════════════════════════════════════════════════════════
# Mini App живёт на GitHub Pages, а данные (профиль, рейтинг, дуэли) лежат у бота.
# Поэтому бот поднимает маленький HTTP-сервер, а наружу его открывает туннель Cloudflare
# (положи cloudflared.exe в папку с bot.py: бот сам его запустит и узнает адрес).
# Свой адрес можно задать переменной API_URL (например, если бот стоит на сервере).
API_PORT = int(os.environ.get("PORT", os.environ.get("API_PORT", "8080")))
API_URL = os.environ.get("API_URL", "").strip().rstrip("/") or "https://duel.onrender.com"
_api = {"token": "", "bot": None, "server": None, "tunnel": None}
_bg_tasks: set = set()
_avatar_cache: dict = {}


def _spawn(coro):
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def app_url(ex=None, duel=None, tab=None):
    params = {}
    if API_URL:
        params["api"] = API_URL
    if tab:
        params["tab"] = tab
    if ex:
        params["ex"] = ex
    if duel:
        params["duel"] = duel
    return WEBAPP_URL + ("?" + urlencode(params) if params else "")


def open_button(lang, ex=None, duel=None, tab=None, label=None):
    return InlineKeyboardButton(label or t(lang, "app_btn"),
                                web_app=WebAppInfo(url=app_url(ex, duel, tab)))


def check_init_data(init_data: str, token: str, max_age: int = 86400):
    """Проверка подписи Telegram (initData из Mini App). Возвращает словарь пользователя или None."""
    try:
        pairs = dict(parse_qsl(init_data, keep_blank_values=True))
        got = pairs.pop("hash", "")
        check = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
        secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
        calc = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
        if not got or not hmac.compare_digest(calc, got):
            return None
        if time.time() - int(pairs.get("auth_date", "0")) > max_age:
            return None
        user = json.loads(pairs["user"])
        return user if "id" in user else None
    except Exception:
        return None


class _ApiUser:
    """Пользователь из initData, с теми же полями, что и у telegram.User."""
    def __init__(self, d):
        self.id = int(d["id"])
        self.full_name = ((d.get("first_name") or "") + " " + (d.get("last_name") or "")).strip()
        self.username = d.get("username")
        self.language_code = d.get("language_code")


def _j(status, obj):
    return status, "application/json; charset=utf-8", json.dumps(obj, ensure_ascii=False).encode()


def _outcome(d, uid):
    """Итог дуэли глазами пользователя: pending / win / lose / draw."""
    if d["status"] != "done":
        return "pending"
    if d["c_reps"] == d["o_reps"]:
        return "draw"
    i_am_creator = d["creator"] == uid
    creator_won = d["c_reps"] > d["o_reps"]
    return "win" if creator_won == i_am_creator else "lose"


def _duel_view(d, uid):
    mine = d["creator"] == uid
    return dict(
        code=d["code"], ex=d["ex"] or "pushup", status=d["status"],
        role="creator" if mine else "opponent",
        me_reps=(d["c_reps"] if mine else d["o_reps"]),
        opp_reps=(d["o_reps"] if mine else d["c_reps"]) if d["status"] == "done" else None,
        opp_name=(d["oname"] if mine else d["cname"]),
        outcome=_outcome(d, uid), ts=d["created"])


def api_me(u, lang):
    total, best, sets = user_stats(u.id)
    lvl, prog = level_info(total)
    skin = q("SELECT skin FROM users WHERE id=?", (u.id,), one=True)["skin"]
    keys = {r["key"] for r in q("SELECT key FROM ach WHERE user_id=?", (u.id,))}
    per_ex = {r["ex"]: int(r["s"]) for r in q(
        "SELECT ex, SUM(reps) s FROM results WHERE user_id=? GROUP BY ex", (u.id,))}
    week = q("SELECT COALESCE(SUM(reps),0) s FROM results WHERE user_id=? AND ts>?",
             (u.id, time.time() - 7 * 86400), one=True)["s"]
    return dict(
        user=dict(id=u.id, name=u.full_name or u.username or str(u.id), username=u.username, lang=lang),
        bot=_api["bot"].username if _api["bot"] else "",
        level=lvl, progress=prog, lo=10 * (lvl - 1) ** 2, hi=10 * lvl ** 2,
        total=total, best=best, sets=sets, week=int(week), duel_wins=duel_wins(u.id),
        bests={k: user_best(u.id, k) for k in EXERCISES},
        totals={k: per_ex.get(k, 0) for k in EXERCISES},
        skin=skin,
        skins=[dict(id=sid, name=skin_name(sid, lang), unlock=sk["unlock"], unlocked=total >= sk["unlock"],
                    c1=sk["c1"], c2=sk["c2"], accent=sk["accent"]) for sid, sk in SKINS.items()],
        ach=[dict(key=k, icon=v[0], name=v[1 if lang == "ru" else 2], unlocked=k in keys) for k, v in ACH.items()],
        recent=[dict(ex=r["ex"] or "pushup", reps=r["reps"], ts=r["ts"]) for r in q(
            "SELECT ex, reps, ts FROM results WHERE user_id=? ORDER BY ts DESC LIMIT 6", (u.id,))])


def api_top(u, query):
    scope = "all" if query.get("scope") == "all" else "week"
    ex = query.get("ex") if query.get("ex") in EXERCISES else "pushup"
    since = time.time() - 7 * 86400 if scope == "week" else 0
    rows = q("""SELECT u.id uid, u.name name, MAX(r.reps) best, SUM(r.reps) total
                FROM results r JOIN users u ON u.id=r.user_id
                WHERE r.ts>? AND r.ex=? GROUP BY r.user_id
                ORDER BY best DESC, total DESC LIMIT 100""", (since, ex))
    out, me = [], None
    for i, r in enumerate(rows, 1):
        item = dict(rank=i, name=r["name"], best=r["best"], total=r["total"], me=(r["uid"] == u.id))
        if item["me"]:
            me = item
        if i <= 10:
            out.append(item)
    return dict(scope=scope, ex=ex, rows=out, me=me if me and me["rank"] > 10 else None)


def api_duels(u):
    rows = q("""SELECT d.*, c.name cname, o.name oname FROM duels d
                LEFT JOIN users c ON c.id=d.creator LEFT JOIN users o ON o.id=d.opponent
                WHERE d.creator=? OR d.opponent=? ORDER BY d.created DESC LIMIT 20""", (u.id, u.id))
    return dict(duels=[_duel_view(d, u.id) for d in rows])


def api_duel_new(u, data):
    ex = data.get("ex") if data.get("ex") in EXERCISES else "pushup"
    old = q("SELECT code FROM duels WHERE creator=? AND status='waiting' AND ex=? ORDER BY created DESC LIMIT 1",
            (u.id, ex), one=True)
    if old:  # не плодим пустые дуэли
        return dict(code=old["code"], ex=ex)
    code = new_code()
    x("INSERT INTO duels(code,creator,status,created,ex) VALUES(?,?,?,?,?)",
      (code, u.id, "waiting", time.time(), ex))
    return dict(code=code, ex=ex)


def api_duel_get(u, query):
    code = (query.get("code") or "").strip().upper()
    d = q("""SELECT d.*, c.name cname, o.name oname FROM duels d
             LEFT JOIN users c ON c.id=d.creator LEFT JOIN users o ON o.id=d.opponent
             WHERE d.code=?""", (code,), one=True)
    if not d:
        return None
    mine = d["creator"] == u.id
    return dict(code=code, ex=d["ex"] or "pushup", status=d["status"], role="creator" if mine else (
        "opponent" if d["opponent"] == u.id else "guest"),
        creator_name=d["cname"], c_reps=d["c_reps"] if d["status"] != "waiting" else None,
        can_accept=(d["status"] == "open" and not mine),
        view=_duel_view(d, u.id) if (mine or d["opponent"] == u.id) else None)


async def api_result(u, lang, data):
    try:
        ex = data.get("ex") if data.get("ex") in EXERCISES else "pushup"
        reps = max(0, min(int(data["reps"]), 500))
        partial = max(0, min(int(data["partial"]), 500))
        duration = max(1.0, min(float(data["duration"]), 600.0))
        valid_ratio = float(data.get("valid_ratio", 1))
    except Exception:
        return _j(400, {"error": "bad"})
    if reps < 1:
        return _j(400, {"error": "empty"})
    if valid_ratio < 0.25:
        return _j(400, {"error": "low_quality"})
    if reps / duration > 3.0:  # антиабьюз: не быстрее 0.3 с на повтор
        return _j(400, {"error": "too_fast"})
    res = RepResult(reps, partial, duration, valid_ratio, 0)

    code = (data.get("duel") or "").strip().upper() or None
    kind = "solo"
    if code:
        row = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
        if row and row["creator"] == u.id and row["status"] == "waiting":
            kind = "duel_create"
        elif row and row["status"] == "open" and row["creator"] != u.id:
            kind = "duel_join"
        else:
            return _j(409, {"error": "duel_gone"})
        ex = row["ex"] or "pushup"  # упражнение в дуэли задаёт сама дуэль
    try:
        info = commit_result(u.id, res, ex, kind, code)
    except ValueError:
        return _j(409, {"error": "duel_gone"})

    duel = None
    if code:
        d = q("""SELECT d.*, c.name cname, o.name oname FROM duels d
                 LEFT JOIN users c ON c.id=d.creator LEFT JOIN users o ON o.id=d.opponent
                 WHERE d.code=?""", (code,), one=True)
        duel = _duel_view(d, u.id)
    _spawn(_send_result_to_chat(u, lang, info, kind, code))
    return _j(200, dict(
        ok=True, ex=ex, reps=info["reps"], partial=info["partial"], duration=info["duration"],
        quality=info["quality"], level=info["lvl"], progress=info["prog"], total=info["total"],
        lo=10 * (info["lvl"] - 1) ** 2, hi=10 * info["lvl"] ** 2, pb=info["pb"],
        ach=[dict(key=k, icon=ACH[k][0], name=ACH[k][1 if lang == "ru" else 2]) for k in info["new_ach"]],
        skins=[dict(id=sid, name=skin_name(sid, lang)) for sid in info["new_skins"]],
        duel=duel))


async def _send_result_to_chat(u, lang, info, kind, code):
    """Карточка результата в чат с ботом (её можно переслать или выложить в сторис)."""
    bot = _api["bot"]
    try:
        card, caption, markup = await build_result_message(bot, lang, u.full_name, info, kind, code)
        await bot.send_photo(u.id, photo=card, caption=caption, parse_mode=ParseMode.HTML, reply_markup=markup)
    except Exception as e:
        log.warning("cannot send result card to %s: %s", u.id, e)
    if info["duel_finished"]:
        await notify_duel_done(bot, code)


async def api_avatar(u):
    cached = _avatar_cache.get(u.id)
    if cached and time.time() - cached[0] < 3600:
        data = cached[1]
    else:
        data = None
        try:
            photos = await _api["bot"].get_user_profile_photos(u.id, limit=1)
            if photos.total_count and photos.photos:
                sizes = photos.photos[0]
                pick = next((p for p in sizes if p.width >= 160), sizes[-1])
                f = await _api["bot"].get_file(pick.file_id)
                data = bytes(await f.download_as_bytearray())
        except Exception as e:
            log.warning("avatar for %s: %s", u.id, e)
        _avatar_cache[u.id] = (time.time(), data)
    if not data:
        return _j(404, {"error": "no_avatar"})
    return 200, "image/jpeg", data


async def api_route(method, path, query, headers, body):
    if method == "GET" and path in ("/", "/health"):
        return _j(200, {"ok": True, "app": "RepDuel"})
    auth = headers.get("authorization", "")
    user = check_init_data(auth[4:].strip(), _api["token"]) if auth.lower().startswith("tma ") else None
    if not user:
        return _j(401, {"error": "auth"})
    try:
        data = json.loads(body) if body else {}
        if not isinstance(data, dict):
            raise ValueError
    except ValueError:
        return _j(400, {"error": "json"})
    u = _ApiUser(user)
    lang = ensure_user(u)
    if method == "GET":
        if path == "/api/me":
            return _j(200, api_me(u, lang))
        if path == "/api/top":
            return _j(200, api_top(u, query))
        if path == "/api/duels":
            return _j(200, api_duels(u))
        if path == "/api/duel":
            r = api_duel_get(u, query)
            return _j(200, r) if r else _j(404, {"error": "duel_gone"})
        if path == "/api/avatar":
            return await api_avatar(u)
    elif method == "POST":
        if path == "/api/result":
            return await api_result(u, lang, data)
        if path == "/api/duel/new":
            return _j(200, api_duel_new(u, data))
        if path == "/api/skin":
            sid = data.get("skin")
            total = user_stats(u.id)[0]
            if sid in SKINS and total >= SKINS[sid]["unlock"]:
                x("UPDATE users SET skin=? WHERE id=?", (sid, u.id))
                return _j(200, {"ok": True, "skin": sid})
            return _j(400, {"error": "locked"})
    return _j(404, {"error": "not_found"})


_CORS = ["Access-Control-Allow-Origin: *",
         "Access-Control-Allow-Headers: Authorization, Content-Type",
         "Access-Control-Allow-Methods: GET, POST, OPTIONS",
         "Access-Control-Max-Age: 86400"]
_REASONS = {200: "OK", 204: "No Content", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
            409: "Conflict", 413: "Payload Too Large", 500: "Internal Server Error"}


async def _http_client(reader, writer):
    """Минимальный HTTP/1.1-сервер на стандартной библиотеке (без aiohttp)."""
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15)
        lines = head.decode("latin1").split("\r\n")
        method, target, _ = lines[0].split(" ", 2)
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length") or 0)
        if n > 65536:
            status, ctype, payload = _j(413, {"error": "too_big"})
        else:
            body = await asyncio.wait_for(reader.readexactly(n), 15) if n else b""
            sp = urlsplit(target)
            if method == "OPTIONS":
                status, ctype, payload = 204, "text/plain", b""
            else:
                status, ctype, payload = await api_route(method, sp.path, dict(parse_qsl(sp.query)), headers, body)
    except (asyncio.IncompleteReadError, asyncio.TimeoutError, asyncio.LimitOverrunError, ConnectionError):
        writer.close()
        return
    except Exception:
        log.exception("api error")
        status, ctype, payload = _j(500, {"error": "server"})
    try:
        out = [f"HTTP/1.1 {status} {_REASONS.get(status, 'OK')}", f"Content-Type: {ctype}",
               f"Content-Length: {len(payload)}", "Connection: close", "Cache-Control: no-store"] + _CORS
        writer.write(("\r\n".join(out) + "\r\n\r\n").encode() + payload)
        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()


def _find_cloudflared():
    for name in ("cloudflared.exe", "cloudflared"):
        p = BASE_DIR / name
        if p.exists():
            return str(p)
    return shutil.which("cloudflared")


async def start_tunnel(port: int):
    """Запускает cloudflared и ждёт адрес вида https://….trycloudflare.com. Возвращает (процесс, адрес)."""
    exe = _find_cloudflared()
    if not exe:
        return None, None
    proc = await asyncio.create_subprocess_exec(
        exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate", "--protocol", "http2",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    fut = asyncio.get_running_loop().create_future()
    tail = deque(maxlen=6)

    async def reader():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            tail.append(line.decode(errors="ignore").strip())
            for m in re.finditer(r"https://[a-z0-9-]+\.trycloudflare\.com", line.decode(errors="ignore")):
                if not m.group(0).startswith("https://api.") and not fut.done():
                    fut.set_result(m.group(0))
        if not fut.done():
            fut.set_result(None)

    _spawn(reader())
    try:
        url = await asyncio.wait_for(fut, 45)
    except asyncio.TimeoutError:
        url = None
    if not url:
        log.warning("cloudflared не дал адрес. Последние строки:\n%s", "\n".join(tail))
    return proc, url


async def start_api(app):
    global API_URL
    _api["token"], _api["bot"] = app.bot.token, app.bot
    try:
        _api["server"] = await asyncio.start_server(_http_client, "0.0.0.0", API_PORT)
    except OSError as e:
        log.error("Не удалось открыть порт %s (%s). Закрой другой запущенный bot.py.", API_PORT, e)
        return
    if not API_URL:
        _api["tunnel"], url = await start_tunnel(API_PORT)
        if url:
            API_URL = url
    if not API_URL:
        log.warning("Нет адреса для приложения: положи cloudflared.exe рядом с bot.py (или задай API_URL). "
                    "Пока бот работает в упрощённом режиме (только кнопка «Считать вживую»).")
        return
    log.info("Приложение подключено: %s", API_URL)
    try:
        await app.bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(text="RepDuel", web_app=WebAppInfo(url=app_url())))
    except TelegramError as e:
        log.warning("Не удалось поставить кнопку меню: %s", e)


async def stop_api(app):
    if _api["server"]:
        _api["server"].close()
    proc = _api["tunnel"]
    if proc and proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass



async def on_text(update, ctx):
    lang = ensure_user(update.effective_user)
    if API_URL:
        return await update.effective_message.reply_text(t(lang, "app_open"), reply_markup=kb_main(lang))
    await update.effective_message.reply_text(t(lang, "send_video"), reply_markup=kb_main(lang))


async def post_init(app: Application):
    await start_api(app)
    await app.bot.set_my_commands([
        BotCommand("start", "Start / Старт"),
        BotCommand("pushups", "Push-ups / Отжимания"),
        BotCommand("squats", "Squats / Приседания"),
        BotCommand("pullups", "Pull-ups / Подтягивания"),
        BotCommand("duel", "Duel / Дуэль"),
        BotCommand("top", "Leaderboard / Рейтинг"),
        BotCommand("me", "Profile / Профиль"),
        BotCommand("skins", "Skins / Скины"),
        BotCommand("help", "How to film / Как снимать"),
    ])


def build_app(token: str) -> Application:
    from telegram.request import HTTPXRequest
    req = HTTPXRequest(connect_timeout=30, read_timeout=60, write_timeout=60, pool_timeout=30)
    upd_req = HTTPXRequest(connect_timeout=30, read_timeout=60, write_timeout=60, pool_timeout=30)
    app = (Application.builder().token(token).request(req).get_updates_request(upd_req)
           .concurrent_updates(True).post_init(post_init).post_shutdown(stop_api).build())
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("pushups", cmd_pushups))
    app.add_handler(CommandHandler("squats", cmd_squats))
    app.add_handler(CommandHandler("pullups", cmd_pullups))
    app.add_handler(CommandHandler("duel", cmd_duel))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("me", cmd_me))
    app.add_handler(CommandHandler("skins", cmd_skins))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO, on_media))
    app.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, on_webapp_data))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app


# ════════════════════════════════════════════════════════════════
#  7. САМОТЕСТ
# ════════════════════════════════════════════════════════════════

def _synthetic_angles(full=10, partial=2, fps=15.0, lo_full=80, lo_part=120, hi=170):
    """Угол сустава: full полных повторений и partial неполных."""
    seq = []

    def rep(low):
        n = int(fps * 1.6)
        for i in range(n):
            seq.append(low + (hi - low) * (0.5 + 0.5 * math.cos(2 * math.pi * i / n)))
    seq.extend([hi + 2.0] * int(fps))
    for _ in range(full):
        rep(lo_full)
    for _ in range(partial):
        rep(lo_part)
    seq.extend([hi + 2.0] * int(fps))
    return seq


def _lm_for(ex, a, w=480, h=270):
    """Синтетические 33 точки: человек выполняет упражнение ex, угол сустава = a."""
    lm = [(0.0, 0.0, 0.0)] * 33
    if ex == "pushup":        # упор лёжа боком
        sh, el = (0.2 * w, 0.5 * h), (0.3 * w, 0.5 * h)
        base = math.atan2(sh[1] - el[1], sh[0] - el[0])
        wr = (el[0] + 0.1 * w * math.cos(base + math.radians(a)),
              el[1] + 0.1 * w * math.sin(base + math.radians(a)))
        pts = dict(sh=sh, el=el, wr=wr, hip=(0.5 * w, 0.52 * h), an=(0.8 * w, 0.55 * h))
        nose = None
    elif ex == "squat":       # присед боком: угол в колене
        L = 0.2 * h
        kn = (0.5 * w, 0.7 * h)
        th = math.radians(-90 + a)
        pts = dict(sh=(kn[0], kn[1] - 2.4 * L), hip=(kn[0], kn[1] - L), kn=kn,
                   an=(kn[0] + L * math.cos(th), kn[1] + L * math.sin(th)))
        nose = None
    else:                     # подтягивание: угол в локте, кисти на перекладине
        L = 0.18 * h
        el = (0.5 * w, 0.5 * h)
        th = math.radians(-90 + a)
        sh = (el[0] + L * math.cos(th), el[1] + L * math.sin(th))
        pts = dict(sh=sh, el=el, wr=(el[0], el[1] - L),
                   hip=(sh[0], sh[1] + 0.2 * h), an=(sh[0], sh[1] + 0.3 * h))
        nose = (sh[0], sh[1] - 0.12 * h)
    for k, p in pts.items():
        lm[SIDES["L"][k]] = (p[0] / w, p[1] / h, 0.99)
        lm[SIDES["R"][k]] = (p[0] / w, p[1] / h, 0.1)
    if nose:
        lm[NOSE] = (nose[0] / w, nose[1] / h, 0.99)
    return lm


class _FakeBackend:
    """Вместо нейросети отдаёт заранее заданные точки: проверка видео-цикла."""

    def __init__(self, ex, angles):
        self.ex, self.angles, self.i = ex, angles, 0

    def landmarks(self, rgb, dt_ms):
        a = self.angles[min(self.i, len(self.angles) - 1)]
        self.i += 1
        return _lm_for(self.ex, a)

    def close(self):
        pass


def selftest(out_dir: str | None = None):
    import tempfile as _t
    global DB_PATH
    fps = 15.0
    cases = {"pushup": (80, 120, 170), "squat": (70, 125, 172), "pullup": (55, 95, 170)}

    # 1. счётчик по синтетическим точкам для каждого упражнения
    for ex, (lo_full, lo_part, hi) in cases.items():
        ang = _synthetic_angles(10, 2, fps, lo_full, lo_part, hi)
        frames = [(i / fps, _lm_for(ex, a), 480, 270) for i, a in enumerate(ang)]
        res = count_from_landmarks(frames, fps, len(ang) / fps, ex)
        assert (res.reps, res.partial) == (10, 2), (ex, res.reps, res.partial)
        print(f"{ex}: {res.reps} полных, {res.partial} неполных  OK")

    # 2. неподходящая поза отклоняется
    ang = _synthetic_angles(10, 2, fps)
    pushup_frames = [(i / fps, _lm_for("pushup", a), 480, 270) for i, a in enumerate(ang)]
    for wrong in ("pullup", "squat"):
        try:
            count_from_landmarks(pushup_frames, fps, 10, wrong)
            raise AssertionError(f"отжимания засчитаны как {wrong}")
        except VideoError as e:
            assert e.code == "no_pose"
    stand = []
    for i, a in enumerate(ang):
        lm = [(0.0, 0.0, 0.0)] * 33
        lm[SIDES["L"]["sh"]] = (0.5, 0.2, 0.99); lm[SIDES["L"]["el"]] = (0.5, 0.35, 0.99)
        lm[SIDES["L"]["wr"]] = (0.5, 0.5, 0.99); lm[SIDES["L"]["hip"]] = (0.5, 0.55, 0.99)
        lm[SIDES["L"]["an"]] = (0.5, 0.95, 0.99)
        stand.append((i / fps, lm, 480, 270))
    try:
        count_from_landmarks(stand, fps, 10, "pushup")
        raise AssertionError("стоящий человек засчитан как отжимания")
    except VideoError as e:
        assert e.code == "no_pose"
    print("неподходящая поза отклоняется  OK")

    # 2b. настоящий видеофайл + подставная нейросеть: проверяем цикл по кадрам
    try:
        import cv2
        import numpy as np
        vid = os.path.join(_t.mkdtemp(), "synthetic.mp4")
        wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (480, 270))
        for _ in ang:
            wr.write(np.zeros((270, 480, 3), dtype=np.uint8))
        wr.release()
        r2 = analyze_video(vid, "pushup", backend=_FakeBackend("pushup", ang))
        assert (r2.reps, r2.partial) == (10, 2), (r2.reps, r2.partial)
        print(f"видео-цикл (mp4, {r2.duration:.0f} с): {r2.reps} + {r2.partial}  OK")
    except ImportError:
        print("видео-цикл пропущен: нет opencv")

    # 3. БД, достижения, дуэль, рейтинг
    DB_PATH = os.path.join(_t.mkdtemp(), "test.db")
    init_db()

    class U:  # заглушка Telegram-пользователя
        def __init__(s, i, n, l): s.id, s.full_name, s.username, s.language_code = i, n, n.lower(), l
    ensure_user(U(1, "Аня", "ru")); ensure_user(U(2, "Bob", "en"))
    now = time.time()
    x("INSERT INTO results(user_id,reps,partial,duration,ts,ex) VALUES(1,32,1,60,?,'pushup')", (now,))
    x("INSERT INTO results(user_id,reps,partial,duration,ts,ex) VALUES(2,25,0,55,?,'pushup')", (now,))
    x("INSERT INTO results(user_id,reps,partial,duration,ts,ex) VALUES(1,40,2,70,?,'squat')", (now,))
    x("INSERT INTO duels(code,creator,opponent,c_reps,o_reps,status,created,ex) VALUES('ABC123',1,2,32,25,'done',?,'pushup')", (now,))
    assert duel_wins(1) == 1 and duel_wins(2) == 0
    total, best, sets = user_stats(1)
    assert total == 72 and user_best(1, "squat") == 40 and user_best(1, "pushup") == 32
    new = unlock_achievements(1, 32, total, duel_wins(1))
    assert set(new) == {"first", "set10", "set25", "duel_win", "total100"} - {"total100"} or "first" in new, new
    assert unlock_achievements(1, 32, total, 1) == []
    text, _ = top_view("ru", "week", "pushup")
    assert "Аня" in text and "32" in text and "Bob" in text
    text, _ = top_view("en", "all", "squat")
    assert "40" in text and "Bob" not in text
    assert "🦵40" in profile_view("ru", 1)
    print("БД, достижения, дуэль, рейтинг, профиль  OK")

    # 4. карточки всех скинов и упражнений
    out = Path(out_dir or tempfile.gettempdir())
    out.mkdir(parents=True, exist_ok=True)
    for sid in SKINS:
        (out / f"card_{sid}.png").write_bytes(render_card("Аня Иванова", 32, 58, 91, 3, sid, "repduel_bot").read())
    for ex in EXERCISES:
        (out / f"card_ex_{ex}.png").write_bytes(
            render_card("Аня Иванова", 24, 61, 88, 4, "neon", "repduel_bot", ex).read())
    print(f"карточки сохранены в {out}  OK")

    # 5. приложение собирается
    build_app("123456:TEST-TOKEN")
    print("приложение Telegram собирается  OK")
    print("\nВсё в порядке. Теперь python bot.py с настоящим BOT_TOKEN.")


def main():
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    if "--selftest" in sys.argv:
        idx = sys.argv.index("--selftest")
        out = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        return selftest(out)
    token = os.environ.get("BOT_TOKEN", "").strip()
    if not token:
        print("Нет токена. Получи его у @BotFather и задай переменную BOT_TOKEN:\n"
              "  Windows (cmd):  set BOT_TOKEN=123456:ABC...\n"
              "  Mac/Linux:      export BOT_TOKEN=123456:ABC...")
        sys.exit(1)
    init_db()
    app = build_app(token)
    log.info("RepDuel запущен. Остановить: Ctrl+C")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
