# Подключение Mini App к bot.py

## 0. Хостинг (обязательно)
Mini App открывается только по **HTTPS**. Положи в одну папку на хостинге:
- `index.html`
- `pose_landmarker_lite.task` (скачай один раз:
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)

Бесплатные варианты: GitHub Pages, Cloudflare Pages, Netlify, Vercel.
Для теста можно `python -m http.server 8000` + `cloudflared tunnel --url http://localhost:8000`.

Задай переменную: `WEBAPP_URL=https://твой-домен/index.html`

## 1. Вынеси финал из on_media
В `on_media` всё, что идёт после блока `# ── сохраняем результат` и до конца функции,
перенеси в новую функцию:

```python
async def finish(update, ctx, res: RepResult, ex: str, mode: dict, code, status=None):
    msg, user = update.effective_message, update.effective_user
    lang = ensure_user(user)
    # ... весь код от "total_before, _b, _s = user_stats(user.id)" до конца ...
```
В `on_media` в конце вызови: `await finish(update, ctx, res, ex, mode, code, status)`.
Внутри `finish` строку `await status.delete()` оберни в `if status:`.

## 2. Добавь в bot.py

```python
import json
from telegram import KeyboardButton, ReplyKeyboardMarkup, WebAppInfo

WEBAPP_URL = os.environ.get("WEBAPP_URL", "")

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
        if float(d.get("valid_ratio", 1)) < 0.25:
            raise ValueError("low quality")
        # антиабьюз: не быстрее 0.3 с на повтор
        if reps / duration > 3.0:
            raise ValueError("too fast")
    except Exception:
        return await msg.reply_text("⚠️")
    res = RepResult(reps, partial, duration, float(d.get("valid_ratio", 1)), 0)

    mode = ctx.user_data.get("mode") or {"kind": "solo"}
    code = d.get("duel") or None
    if code and mode.get("code") != code:
        # дуэль открыта из ссылки: определяем роль по базе
        row = q("SELECT * FROM duels WHERE code=?", (code,), one=True)
        if row and row["creator"] == update.effective_user.id and row["status"] == "waiting":
            mode = {"kind": "duel_create", "code": code}
        elif row and row["status"] == "open":
            mode = {"kind": "duel_join", "code": code}
        else:
            mode, code = {"kind": "solo"}, None
    await finish(update, ctx, res, ex, mode, code)
```

## 3. Регистрация и кнопки
В `build_app` добавь **до** `MessageHandler(filters.TEXT...)`:

```python
app.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, on_webapp_data))
```

В `_choose_ex`, в `on_button` (ветка `ex:`) и в ветках дуэли `duel_new` / `join:` добавь к ответу
`reply_markup=kb_live(lang, ex, code)` (если не `None`). Для обычного упражнения `code=""`.

Пример для `_choose_ex`:

```python
await update.effective_message.reply_text(
    howto_text(lang, ex) + "\n\n" + t(lang, "send_video"),
    parse_mode=ParseMode.HTML, reply_markup=kb_live(lang, ex))
```
Видео по-прежнему можно слать как раньше: старый путь остаётся запасным.

## Ограничения
- `sendData` работает только при открытии через reply-кнопку (не inline) и закрывает Mini App после отправки.
- Результат считается на телефоне, поэтому его можно подделать. Пока это ок для друзей, но для публичного
  рейтинга лучше оставить проверку по видео (старый путь) или записывать видео в Mini App и грузить на сервер.
