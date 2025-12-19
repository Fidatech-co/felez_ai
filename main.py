from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional, Iterable

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from number_reader import read_gauges
from screen_detector import extract_screen

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OCR API", version="1.0.0")



class GaugeReadingsOut(BaseModel):
    fuel_level: Optional[float] = None
    cool_temp: Optional[float] = None
    sys_volt: Optional[float] = None
    oil_press: Optional[float] = None
    mach_hours: Optional[float] = None
    rpm: Optional[int] = None

    screen_file: Optional[str] = None

_num_re = re.compile(r"\d+(?:\.\d+)?")

def pick_number(texts: Iterable[str]) -> Optional[float]:
    for t in texts or []:
        m = _num_re.search(str(t))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                continue
    return None

def _save_upload(upload: UploadFile, suffix: str = ".jpg") -> Path:
    file_id = uuid.uuid4().hex
    out = DATA_DIR / f"{file_id}{suffix}"
    with out.open("wb") as f:
        f.write(upload.file.read())
    return out


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/gauges/read", response_model=GaugeReadingsOut)
async def gauges_read(
    image: UploadFile = File(...),
    save_screen: bool = Form(False),
):
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported image type")

    raw_path = _save_upload(image)

    screen_path = None
    try:
        screen_out = None
        if save_screen:
            screen_out = raw_path.with_name(f"{raw_path.stem}_screen.png")

        screen_path = extract_screen(raw_path, screen_out)

        raw_readings = read_gauges(screen_path)

        out = GaugeReadingsOut(
            fuel_level=pick_number(raw_readings.get("fuel_level", [])),
            cool_temp=pick_number(raw_readings.get("cool_temp", [])),
            sys_volt=pick_number(raw_readings.get("sys_volt", [])),
            oil_press=pick_number(raw_readings.get("oil_press", [])),
            mach_hours=pick_number(raw_readings.get("mach_hours", [])),
        )

        rpm_val = pick_number(raw_readings.get("rpm", []))
        out.rpm = int(rpm_val) if rpm_val is not None else None

        if save_screen and screen_path:
            out.screen_file = str(screen_path)

        return out

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        if screen_path and not save_screen:
            try:
                screen_path.unlink(missing_ok=True)
            except Exception:
                pass