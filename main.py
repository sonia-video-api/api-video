import os
import json
import uuid
import base64
import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_NAME = "MonAPI Video"
DB_PATH = os.getenv("DB_PATH", "/tmp/jobs.db")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://api-video-4ajs.onrender.com")

# Fal.ai
FAL_API_KEY = os.getenv("FAL_API_KEY")
FAL_MODEL = os.getenv("FAL_MODEL", "fal-ai/fast-video")

# Replicate (optionnel)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

DEFAULT_PROVIDER = "fal" if FAL_API_KEY else ("replicate" if REPLICATE_API_TOKEN else None)

app = FastAPI(
    title=APP_NAME,
    description="API de génération de vidéos IA — Render + GPU externe Fal.ai",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# DB (SQLite)
# ---------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            api_key_id TEXT NOT NULL,
            status TEXT DEFAULT 'queued',
            provider TEXT,
            prompt TEXT NOT NULL,
            params TEXT,
            result_url TEXT,
            error TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


@app.on_event("startup")
def startup():
    init_db()


# ---------------------------
# Auth
# ---------------------------
def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key():
    raw = "mak_" + base64.urlsafe_b64encode(secrets.token_bytes(24)).decode().rstrip("=")
    return raw, hash_key(raw)


def verify_key(key: str) -> Optional[dict]:
    if not key or not key.startswith("mak_"):
        return None
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM api_keys WHERE key_hash = ?", (hash_key(key),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------
# Schemas
# ---------------------------
class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=2000)
    seconds: int = Field(default=5, ge=1, le=6)
    fps: int = Field(default=16, ge=8, le=24)
    width: int = Field(default=768, ge=256, le=1280)
    height: int = Field(default=432, ge=256, le=720)
    seed: Optional[int] = None
    provider: Optional[str] = None


class JobResponse(BaseModel):
    id: str
    status: str
    message: str


class JobStatus(BaseModel):
    id: str
    status: str
    prompt: str
    provider: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ---------------------------
# Fal.ai — génération réelle
# ---------------------------
def fal_generate_video(prompt: str, seconds: int, fps: int, width: int, height: int, seed: Optional[int]) -> str:
    """Appel synchrone à Fal.ai pour générer une vraie vidéo."""
    if not FAL_API_KEY:
        raise RuntimeError("FAL_API_KEY non configuré sur Render")

    url = f"https://fal.run/{FAL_MODEL}"

    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "duration": seconds,
        "fps": fps,
        "width": width,
        "height": height,
    }
    if seed is not None:
        payload["seed"] = seed

    r = requests.post(url, json=payload, headers=headers, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"Fal.ai erreur {r.status_code}: {r.text}")

    data = r.json()

    # Fal retourne selon le modèle : video_url, video, ou outputs[0].url
    video_url = data.get("video_url") or data.get("video")
    if not video_url and isinstance(data.get("outputs"), list) and data["outputs"]:
        video_url = data["outputs"][0].get("url")
    if not video_url and isinstance(data.get("video"), dict):
        video_url = data["video"].get("url")

    if not video_url:
        raise RuntimeError(f"Fal.ai n'a pas retourné d'URL vidéo: {json.dumps(data)[:500]}")

    return video_url


# ---------------------------
# Tâche de fond
# ---------------------------
def process_video_job(job_id: str, prompt: str, params: dict, provider: str):
    conn = get_db()
    try:
        conn.execute(
            "UPDATE jobs SET status='processing', updated_at=? WHERE id=?",
            (now_iso(), job_id)
        )
        conn.commit()

        if provider == "fal":
            video_url = fal_generate_video(
                prompt=prompt,
                seconds=params["seconds"],
                fps=params["fps"],
                width=params["width"],
                height=params["height"],
                seed=params.get("seed"),
            )
        else:
            raise RuntimeError(f"Provider '{provider}' non supporté. Utilisez 'fal'.")

        conn.execute(
            "UPDATE jobs SET status='completed', result_url=?, updated_at=? WHERE id=?",
            (video_url, now_iso(), job_id)
        )
        conn.commit()

    except Exception as e:
        conn.execute(
            "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?",
            (str(e), now_iso(), job_id)
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def home():
    return {
        "name": APP_NAME,
        "version": "3.0.0",
        "provider_defaut": DEFAULT_PROVIDER or "AUCUN (configurez FAL_API_KEY)",
        "fal_configure": bool(FAL_API_KEY),
        "model": FAL_MODEL,
        "tarifs": {
            "video_5s": "~0.25$ (Fal.ai)",
            "video_10s": "~0.50$ (Fal.ai)",
        },
        "endpoints": {
            "creer_cle": "POST /admin/create-key",
            "generer_video": "POST /v1/video  (Authorization: Bearer mak_xxx)",
            "statut_job": "GET /v1/jobs/{id}",
            "liste_jobs": "GET /v1/jobs",
        },
        "documentation": f"{PUBLIC_BASE_URL}/docs",
    }


@app.post("/admin/create-key")
def create_key(name: Optional[str] = "default"):
    raw, hashed = generate_api_key()
    conn = get_db()
    key_id = "key_" + uuid.uuid4().hex
    conn.execute(
        "INSERT INTO api_keys (id, key_hash, name, created_at) VALUES (?, ?, ?, ?)",
        (key_id, hashed, name, now_iso())
    )
    conn.commit()
    conn.close()
    return {
        "api_key": raw,
        "key_id": key_id,
        "name": name,
        "message": "Conservez cette clé, elle ne sera plus affichée !",
        "utilisation": f"Authorization: Bearer {raw}",
    }


@app.post("/v1/video", response_model=JobResponse)
def create_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Header Authorization manquant. Format: Bearer mak_xxx")
    key = authorization.replace("Bearer ", "").strip()
    user = verify_key(key)
    if not user:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    provider = request.provider or DEFAULT_PROVIDER
    if not provider:
        raise HTTPException(
            status_code=503,
            detail="Aucun provider GPU configuré. Ajoutez FAL_API_KEY dans Render > Environment."
        )

    params = {
        "seconds": request.seconds,
        "fps": request.fps,
        "width": request.width,
        "height": request.height,
        "seed": request.seed,
    }

    job_id = "job_" + uuid.uuid4().hex[:8]
    conn = get_db()
    conn.execute(
        "INSERT INTO jobs (id, api_key_id, status, provider, prompt, params, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (job_id, user["id"], "queued", provider, request.prompt, json.dumps(params), now_iso(), now_iso())
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(process_video_job, job_id, request.prompt, params, provider)

    return JobResponse(
        id=job_id,
        status="queued",
        message=f"Job créé ! Vérifiez le statut sur /v1/jobs/{job_id} dans 30-90 secondes."
    )


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Header Authorization manquant")
    key = authorization.replace("Bearer ", "").strip()
    if not verify_key(key):
        raise HTTPException(status_code=401, detail="Clé API invalide")

    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Job introuvable")

    job = dict(row)
    return JobStatus(
        id=job["id"],
        status=job["status"],
        prompt=job["prompt"],
        provider=job.get("provider"),
        result_url=job.get("result_url"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@app.get("/v1/jobs")
def list_jobs(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Header Authorization manquant")
    key = authorization.replace("Bearer ", "").strip()
    user = verify_key(key)
    if not user:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    conn = get_db()
    rows = conn.execute(
        "SELECT id, status, prompt, provider, result_url, created_at, updated_at FROM jobs WHERE api_key_id = ? ORDER BY created_at DESC LIMIT 50",
        (user["id"],)
    ).fetchall()
    conn.close()
    return {"jobs": [dict(r) for r in rows], "total": len(rows)}
