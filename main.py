from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
import os
import json
import uuid
import hashlib
import secrets
import base64
import httpx
from datetime import datetime, timezone
import sqlite3

app = FastAPI(
    title="MonAPI Video - 100% Online",
    description="API de génération de vidéos IA sur Render + GPU externe (Fal.ai / Replicate)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ CONFIGURATION ============
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
FAL_API_KEY = os.getenv("FAL_API_KEY")

PRICING = {
    "replicate": {"per_second": 0.08, "min": 0.50},
    "fal": {"per_second": 0.05, "min": 0.30},
}

DEFAULT_PROVIDER = "fal" if FAL_API_KEY else ("replicate" if REPLICATE_API_TOKEN else None)

DB_PATH = os.getenv("DB_PATH", "/tmp/jobs.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            api_key_id TEXT NOT NULL,
            status TEXT DEFAULT 'queued',
            provider TEXT,
            prompt TEXT NOT NULL,
            params TEXT,
            cost_estimate REAL,
            result_url TEXT,
            error TEXT,
            created_at TEXT,
            updated_at TEXT,
            provider_job_id TEXT,
            status_url TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT UNIQUE NOT NULL,
            name TEXT,
            credits_spent REAL DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup():
    init_db()

# ============ AUTH ============
def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:32]

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

# ============ MODÈLES ============
class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=1000)
    seconds: int = Field(default=6, ge=2, le=10)
    fps: int = Field(default=16, ge=8, le=24)
    width: int = Field(default=768, ge=256, le=1024)
    height: int = Field(default=432, ge=256, le=768)
    provider: Optional[Literal["fal", "replicate"]] = None

class JobResponse(BaseModel):
    id: str
    status: str
    estimated_cost_usd: float
    message: str

class JobStatus(BaseModel):
    id: str
    status: str
    prompt: str
    provider: Optional[str] = None
    cost_usd: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    progress: int = 0

# ============ GÉNÉRATION EXTERNE ============
async def generate_with_fal(prompt: str, params: dict) -> dict:
    if not FAL_API_KEY:
        raise Exception("FAL_API_KEY non configuré dans les variables d'environnement Render")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://queue.fal.run/fal-ai/fast-svd-lcm",
            headers={"Authorization": f"Key {FAL_API_KEY}"},
            json={
                "prompt": prompt,
                "num_frames": params["seconds"] * params["fps"],
                "fps": params["fps"],
                "width": params["width"],
                "height": params["height"],
            },
            timeout=30.0
        )
        if response.status_code not in (200, 201):
            raise Exception(f"Fal API error {response.status_code}: {response.text}")
        result = response.json()
        return {
            "provider_job_id": result.get("request_id"),
            "status_url": result.get("status_url") or f"https://queue.fal.run/fal-ai/fast-svd-lcm/requests/{result.get('request_id')}/status",
            "estimated_cost": params["seconds"] * PRICING["fal"]["per_second"]
        }

async def generate_with_replicate(prompt: str, params: dict) -> dict:
    if not REPLICATE_API_TOKEN:
        raise Exception("REPLICATE_API_TOKEN non configuré dans les variables d'environnement Render")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"},
            json={
                "version": "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                "input": {
                    "prompt": prompt,
                    "num_frames": params["seconds"] * params["fps"],
                    "fps": params["fps"],
                    "width": params["width"],
                    "height": params["height"]
                }
            },
            timeout=30.0
        )
        if response.status_code != 201:
            raise Exception(f"Replicate error {response.status_code}: {response.text}")
        result = response.json()
        return {
            "provider_job_id": result["id"],
            "status_url": result["urls"]["get"],
            "estimated_cost": PRICING["replicate"]["min"] + (params["seconds"] * PRICING["replicate"]["per_second"])
        }

async def check_fal_status(status_url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            status_url,
            headers={"Authorization": f"Key {FAL_API_KEY}"},
            timeout=10.0
        )
        if response.status_code == 200:
            data = response.json()
            video = data.get("video")
            video_url = video.get("url") if isinstance(video, dict) else video
            return {
                "status": "completed" if video_url else "processing",
                "video_url": video_url,
            }
        return {"status": "processing"}

async def check_replicate_status(status_url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            status_url,
            headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"},
            timeout=10.0
        )
        if response.status_code == 200:
            data = response.json()
            st = data.get("status")
            return {
                "status": "completed" if st == "succeeded" else ("failed" if st == "failed" else "processing"),
                "video_url": data.get("output") if isinstance(data.get("output"), str) else None,
                "error": data.get("error")
            }
        return {"status": "processing"}

# ============ TÂCHE DE FOND ============
async def process_video_job(job_id: str, prompt: str, params: dict, provider: str, api_key_id: str):
    conn = get_db()
    try:
        conn.execute(
            "UPDATE jobs SET status='processing', updated_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), job_id)
        )
        conn.commit()

        if provider == "fal":
            result = await generate_with_fal(prompt, params)
        else:
            result = await generate_with_replicate(prompt, params)

        conn.execute(
            "UPDATE jobs SET provider_job_id=?, status_url=?, cost_estimate=?, updated_at=? WHERE id=?",
            (result["provider_job_id"], result["status_url"], result["estimated_cost"],
             datetime.now(timezone.utc).isoformat(), job_id)
        )
        conn.commit()

        # Polling jusqu'à complétion (max 5 min)
        import asyncio
        for _ in range(60):
            await asyncio.sleep(5)
            if provider == "fal":
                status = await check_fal_status(result["status_url"])
            else:
                status = await check_replicate_status(result["status_url"])

            if status["status"] == "completed":
                conn.execute(
                    "UPDATE jobs SET status='completed', result_url=?, updated_at=? WHERE id=?",
                    (status.get("video_url"), datetime.now(timezone.utc).isoformat(), job_id)
                )
                conn.commit()
                return
            elif status["status"] == "failed":
                raise Exception(status.get("error", "Génération échouée"))

        raise Exception("Timeout: génération trop longue (>5 min)")

    except Exception as e:
        conn.execute(
            "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?",
            (str(e), datetime.now(timezone.utc).isoformat(), job_id)
        )
        conn.commit()
    finally:
        conn.close()

# ============ ENDPOINTS ============
@app.get("/")
def home():
    return {
        "name": "MonAPI Video - 100% Online",
        "version": "2.0.0",
        "mode": "Render (API) + GPU externe (Fal.ai / Replicate)",
        "tarifs": {
            "video_5s": "~0.25$ (Fal.ai)",
            "video_10s": "~0.50$ (Fal.ai)",
        },
        "endpoints": {
            "creer_cle": "POST /admin/create-key",
            "generer_video": "POST /v1/video  (header: Authorization: Bearer mak_xxx)",
            "statut_job": "GET /v1/jobs/{id}",
            "liste_jobs": "GET /v1/jobs",
        },
        "configuration": {
            "fal_configure": bool(FAL_API_KEY),
            "replicate_configure": bool(REPLICATE_API_TOKEN),
            "provider_defaut": DEFAULT_PROVIDER or "AUCUN (configurez FAL_API_KEY ou REPLICATE_API_TOKEN)"
        },
        "documentation": "/docs"
    }

@app.post("/admin/create-key")
def create_key(name: Optional[str] = "default"):
    raw, hashed = generate_api_key()
    conn = get_db()
    key_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO api_keys (id, key_hash, name, created_at) VALUES (?, ?, ?, ?)",
        (key_id, hashed, name, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()
    return {
        "api_key": raw,
        "key_id": key_id,
        "name": name,
        "message": "Conservez cette clé, elle ne sera plus affichée !",
        "utilisation": "Ajoutez le header: Authorization: Bearer " + raw
    }

@app.post("/v1/video", response_model=JobResponse)
async def create_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None)
):
    # Vérification clé API
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Header Authorization manquant. Format: Bearer mak_xxx")
    key = authorization.replace("Bearer ", "").strip()
    user = verify_key(key)
    if not user:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    # Sélection du provider
    provider = request.provider or DEFAULT_PROVIDER
    if not provider:
        raise HTTPException(
            status_code=503,
            detail="Aucun provider GPU configuré. Ajoutez FAL_API_KEY ou REPLICATE_API_TOKEN dans Render > Environment."
        )

    params = {
        "seconds": request.seconds,
        "fps": request.fps,
        "width": request.width,
        "height": request.height,
    }
    cost = request.seconds * PRICING[provider]["per_second"]

    job_id = "job_" + str(uuid.uuid4())[:8]
    conn = get_db()
    conn.execute(
        "INSERT INTO jobs (id, api_key_id, status, provider, prompt, params, cost_estimate, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (job_id, user["id"], "queued", provider, request.prompt,
         json.dumps(params), cost,
         datetime.now(timezone.utc).isoformat(),
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(process_video_job, job_id, request.prompt, params, provider, user["id"])

    return JobResponse(
        id=job_id,
        status="queued",
        estimated_cost_usd=round(cost, 3),
        message=f"Job créé ! Vérifiez le statut sur /v1/jobs/{job_id} dans 30-60 secondes."
    )

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, authorization: Optional[str] = Header(None)):
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
    progress = {"queued": 0, "processing": 50, "completed": 100, "failed": 0}.get(job["status"], 0)
    return JobStatus(
        id=job["id"],
        status=job["status"],
        prompt=job["prompt"],
        provider=job.get("provider"),
        cost_usd=job.get("cost_estimate"),
        result_url=job.get("result_url"),
        error=job.get("error"),
        created_at=job["created_at"],
        progress=progress
    )

@app.get("/v1/jobs")
def list_jobs(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Header Authorization manquant")
    key = authorization.replace("Bearer ", "").strip()
    user = verify_key(key)
    if not user:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    conn = get_db()
    rows = conn.execute(
        "SELECT id, status, prompt, provider, cost_estimate, result_url, created_at FROM jobs WHERE api_key_id = ? ORDER BY created_at DESC LIMIT 50",
        (user["id"],)
    ).fetchall()
    conn.close()
    return {"jobs": [dict(r) for r in rows], "total": len(rows)}
