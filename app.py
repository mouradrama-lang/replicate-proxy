import asyncio
import json
import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

REPLICATE_API = "https://api.replicate.com/v1"
TOKEN = os.environ.get("REPLICATE_TOKEN", "")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

app = FastAPI(title="replicate-proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def get_token(request: Request) -> str:
    # Priorité à la variable d'environnement
    if TOKEN:
        return TOKEN
    # Sinon, lit depuis le header Authorization
    h = request.headers.get("Authorization", "")
    if h.startswith("Bearer "):
        return h[7:].strip()
    return h.strip()


async def replicate_call(method: str, path: str, body=None, token: str = ""):
    url = f"{REPLICATE_API}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": UA,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15)) as client:
        if method == "GET":
            r = await client.get(url, headers=headers)
        else:
            r = await client.post(url, headers=headers, json=body)
        return r


@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok", "service": "replicate-proxy", "upstream": REPLICATE_API}


@app.get("/v1/models")
async def list_models():
    """Liste des modèles vidéo populaires sur Replicate."""
    models = [
        {"id": "lightricks/ltx-video", "name": "LTX-Video (rapide)"},
        {"id": "wan-video/wan-2.5-t2v-fast", "name": "Wan 2.5 T2V Fast"},
        {"id": "minimax/video-01", "name": "MiniMax Video-01"},
        {"id": "tencent/hunyuan-video", "name": "HunyuanVideo"},
        {"id": "stability-ai/stable-video-diffusion", "name": "Stable Video Diffusion"},
    ]
    return {"object": "list", "data": models}


@app.post("/v1/videos")
async def create_video(request: Request):
    """
    Crée une prédiction Replicate.
    Body attendu : { model, prompt, aspect_ratio, num_frames, fps, image (optionnel) }
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"JSON invalide : {str(e)}"}, status_code=400)

    token = get_token(request)
    if not token:
        return JSONResponse({"error": "Token Replicate manquant"}, status_code=401)

    model = body.get("model", "lightricks/ltx-video")
    prompt = body.get("prompt", "")

    if not prompt:
        return JSONResponse({"error": "Prompt manquant"}, status_code=400)

    # Construction des paramètres
    input_data = {
        "prompt": prompt,
        "num_frames": body.get("num_frames", 97),
        "fps": body.get("fps", 24),
    }
    if body.get("aspect_ratio"):
        input_data["aspect_ratio"] = body["aspect_ratio"]
    if body.get("image"):
        input_data["image"] = body["image"]
    if body.get("num_inference_steps"):
        input_data["num_inference_steps"] = body["num_inference_steps"]

    try:
        # Crée la prédiction
        if "/" in model and ":" not in model:
            # Format owner/name — on utilise l'endpoint /models/{owner}/{name}/predictions
            path = f"/models/{model}/predictions"
            r = await replicate_call("POST", path, {"input": input_data}, token)
        else:
            # Format avec version hash
            r = await replicate_call("POST", "/predictions", {"version": model, "input": input_data}, token)

        if r.status_code not in (200, 201):
            return JSONResponse(
                {"error": f"Replicate HTTP {r.status_code}", "detail": r.text[:300]},
                status_code=r.status_code,
            )

        data = r.json()
        pred_id = data.get("id")
        if not pred_id:
            return JSONResponse({"error": "Pas d'ID de prédiction"}, status_code=502)

        return {
            "id": pred_id,
            "status": data.get("status", "starting"),
            "urls": data.get("urls", {}),
        }
    except Exception as e:
        return JSONResponse({"error": f"Erreur : {str(e)}"}, status_code=502)


@app.get("/v1/videos/{job_id}")
async def get_video_status(job_id: str, request: Request):
    token = get_token(request)
    if not token:
        return JSONResponse({"error": "Token Replicate manquant"}, status_code=401)

    try:
        r = await replicate_call("GET", f"/predictions/{job_id}", None, token)
        if r.status_code != 200:
            return JSONResponse(
                {"error": f"Replicate HTTP {r.status_code}", "detail": r.text[:200]},
                status_code=r.status_code,
            )

        data = r.json()
        output = data.get("output")
        video_url = None
        if isinstance(output, list) and len(output) > 0:
            video_url = output[0]
        elif isinstance(output, str):
            video_url = output

        return {
            "id": job_id,
            "status": data.get("status", "unknown"),
            "progress": _compute_progress(data.get("status", "")),
            "error": data.get("error"),
            "video_url": video_url,
            "output": output,
        }
    except Exception as e:
        return JSONResponse({"error": f"Erreur : {str(e)}"}, status_code=502)


def _compute_progress(status: str) -> int:
    return {
        "starting": 5,
        "processing": 50,
        "succeeded": 100,
        "failed": 0,
        "canceled": 0,
    }.get(status, 0)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
