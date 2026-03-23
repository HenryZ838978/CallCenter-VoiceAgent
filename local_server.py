"""Local HTTP server: static files + LiveKit token API.

Serves the LiveKit playground and test pages on port 18123,
replacing the VPS Nginx + Flask token server.

Usage:
    python local_server.py
"""
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit.api import AccessToken, VideoGrants

API_KEY = "hzai_key"
API_SECRET = "hzai_secret_long_enough_for_production_use_2026"

STATIC_DIR = Path(__file__).parent / "static" / "livekit"

app = FastAPI()


@app.get("/livekit/token")
async def get_token():
    room = f"room-{int(time.time())}"
    identity = f"user-{int(time.time()) % 10000}"
    token = AccessToken(API_KEY, API_SECRET)
    token.with_identity(identity)
    token.with_grants(VideoGrants(
        room_join=True, room=room,
        can_publish=True, can_subscribe=True,
    ))
    return JSONResponse({"token": token.to_jwt(), "room": room})


@app.get("/")
async def root():
    return RedirectResponse("/livekit/")


app.mount("/livekit", StaticFiles(directory=str(STATIC_DIR), html=True), name="livekit")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18123)
