"""Generate a LiveKit access token for connecting to a room."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livekit.api import AccessToken, VideoGrants

API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
ROOM = sys.argv[1] if len(sys.argv) > 1 else "voxlabs-room"
IDENTITY = sys.argv[2] if len(sys.argv) > 2 else "user"

token = AccessToken(API_KEY, API_SECRET)
token.with_identity(IDENTITY)
token.with_grants(VideoGrants(room_join=True, room=ROOM))

print(token.to_jwt())
