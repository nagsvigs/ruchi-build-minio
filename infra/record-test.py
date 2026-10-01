#!/usr/bin/env python3
"""Smoke-test for the local LiveKit Egress -> MinIO pipeline.

Starts a **Web Egress** (records a web page, needs no room participant)
so you can confirm recordings actually land in MinIO. Uses only the
Python standard library: builds a LiveKit JWT (HS256) and calls the
LiveKit server's twirp Egress API.

Usage (server stack must be running):
    python3 infra/record-test.py                 # records https://example.com
    python3 infra/record-test.py <url> <name.mp4>
    python3 infra/record-test.py --list
    python3 infra/record-test.py --stop <egress_id>

Then watch it finish:
    docker compose -f infra/docker-compose.yaml logs -f egress
    open http://127.0.0.1:9001  (minioadmin / minioadmin) -> recordings
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sys
import time
import urllib.error
import urllib.request

LIVEKIT_HTTP = "http://127.0.0.1:7880"
API_KEY = "devkey"
API_SECRET = "secretsecretsecretsecretsecretsecret12"
TWIRP = f"{LIVEKIT_HTTP}/twirp/livekit.Egress"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def make_token(ttl: int = 600) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": API_KEY,
        "sub": API_KEY,
        "nbf": now - 10,
        "exp": now + ttl,
        # Permissions required to drive the Egress API.
        "video": {"roomRecord": True, "roomAdmin": True},
    }
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    sig = hmac.new(API_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + _b64url(sig)


def call(method: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{TWIRP}/{method}",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {make_token()}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()[:500]}", file=sys.stderr)
        raise


def egress_id(info: dict) -> str:
    return info.get("egressId") or info.get("egress_id") or ""


def main() -> int:
    args = sys.argv[1:]

    if args and args[0] == "--list":
        print(json.dumps(call("ListEgress", {}), indent=2))
        return 0

    if args and args[0] == "--stop":
        if len(args) < 2:
            print("usage: record-test.py --stop <egress_id>", file=sys.stderr)
            return 2
        print(json.dumps(call("StopEgress", {"egressId": args[1]}), indent=2))
        return 0

    url = args[0] if args else "https://example.com"
    filename = args[1] if len(args) > 1 else "smoke-test.mp4"

    print(f"Starting web egress: {url} -> {filename}")
    info = call("StartWebEgress", {"url": url, "fileOutputs": [{"filepath": filename}]})
    print(json.dumps(info, indent=2))
    eid = egress_id(info)
    print()
    print(f"egress_id : {eid}")
    print(f"status    : {info.get('status')}")
    print(f"console   : http://127.0.0.1:9001  (minioadmin / minioadmin) -> recordings")
    print(f"direct    : http://127.0.0.1:9000/recordings/{filename}")
    print(f"watch     : docker compose -f infra/docker-compose.yaml logs -f egress")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
