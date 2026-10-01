"""Upload session recordings to MinIO (S3-compatible object storage).

The local ``recordings/`` flow is unchanged — after ``recording.finalize``
builds a session MP3, it is uploaded here in a background thread so the
download response is never delayed.

Config (all optional):
    MINIO_ENABLED=1
    MINIO_ENDPOINT=http://127.0.0.1:9000
    MINIO_ACCESS_KEY=minioadmin
    MINIO_SECRET_KEY=(falls back to MINIO_ROOT_PASSWORD, then minioadmin)
    MINIO_BUCKET=recordings
    MINIO_PUBLIC_URL=http://127.0.0.1:9000   (base for playback links)

The ``recordings`` bucket is created public-read by
``infra/minio-init``, so uploaded MP3s play straight from their URL.
Every function is best-effort: MinIO down → log + skip, never raise.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger("ruchi.minio")


def enabled() -> bool:
    if os.getenv("MINIO_ENABLED", "1").lower() in ("0", "false", "no", "off", ""):
        return False
    return True


def endpoint() -> str:
    return os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000").rstrip("/")


def bucket() -> str:
    return os.getenv("MINIO_BUCKET", "recordings")


def public_url(key: str) -> str:
    base = os.getenv("MINIO_PUBLIC_URL", endpoint()).rstrip("/")
    return f"{base}/{bucket()}/{key.lstrip('/')}"


def _client():
    import boto3
    from botocore.config import Config

    secret = (
        os.getenv("MINIO_SECRET_KEY")
        or os.getenv("MINIO_ROOT_PASSWORD")
        or "minioadmin"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint(),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=secret,
        config=Config(
            signature_version="s3v4",
            connect_timeout=3,
            read_timeout=15,
            retries={"max_attempts": 2},
        ),
    )


def upload_file(local: str | Path, key: str | None = None) -> Optional[str]:
    """Upload ``local`` to the bucket. Returns the public URL or ``None``."""
    if not enabled():
        return None
    path = Path(local)
    if not path.is_file():
        return None
    key = key or path.name
    try:
        _client().upload_file(
            str(path), bucket(), key, ExtraArgs={"ContentType": "audio/mpeg"}
        )
    except Exception as exc:  # noqa: BLE001 - never break the voice loop
        log.warning("minio upload failed for %s: %s", path.name, exc)
        return None
    url = public_url(key)
    log.info("minio: uploaded %s → %s", path.name, url)
    return url


def upload_async(local: str | Path, key: str | None = None) -> None:
    """Fire-and-forget upload in a daemon thread."""
    if not enabled():
        return
    threading.Thread(
        target=upload_file, args=(local, key), name="minio-upload", daemon=True
    ).start()


__all__ = ["enabled", "endpoint", "bucket", "public_url", "upload_file", "upload_async"]
