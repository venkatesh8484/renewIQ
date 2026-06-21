"""
Redis-backed response cache middleware.

Caches GET responses (market data, reports) for a configurable TTL.
POST /chat is NOT cached — each query triggers a fresh agent run.

If Redis is unavailable (e.g., CI, local without docker compose),
the middleware is a no-op: requests pass through normally.

Cache key: method + path + sorted query string
TTL: 900s (15 min) for market data, 3600s (1h) for reports
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger(__name__)

# Paths to cache and their TTLs (seconds)
_CACHE_RULES: dict[str, int] = {
    "/reports": 3600,     # report content is immutable once generated
}

# Market data TTL — applied to any path containing "market"
_MARKET_TTL = 900


def _should_cache(method: str, path: str) -> tuple[bool, int]:
    """Return (should_cache, ttl_seconds)."""
    if method != "GET":
        return False, 0
    for prefix, ttl in _CACHE_RULES.items():
        if path.startswith(prefix):
            return True, ttl
    if "market" in path:
        return True, _MARKET_TTL
    return False, 0


def _cache_key(request: Request) -> str:
    raw = f"{request.method}:{request.url.path}:{str(sorted(request.query_params.items()))}"
    return "renewiq:cache:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


class RedisCacheMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that caches GET responses in Redis.
    Gracefully degrades to passthrough when Redis is unavailable.
    """

    def __init__(self, app, redis_url: str = "redis://localhost:6379"):
        super().__init__(app)
        self._redis = None
        self._redis_url = redis_url
        self._connect(redis_url)

    def _connect(self, url: str) -> None:
        try:
            import redis
            client = redis.from_url(url, decode_responses=False, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            logger.info(f"[Cache] Redis connected at {url}")
        except Exception as exc:
            logger.warning(f"[Cache] Redis unavailable ({exc}) — cache disabled")
            self._redis = None

    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        should_cache, ttl = _should_cache(request.method, request.url.path)

        if not should_cache or self._redis is None:
            return await call_next(request)

        key = _cache_key(request)

        # Cache hit
        try:
            cached = self._redis.get(key)
            if cached:
                data = json.loads(cached)
                logger.debug(f"[Cache] HIT {key[:20]}")
                return Response(
                    content=data["body"],
                    status_code=data["status_code"],
                    headers={**data["headers"], "X-Cache": "HIT"},
                    media_type=data["media_type"],
                )
        except Exception as exc:
            logger.warning(f"[Cache] Redis read error: {exc}")

        # Cache miss — call next and store response
        response = await call_next(request)

        if response.status_code == 200:
            try:
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk

                self._redis.setex(
                    key,
                    ttl,
                    json.dumps({
                        "body": body.decode("utf-8", errors="replace"),
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "media_type": response.media_type,
                    }),
                )
                logger.debug(f"[Cache] STORED {key[:20]} ttl={ttl}s")

                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers={**dict(response.headers), "X-Cache": "MISS"},
                    media_type=response.media_type,
                )
            except Exception as exc:
                logger.warning(f"[Cache] Redis write error: {exc}")

        return response
