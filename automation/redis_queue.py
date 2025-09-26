"""Minimal Redis queue wrapper for automation workers."""

from __future__ import annotations

import json
from typing import Any, Optional

import redis

from .config import config


class RedisQueue:
    def __init__(self, queue_name: str) -> None:
        self.queue_name = queue_name
        self.client = redis.Redis.from_url(config.redis_url, decode_responses=True)

    def push(self, payload: Any) -> None:
        data = json.dumps(payload)
        self.client.rpush(self.queue_name, data)

    def pop(self, block: bool = True, timeout: int = 5) -> Optional[Any]:
        if block:
            result = self.client.blpop(self.queue_name, timeout=timeout)
            if result is None:
                return None
            _, data = result
        else:
            data = self.client.lpop(self.queue_name)
            if data is None:
                return None
        return json.loads(data)

    def length(self) -> int:
        return self.client.llen(self.queue_name)
