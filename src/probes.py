"""One outstanding dependency check per probe, including during outages."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class Probe:
    def __init__(self, check, ttl: float = 10, wait: float = 1):
        if ttl < 0 or wait <= 0:
            raise ValueError("invalid health probe budget")
        self.check, self.ttl, self.wait = check, ttl, wait
        self._lock = threading.Lock()
        self._future = None
        self._cache = None
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="rag-health")

    def get(self):
        with self._lock:
            now = time.monotonic()
            if self._cache and now - self._cache[0] < self.ttl:
                return self._cache[1]
            if self._future is None:
                self._future = self._executor.submit(self.check)
            future = self._future
        try:
            result = future.result(timeout=self.wait)
        except TimeoutError:
            if not future.done():
                return "not_ready", {"checking": True, "note": "必要依賴檢查尚未完成"}
            result = ("not_ready", {"error_type": "TimeoutError"})
        except Exception as exc:
            result = ("not_ready", {"error_type": type(exc).__name__})
        with self._lock:
            if self._future is future:
                self._cache = (time.monotonic(), result)
                self._future = None
        return result

    async def aget(self):
        """Wait without occupying Starlette's shared sync endpoint threads."""
        with self._lock:
            now = time.monotonic()
            if self._cache and now - self._cache[0] < self.ttl:
                return self._cache[1]
            if self._future is None:
                self._future = self._executor.submit(self.check)
            future = self._future
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        try:
            result = await asyncio.wait_for(asyncio.shield(wrapped), self.wait)
        except TimeoutError:
            if not future.done():
                return "not_ready", {"checking": True, "note": "必要依賴檢查尚未完成"}
            result = ("not_ready", {"error_type": "TimeoutError"})
        except Exception as exc:
            result = ("not_ready", {"error_type": type(exc).__name__})
        with self._lock:
            if self._future is future:
                self._cache = (time.monotonic(), result)
                self._future = None
        return result

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
