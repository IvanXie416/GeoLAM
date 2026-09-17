"""Bad-sample persistence and retryable data-error helpers."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None


class RetryableSampleError(RuntimeError):
    """Exception type for data-corruption/read errors that should be retried."""

    def __init__(self, message: str, failed_paths: list[str] | None = None) -> None:
        super().__init__(message)
        self.failed_paths = [str(Path(p)) for p in (failed_paths or []) if str(p).strip()]


_RUNTIME_RETRY_KEYWORDS = (
    "truncated",
    "cannot identify image file",
    "failed to read frame",
    "decompression failed",
    "corrupt",
    "unexpected end",
    "crc",
    "end of data",
)


def is_retryable_data_error(exc: Exception) -> bool:
    """Heuristic filter: skip only probable data I/O/corruption errors."""

    if isinstance(exc, RetryableSampleError):
        return True
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError, EOFError, OSError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(token in msg for token in _RUNTIME_RETRY_KEYWORDS)
    return False


def failed_paths_from_exception(exc: Exception) -> list[str]:
    if isinstance(exc, RetryableSampleError):
        return list(exc.failed_paths)
    return []


class BadSampleRegistry:
    """Process-safe JSON registry for bad samples and bad paths."""

    def __init__(self, path: str | Path, refresh_seconds: float = 1.0) -> None:
        self.path = Path(path)
        self.refresh_seconds = max(0.0, float(refresh_seconds))
        self._last_refresh_ts = 0.0
        self._loaded_mtime_ns: int | None = None
        self._bad_sample_keys: set[str] = set()
        self._bad_paths: set[str] = set()
        self._reload(force=True)

    def _now_iso(self) -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _norm_path(self, value: str | Path) -> str:
        try:
            return str(Path(value).resolve())
        except Exception:
            return str(Path(value))

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "updated_at": self._now_iso(),
            "items": [],
        }

    def _reload(self, force: bool = False) -> None:
        self._last_refresh_ts = time.time()
        try:
            stat = self.path.stat()
            mtime_ns = int(stat.st_mtime_ns)
        except FileNotFoundError:
            self._loaded_mtime_ns = None
            self._bad_sample_keys = set()
            self._bad_paths = set()
            return
        except Exception:
            return

        if not force and self._loaded_mtime_ns == mtime_ns:
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        items = payload.get("items", [])
        if not isinstance(items, list):
            items = []

        sample_keys: set[str] = set()
        bad_paths: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("sample_key", "")).strip()
            if key:
                sample_keys.add(key)
            for raw_path in item.get("failed_paths", []) or []:
                p = str(raw_path).strip()
                if p:
                    bad_paths.add(self._norm_path(p))

        self._loaded_mtime_ns = mtime_ns
        self._bad_sample_keys = sample_keys
        self._bad_paths = bad_paths

    def _refresh_if_needed(self) -> None:
        if (time.time() - self._last_refresh_ts) >= self.refresh_seconds:
            self._reload(force=False)

    def is_bad_sample(self, sample_key: str) -> bool:
        if not sample_key:
            return False
        self._refresh_if_needed()
        return sample_key in self._bad_sample_keys

    def has_any_bad_path(self, paths: list[str]) -> bool:
        if not paths:
            return False
        self._refresh_if_needed()
        for path in paths:
            p = str(path).strip()
            if not p:
                continue
            if self._norm_path(p) in self._bad_paths:
                return True
        return False

    def mark_bad(
        self,
        dataset: str,
        sample_key: str,
        sample_paths: list[str],
        failed_paths: list[str],
        error: str,
    ) -> None:
        key = str(sample_key).strip()
        if not key:
            return

        normalized_sample_paths = []
        seen_sample_paths: set[str] = set()
        for raw in sample_paths:
            p = str(raw).strip()
            if not p:
                continue
            norm = self._norm_path(p)
            if norm in seen_sample_paths:
                continue
            seen_sample_paths.add(norm)
            normalized_sample_paths.append(norm)

        normalized_failed_paths = []
        seen_failed_paths: set[str] = set()
        for raw in failed_paths:
            p = str(raw).strip()
            if not p:
                continue
            norm = self._norm_path(p)
            if norm in seen_failed_paths:
                continue
            seen_failed_paths.add(norm)
            normalized_failed_paths.append(norm)

        now = self._now_iso()

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.seek(0)
                    raw = handle.read()
                    if raw.strip():
                        payload = json.loads(raw)
                    else:
                        payload = self._empty_payload()
                    items = payload.get("items", [])
                    if not isinstance(items, list):
                        items = []
                        payload["items"] = items

                    existing = None
                    for item in items:
                        if isinstance(item, dict) and str(item.get("sample_key", "")).strip() == key:
                            existing = item
                            break

                    if existing is None:
                        items.append(
                            {
                                "dataset": str(dataset),
                                "sample_key": key,
                                "sample_paths": normalized_sample_paths,
                                "failed_paths": normalized_failed_paths,
                                "error_count": 1,
                                "first_seen_utc": now,
                                "last_seen_utc": now,
                                "last_error": str(error),
                            }
                        )
                    else:
                        existing["dataset"] = str(dataset)
                        existing["error_count"] = int(existing.get("error_count", 0)) + 1
                        existing["last_seen_utc"] = now
                        if "first_seen_utc" not in existing:
                            existing["first_seen_utc"] = now
                        existing["last_error"] = str(error)

                        sample_paths_all = []
                        seen = set()
                        for p in existing.get("sample_paths", []) + normalized_sample_paths:
                            ps = str(p).strip()
                            if not ps or ps in seen:
                                continue
                            seen.add(ps)
                            sample_paths_all.append(ps)
                        existing["sample_paths"] = sample_paths_all

                        failed_paths_all = []
                        seen = set()
                        for p in existing.get("failed_paths", []) + normalized_failed_paths:
                            ps = str(p).strip()
                            if not ps or ps in seen:
                                continue
                            seen.add(ps)
                            failed_paths_all.append(ps)
                        existing["failed_paths"] = failed_paths_all

                    payload["version"] = 1
                    payload["updated_at"] = now

                    handle.seek(0)
                    handle.truncate(0)
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            # Keep training resilient even if bad-sample logging path is unavailable.
            pass

        self._bad_sample_keys.add(key)
        for path in normalized_failed_paths:
            self._bad_paths.add(path)
