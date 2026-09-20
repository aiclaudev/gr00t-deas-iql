"""Process-local video decoder overrides for explicit worker benchmarks only.

Nothing is installed merely by importing this module. The production dataset and
video helpers are not modified on disk. ``decord`` installs the original helper;
its automatic thread selection, per-call reader creation and timestamp scan are
preserved. Both cached variants use the same bounded per-process LRU.

TorchCodec's timestamp lookup selects the frame displayed *at* a timestamp,
whereas this dataset chooses the nearest frame start. To preserve the dataset's
semantics for variable-frame-rate videos and ties, TorchCodec uses a timestamp
index read once with Decord on every cache miss, then decodes those exact integer
indices with TorchCodec. Both index construction and exact-seek initialization
therefore remain part of the measured workload. No constant-FPS assumption is
made, and RGB values can still differ slightly between decoder implementations.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib
import os
from typing import Any, Callable

import numpy as np


BACKENDS = ("decord", "decord_cached", "torchcodec")


def _open_decord(video_path: str, num_threads: int):
    import decord
    return decord.VideoReader(video_path, num_threads=num_threads)


def _open_torchcodec(video_path: str, num_threads: int):
    from torchcodec.decoders import VideoDecoder
    return VideoDecoder(
        video_path, device="cpu", dimension_order="NHWC",
        num_ffmpeg_threads=num_threads, seek_mode="exact",
    )


@dataclass
class _Entry:
    reader: Any
    starts: np.ndarray


class BenchmarkVideoBackend:
    """Callable replacement plus local counters; safe across DataLoader forks.

    Decoder state and counters are local to each OS process. Calling ``stats``
    on the parent does not aggregate DataLoader worker counters. ``clear`` drops
    decoder references; underlying libraries own native decoder destruction.
    """

    def __init__(
        self, backend: str, reader_cache_size: int = 16, num_threads: int = 1,
        original: Callable | None = None,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown video benchmark backend: {backend!r}")
        if reader_cache_size < 1:
            raise ValueError("reader_cache_size must be positive")
        if num_threads < 1:
            raise ValueError("num_threads must be positive for cached decoders")
        self.backend = backend
        self.reader_cache_size = int(reader_cache_size)
        self.num_threads = int(num_threads)
        self.original = original
        self._pid = os.getpid()
        self._cache: OrderedDict[str, _Entry] = OrderedDict()
        self._counters = self._empty_counters()

    @staticmethod
    def _empty_counters() -> dict[str, int]:
        return dict(requests=0, frames=0, hits=0, misses=0, evictions=0)

    def _check_process(self) -> None:
        if self._pid != os.getpid():
            # Never reuse a decoder inherited from a different process.
            self._cache.clear()
            self._counters = self._empty_counters()
            self._pid = os.getpid()

    def clear(self) -> None:
        self._check_process()
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        self._check_process()
        return {
            "backend": self.backend,
            "pid": self._pid,
            "reader_cache_size": self.reader_cache_size,
            "cached_readers": len(self._cache),
            "num_threads": None if self.backend == "decord" else self.num_threads,
            "thread_policy": "original" if self.backend == "decord" else "explicit",
            "timestamp_index": "decord_nearest_frame_start",
            "counters_observed": self.backend != "decord",
            **self._counters,
        }

    def _entry(self, video_path: str) -> _Entry:
        if video_path in self._cache:
            self._counters["hits"] += 1
            self._cache.move_to_end(video_path)
            return self._cache[video_path]
        self._counters["misses"] += 1
        # Evict before opening, bounding persistent native decoder resources.
        if len(self._cache) >= self.reader_cache_size:
            self._cache.popitem(last=False)
            self._counters["evictions"] += 1
        reader = _open_decord(video_path, self.num_threads)
        count = len(reader)
        if count < 1:
            raise ValueError(f"Video contains no frames: {video_path}")
        table = np.asarray(reader.get_frame_timestamp(range(count)))
        if table.ndim != 2 or table.shape[0] != count or table.shape[1] < 1:
            raise ValueError(f"Invalid Decord frame timestamp table: {video_path}")
        # Preserve Decord's dtype: casting could change a nearest-frame tie.
        starts = table[:, :1].copy()
        if not np.isfinite(starts).all():
            raise ValueError(f"Nonfinite video frame timestamps: {video_path}")
        if self.backend == "torchcodec":
            del reader
            reader = _open_torchcodec(video_path, self.num_threads)
            if len(reader) != count:
                raise ValueError(
                    f"Decoder frame-count mismatch for {video_path}: "
                    f"Decord={count}, TorchCodec={len(reader)}"
                )
        result = _Entry(reader=reader, starts=starts)
        self._cache[video_path] = result
        return result

    def __call__(
        self, video_path: str, timestamps: list[float] | np.ndarray,
        video_backend: str = "decord", video_backend_kwargs: dict | None = None,
    ) -> np.ndarray:
        if self.backend == "decord":
            if self.original is None:
                raise RuntimeError("Baseline requires the original video helper")
            return self.original(
                video_path, timestamps, video_backend,
                {} if video_backend_kwargs is None else video_backend_kwargs,
            )
        self._check_process()
        if video_backend != "decord":
            raise ValueError("Benchmark override expects a dataset configured with decord")
        kwargs = {} if video_backend_kwargs is None else video_backend_kwargs
        if set(kwargs) - {"num_threads"}:
            raise ValueError("Cached benchmark supports only num_threads decoder kwargs")
        if "num_threads" in kwargs and kwargs["num_threads"] != self.num_threads:
            raise ValueError("Dataset num_threads conflicts with benchmark num_threads")
        requested = np.asarray(timestamps)
        if requested.ndim != 1 or requested.size == 0 or not np.isfinite(requested).all():
            raise ValueError("timestamps must be a nonempty, finite one-dimensional array")
        self._counters["requests"] += 1
        entry = self._entry(str(video_path))
        # Intentionally identical to gr00t.utils.video's nearest-start rule,
        # including first-index tie breaking and out-of-range behavior.
        indices = np.abs(entry.starts - requested).argmin(axis=0)
        if self.backend == "decord_cached":
            result = entry.reader.get_batch(indices).asnumpy()
        else:
            result = entry.reader.get_frames_at(indices=indices.tolist()).data.cpu().numpy()
        result = np.asarray(result)
        if result.dtype != np.uint8 or result.ndim != 4 or result.shape[-1] != 3:
            raise ValueError("Decoder must return RGB uint8 frames in NHWC order")
        if result.shape[0] != len(requested):
            raise ValueError("Decoder returned an unexpected frame count")
        self._counters["frames"] += len(requested)
        return result


def install_benchmark_video_backend(
    backend: str = "decord", reader_cache_size: int = 16, num_threads: int = 1,
) -> BenchmarkVideoBackend:
    """Override the dataset's imported helper in this benchmark process only.

    Call before constructing DataLoader workers. Linux ``fork`` workers inherit
    the override, then start with empty process-local reader caches. For
    ``spawn`` workers, call this installer explicitly in their worker_init_fn.
    The baseline installs the original function directly, with no wrapper or
    changed thread setting. Return value exposes process-local cache diagnostics.
    """
    original = importlib.import_module("gr00t.utils.video").get_frames_by_timestamps
    loader = BenchmarkVideoBackend(backend, reader_cache_size, num_threads, original)
    dataset_module = importlib.import_module("gr00t.data.dataset")
    dataset_module.get_frames_by_timestamps = original if backend == "decord" else loader
    return loader
