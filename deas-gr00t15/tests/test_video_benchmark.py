"""CPU-only contract checks using fake decoders; no speed measurements."""
import types
import unittest
from unittest.mock import patch

import numpy as np

from gr00t.utils import video_benchmark as vb


class FakeArray:
    def __init__(self, values):
        self.values = values

    def asnumpy(self):
        return self.values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeDecord:
    def __init__(self, starts):
        self.starts = np.asarray(starts, dtype=np.float32)
        self.timestamp_reads = 0
        self.index_requests = []

    def __len__(self):
        return len(self.starts)

    def get_frame_timestamp(self, indices):
        self.timestamp_reads += 1
        np.testing.assert_array_equal(list(indices), np.arange(len(self)))
        return np.stack((self.starts, self.starts + .125), axis=1)

    def get_batch(self, indices):
        indices = np.asarray(indices)
        self.index_requests.append(indices.tolist())
        return FakeArray(np.broadcast_to(indices[:, None, None, None], (len(indices), 2, 3, 3)).astype(np.uint8))


class FakeCodec(FakeDecord):
    def get_frames_at(self, *, indices):
        return types.SimpleNamespace(data=self.get_batch(indices))


class BenchmarkVideoTests(unittest.TestCase):
    def test_irregular_timestamps_ties_duplicates_and_bounds_match_original(self):
        starts = [0, .125, .5, 1.125]
        requested = np.asarray([1.1, .3125, -.1, 100, .5, .5], dtype=np.float32)
        reader = FakeDecord(starts)
        loader = vb.BenchmarkVideoBackend("decord_cached")
        with patch.object(vb, "_open_decord", return_value=reader) as factory:
            result = loader("example.mp4", requested)
            loader("example.mp4", [.125])
        expected = np.abs(reader.starts[:, None] - requested).argmin(axis=0)
        np.testing.assert_array_equal(result[:, 0, 0, 0], expected)
        self.assertEqual(reader.index_requests[0], [3, 1, 0, 3, 2, 2])
        self.assertEqual(reader.timestamp_reads, 1)
        factory.assert_called_once_with("example.mp4", 1)
        self.assertEqual(loader.stats()["hits"], 1)
        self.assertEqual(loader.stats()["misses"], 1)

    def test_lru_capacity_and_recency(self):
        loader = vb.BenchmarkVideoBackend("decord_cached", reader_cache_size=2)
        with patch.object(vb, "_open_decord", side_effect=lambda *_: FakeDecord([0, 1])) as factory:
            for name in ("a", "b", "a", "c", "a", "b"):
                loader(name, [0])
        self.assertEqual(factory.call_count, 4)
        self.assertEqual(loader.stats()["cached_readers"], 2)
        self.assertEqual(loader.stats()["evictions"], 2)
        self.assertEqual(loader.stats()["hits"], 2)
        loader.clear()
        self.assertEqual(loader.stats()["cached_readers"], 0)

    def test_forked_process_starts_new_cache_and_counters(self):
        with patch.object(vb.os, "getpid", return_value=100):
            loader = vb.BenchmarkVideoBackend("decord_cached")
            with patch.object(vb, "_open_decord", side_effect=lambda *_: FakeDecord([0, 1])) as factory:
                loader("a", [0])
                with patch.object(vb.os, "getpid", return_value=101):
                    loader("a", [1])
                    self.assertEqual(loader.stats()["requests"], 1)
                    self.assertEqual(loader.stats()["pid"], 101)
            self.assertEqual(factory.call_count, 2)

    def test_torchcodec_uses_same_cached_integer_index(self):
        starts = [0, .125, .5, 1.125]
        index_reader, decoder = FakeDecord(starts), FakeCodec(starts)
        loader = vb.BenchmarkVideoBackend("torchcodec", reader_cache_size=3, num_threads=2)
        with patch.object(vb, "_open_decord", return_value=index_reader) as index_factory:
            with patch.object(vb, "_open_torchcodec", return_value=decoder) as codec_factory:
                result = loader("a", [.3125, 1.2, .3125])
                loader("a", [.5])
        np.testing.assert_array_equal(result[:, 0, 0, 0], [1, 3, 1])
        index_factory.assert_called_once_with("a", 2)
        codec_factory.assert_called_once_with("a", 2)
        self.assertEqual(index_reader.timestamp_reads, 1)
        self.assertEqual(decoder.timestamp_reads, 0)
        self.assertEqual(loader.stats()["frames"], 4)

    def test_mismatched_frame_counts_fail_instead_of_changing_samples(self):
        loader = vb.BenchmarkVideoBackend("torchcodec")
        with patch.object(vb, "_open_decord", return_value=FakeDecord([0, 1])):
            with patch.object(vb, "_open_torchcodec", return_value=FakeCodec([0])):
                with self.assertRaisesRegex(ValueError, "frame-count mismatch"):
                    loader("a", [0])
        self.assertEqual(loader.stats()["cached_readers"], 0)

    def test_explicit_cpu_exact_and_nhwc_codec_construction(self):
        decoder_module = types.ModuleType("torchcodec.decoders")
        from unittest.mock import Mock
        decoder_module.VideoDecoder = Mock()
        with patch.dict("sys.modules", {"torchcodec.decoders": decoder_module}):
            vb._open_torchcodec("example", 3)
        decoder_module.VideoDecoder.assert_called_once_with(
            "example", device="cpu", dimension_order="NHWC",
            num_ffmpeg_threads=3, seek_mode="exact",
        )

    def test_baseline_install_restores_unwrapped_original(self):
        original = lambda *args, **kwargs: None
        original_module = types.SimpleNamespace(get_frames_by_timestamps=original)
        dataset_module = types.SimpleNamespace(get_frames_by_timestamps=None)
        def modules(name):
            return original_module if name == "gr00t.utils.video" else dataset_module
        with patch.object(vb.importlib, "import_module", side_effect=modules):
            cached = vb.install_benchmark_video_backend("decord_cached")
            self.assertIs(dataset_module.get_frames_by_timestamps, cached)
            baseline = vb.install_benchmark_video_backend("decord", num_threads=7)
            self.assertIs(dataset_module.get_frames_by_timestamps, original)
        self.assertFalse(baseline.stats()["counters_observed"])
        self.assertIsNone(baseline.stats()["num_threads"])

    def test_invalid_settings_do_not_silently_change_decode(self):
        with self.assertRaises(ValueError):
            vb.BenchmarkVideoBackend("other")
        with self.assertRaises(ValueError):
            vb.BenchmarkVideoBackend("torchcodec", reader_cache_size=0)
        with self.assertRaises(ValueError):
            vb.BenchmarkVideoBackend("torchcodec", num_threads=0)
        loader = vb.BenchmarkVideoBackend("decord_cached")
        for kwargs in ({"width": 64}, {"num_threads": 4}):
            with self.assertRaises(ValueError):
                loader("a", [0], video_backend_kwargs=kwargs)
        with self.assertRaises(ValueError):
            loader("a", [0], video_backend="opencv")
        for requested in ([], [float("nan")], [[0]]):
            with self.assertRaises(ValueError):
                loader("a", requested)


if __name__ == "__main__":
    unittest.main()
