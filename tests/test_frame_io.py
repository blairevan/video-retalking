"""Regression tests for the disk-backed inference frame cache."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from utils.frame_io import open_source_frames


class OpenSourceFramesTest(unittest.TestCase):
    """Exercise frame access without requiring model or OpenCV installation."""

    def setUp(self):
        """Create a fake decoder and a file-backed image codec."""
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.source = self.root / "clip.mp4"
        self.source.touch()
        self.cache_root = self.root / "cache"
        self.capture = Mock()
        self.capture.get.return_value = 29.97
        self.originals = [
            np.full((4, 6, 3), value, dtype=np.uint8) for value in range(3)
        ]
        self.capture.read.side_effect = [
            *((True, frame) for frame in self.originals),
            (False, None),
        ]
        self.codec = Mock(
            VideoCapture=Mock(return_value=self.capture),
            imwrite=Mock(side_effect=self.write_frame),
            imread=Mock(side_effect=self.read_frame),
            CAP_PROP_FPS=5,
            IMWRITE_PNG_COMPRESSION=16,
        )
        codec_patch = patch.dict(sys.modules, {"cv2": self.codec})
        codec_patch.start()
        self.addCleanup(codec_patch.stop)

    @staticmethod
    def write_frame(path, frame, _options):
        """Save pixels losslessly using the filesystem."""
        with open(path, "wb") as handle:
            np.save(handle, frame, allow_pickle=False)
        return True

    @staticmethod
    def read_frame(path):
        """Return a newly decoded array for each read."""
        with open(path, "rb") as handle:
            return np.load(handle, allow_pickle=False)

    def test_reads_sliced_frames_on_demand_in_forward_and_reverse_order(self):
        """Slicing keeps frame indexes and never decodes the selected pixels."""
        with open_source_frames(self.source, (1, -1, 2, -1), 25, self.cache_root) as (
            frames,
            fps,
            is_image,
        ):
            self.capture.release.assert_called_once()
            self.assertEqual(fps, 29.97)
            self.assertFalse(is_image)
            self.assertEqual(len(frames), 3)
            window = frames[:2]
            self.codec.imread.assert_not_called()
            observed = []
            for index in (0, 1, 2, 1, 0):
                frame = frames[index]
                np.testing.assert_array_equal(frame, self.originals[index][1:, 2:])
                observed.append(int(frame[0, 0, 0]))
                frame[:] = 99
            self.assertEqual(observed, [0, 1, 2, 1, 0])
            self.assertEqual(len(window), 2)
            np.testing.assert_array_equal(window[-1], self.originals[1][1:, 2:])
        self.assertEqual(list(self.cache_root.iterdir()), [])

    def test_uses_full_static_image_and_provided_frame_rate(self):
        """Static images keep their existing full-frame input and FPS."""
        image = self.root / "head.reference.PNG"
        image.touch()
        self.codec.imread.side_effect = None
        self.codec.imread.return_value = self.originals[0]
        with open_source_frames(image, (1, 2, 1, 2), 24, self.cache_root) as (
            frames,
            fps,
            is_image,
        ):
            self.assertTrue(is_image)
            self.assertEqual(fps, 24)
            self.assertEqual(len(frames), 1)
            np.testing.assert_array_equal(frames[0], self.originals[0])
        self.codec.VideoCapture.assert_not_called()

    def test_removes_partial_cache_and_releases_decoder_on_write_failure(self):
        """A partial cache from a failed decode pass is removed."""
        self.codec.imwrite.side_effect = [True, False]
        with self.assertRaisesRegex(OSError, "frame 1"):
            with open_source_frames(self.source, (0, -1, 0, -1), 25, self.cache_root):
                self.fail("A partial cache must never be returned for inference.")
        self.capture.release.assert_called_once()
        self.assertEqual(list(self.cache_root.iterdir()), [])

    def test_removes_cache_when_inference_fails(self):
        """An exception after preprocessing also releases temporary frames."""
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with open_source_frames(self.source, (0, -1, 0, -1), 25, self.cache_root):
                raise RuntimeError("inference failed")
        self.assertEqual(list(self.cache_root.iterdir()), [])
