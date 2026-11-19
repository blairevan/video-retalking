"""Keep decoded video frames on disk until a stage needs their pixels."""

import math
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


class FrameSequence(Sequence):
    """An indexable collection of frame paths with lazy pixel reads."""

    def __init__(self, frame_paths, read_frame):
        """Store paths and the decoder used for individual reads."""
        self._paths = tuple(frame_paths)
        self._read_frame = read_frame

    def __len__(self):
        """Return the number of available frames."""
        return len(self._paths)

    def __getitem__(self, index):
        """Decode one frame or return a slice containing only paths."""
        if isinstance(index, slice):
            return FrameSequence(self._paths[index], self._read_frame)
        frame_path = self._paths[index]
        frame = self._read_frame(str(frame_path))
        if frame is None:
            raise OSError(f"Could not read cached frame: {frame_path}")
        return frame


@contextmanager
def open_source_frames(source, crop, image_fps, cache_root="temp"):
    """Yield the frames, source FPS and static-image flag for one inference run."""
    import cv2

    source = Path(source)
    if not source.is_file():
        raise ValueError("--face must point to a video or image file")

    if source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Could not read image: {source}")
        yield [frame], image_fps, True
        return

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    top, bottom, left, right = crop
    with TemporaryDirectory(prefix="video-frames-", dir=cache_root) as cache_dir:
        frame_paths = []
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_bottom = frame.shape[0] if bottom == -1 else bottom
                frame_right = frame.shape[1] if right == -1 else right
                cropped = frame[top:frame_bottom, left:frame_right]
                if cropped.size == 0:
                    raise ValueError(f"Video crop is empty at frame {len(frame_paths)}")
                frame_path = Path(cache_dir) / f"{len(frame_paths):08d}.png"
                if not cv2.imwrite(
                    str(frame_path), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                ):
                    raise OSError(f"Could not cache video frame {len(frame_paths)}")
                frame_paths.append(frame_path)
        finally:
            capture.release()

        if not frame_paths:
            raise ValueError(f"Could not decode any video frames: {source}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video frame rate: {fps}")
        del frame, cropped
        yield FrameSequence(frame_paths, cv2.imread), fps, False
