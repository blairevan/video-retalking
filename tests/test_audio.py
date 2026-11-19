import unittest
from utils.audio import parse_bool, audio_samples_to_frame_range


class ParseBoolTest(unittest.TestCase):
    """Tests for CLI boolean parsing."""

    def test_parse_bool_accepts_common_values(self):
        """Common case-insensitive boolean spellings should be supported."""
        self.assertTrue(parse_bool("TRUE"))
        self.assertTrue(parse_bool("on"))
        self.assertTrue(parse_bool("1"))
        self.assertTrue(parse_bool("yes"))
        self.assertFalse(parse_bool("0"))
        self.assertFalse(parse_bool("No"))
        self.assertFalse(parse_bool("off"))
        self.assertFalse(parse_bool("false"))

    def test_parse_bool_rejects_unknown_value(self):
        """Unknown values should fail instead of silently changing behavior."""
        with self.assertRaises(ValueError):
            parse_bool("maybe")


class ActiveAudioFrameRangeTest(unittest.TestCase):
    """Tests for mapping detected audio boundaries to video frames."""

    def test_maps_leading_and_trailing_silence_to_frame_range(self):
        """The start rounds down while the end rounds up to preserve speech."""
        frame_range = audio_samples_to_frame_range(
            start_sample=4800,
            end_sample=20800,
            sample_rate=16000,
            fps=25,
        )
        self.assertEqual(frame_range, (7, 33))


if __name__ == "__main__":
    unittest.main()
