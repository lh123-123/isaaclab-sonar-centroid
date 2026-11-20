import unittest

import numpy as np

from auto_focus_scan import clarity_score


def checkerboard(height, width, tile=4):
    yy, xx = np.indices((height, width))
    return (((xx // tile + yy // tile) % 2) * 255).astype(np.uint8)


class ClawCenterClarityTests(unittest.TestCase):
    def test_claw_center_score_prioritizes_lower_center_detail(self):
        lower_center = np.zeros((200, 300), dtype=np.uint8)
        lower_center[90:190, 90:210] = checkerboard(100, 120)
        upper_corner = np.zeros((200, 300), dtype=np.uint8)
        upper_corner[0:100, 0:120] = checkerboard(100, 120)

        self.assertGreater(
            clarity_score(lower_center, roi="claw_center"),
            clarity_score(upper_corner, roi="claw_center") * 3,
        )

    def test_unknown_roi_is_rejected(self):
        with self.assertRaises(ValueError):
            clarity_score(np.zeros((100, 100), dtype=np.uint8), roi="unknown")


if __name__ == "__main__":
    unittest.main()
