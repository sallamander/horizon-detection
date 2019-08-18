"""Unit tests for utils.py"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from utils import detect_horizon_line


class TestDetectHorizonLine():
    """Test detect_horizon_line"""

    @pytest.fixture(scope='class')
    def image_closed1(self):
        """image_closed1 object fixture

        This is used to test the final couple lines of code in
        `detect_horizon_line`, which finds the y-coordinates of the horizon
        line based off of the location of the max index holding a 0 value in
        the first / last column of the image (i.e. the first and last possible
        x-position). The image returned here should result in y-coordinates of
        (31, 15) being returned.
        """

        image_closed1 = np.ones((64, 48))
        image_closed1[:32, :] = 0
        image_closed1[16:, 47] = 1

        return image_closed1

    def test_detect_horizon_line(self, image_closed1, monkeypatch):
        """Test detect_horizon_line

        This tests two big things:
        - The flow of inputs / outputs through operations in
          `detect_horizon_line` is as expected (e.g. the output of the gaussian
          blur operation is passed to the threshold operation)
        - The returned (x1, x2, y1, y2) coordinates denoting the horizon are
          as expected

        :param image_closed1: image_closed1 object fixture
        :type image_closed1: np.ndarray
        """

        mock_blur = MagicMock()
        mock_blur.return_value = 'blur_return'
        monkeypatch.setattr('utils.cv2.GaussianBlur', mock_blur)

        mock_threshold = MagicMock()
        mock_threshold.return_value = (
            'threshold_return1', 3
        )
        monkeypatch.setattr('utils.cv2.threshold', mock_threshold)

        mock_morphology = MagicMock()
        mock_morphology.return_value = image_closed1
        monkeypatch.setattr('utils.cv2.morphologyEx', mock_morphology)

        image_grayscale = np.ones_like(image_closed1)
        horizon_x1, horizon_x2, horizon_y1, horizon_y2 = (
            detect_horizon_line(image_grayscale)
        )

        assert horizon_x1 == 0
        assert horizon_x2 == 47
        assert horizon_y1 == 31
        assert horizon_y2 == 15
