import pytest
import numpy as np
from pygrex.utils.scale import Scale


class TestScale:
    """Test suite for the Scale class."""

    def test_quantile_basic(self):
        """Test basic functionality of the quantile method."""
        raw_values = [3.0, 1.0, 4.0, 2.0, 6.0]
        scaled = Scale.quantile(raw_values)
        # With 5 elements, we expect ranks to map evenly across 1-5
        assert np.allclose(scaled, [3, 1, 4, 2, 5])

    def test_quantile_empty(self):
        """Test quantile method with empty input."""
        with pytest.raises(ValueError, match="Raw predictions array is empty"):
            Scale.quantile([])

    def test_quantile_single_element(self):
        """Test quantile method with a single element."""
        result = Scale.quantile([7.5])
        assert np.allclose(result, [3])  # Middle of the default range [1, 5]

    def test_quantile_custom_range(self):
        """Test quantile method with custom target range."""
        raw_values = [10, 20, 30, 40, 50]
        scaled = Scale.quantile(raw_values, target_min=0, target_max=1)
        # Should map to [0, 0.25, 0.5, 0.75, 1]
        assert np.allclose(scaled, [0, 0.25, 0.5, 0.75, 1])

    def test_quantile_equal_values(self):
        """Test quantile method with all equal values."""
        raw_values = [5, 5, 5, 5]
        scaled = Scale.quantile(raw_values)
        # All values should get the average rank (2.5), which maps to the middle of [1, 5]
        assert np.allclose(scaled, [3, 3, 3, 3])

    def test_quantile_numpy_input(self):
        """Test quantile method with numpy array input."""
        raw_values = np.array([3.0, 1.0, 4.0, 2.0, 6.0])
        scaled = Scale.quantile(raw_values)
        assert np.allclose(scaled, [3, 1, 4, 2, 5])

    def test_linear_basic(self):
        """Test basic functionality of the linear method."""
        raw_values = [2.0, 4.0, 6.0, 8.0, 10.0]
        scaled = Scale.linear(raw_values)
        # Should map linearly from [2, 10] to [1, 5]
        assert np.allclose(scaled, [1, 2, 3, 4, 5])

    def test_linear_empty(self):
        """Test linear method with empty input."""
        with pytest.raises(ValueError, match="Raw predictions array is empty"):
            Scale.linear([])

    def test_linear_single_element(self):
        """Test linear method with a single element."""
        result = Scale.linear([7.5])
        assert np.allclose(result, [3])  # Middle of the default range [1, 5]

    def test_linear_custom_range(self):
        """Test linear method with custom target range."""
        raw_values = [0, 5, 10]
        scaled = Scale.linear(raw_values, target_min=0, target_max=100)
        assert np.allclose(scaled, [0, 50, 100])

    def test_linear_custom_ref_range(self):
        """Test linear method with custom reference range."""
        raw_values = [2, 5, 8]
        scaled = Scale.linear(raw_values, ref_min=0, ref_max=10)
        # Should map from [0, 10] to [1, 5] regardless of actual min/max
        expected = [1 + (2 / 10) * 4, 1 + (5 / 10) * 4, 1 + (8 / 10) * 4]
        assert np.allclose(scaled, expected)

    def test_linear_with_outliers(self):
        """Test linear method with outlier handling."""
        # Values with outliers
        raw_values = [5, 6, 7, 8, 20]  # 20 is an outlier

        # With outlier handling (default)
        scaled_with_handling = Scale.linear(raw_values)

        # Without outlier handling
        scaled_without_handling = Scale.linear(raw_values, handle_outliers=False)

        # The result with outlier handling should be different
        assert not np.allclose(scaled_with_handling, scaled_without_handling)

        # The outlier should be scaled to the max value (5) without handling
        assert scaled_without_handling[-1] == 5

        # With handling, the outlier should still be clamped to the max
        assert scaled_with_handling[-1] == 5

        # But other values should be more spread out with handling
        assert np.max(scaled_with_handling[:-1]) > np.max(scaled_without_handling[:-1])

    def test_linear_equal_values(self):
        """Test linear method with all equal values."""
        raw_values = [7, 7, 7, 7]
        scaled = Scale.linear(raw_values)
        # All equal values should map to the middle of target range
        assert np.allclose(scaled, [3, 3, 3, 3])

    def test_linear_equal_ref_bounds(self):
        """Test linear method with equal reference bounds."""
        raw_values = [5, 6, 7]
        scaled = Scale.linear(raw_values, ref_min=5, ref_max=5)
        # When ref bounds are equal, should map to middle of target range
        assert np.allclose(scaled, [3, 3, 3])

    def test_linear_numpy_input(self):
        """Test linear method with numpy array input."""
        raw_values = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        scaled = Scale.linear(raw_values)
        assert np.allclose(scaled, [1, 2, 3, 4, 5])

    def test_linear_clipping(self):
        """Test that linear scaling properly clips out-of-bounds values."""
        # Values outside the reference range
        raw_values = [0, 5, 10, 15]
        scaled = Scale.linear(raw_values, ref_min=5, ref_max=10)
        # Values below ref_min should be clipped to target_min
        assert scaled[0] == 1
        # Values above ref_max should be clipped to target_max
        assert scaled[3] == 5

    def test_linear_negative_values(self):
        """Test linear scaling with negative values."""
        raw_values = [-10, -5, 0, 5, 10]
        scaled = Scale.linear(raw_values)
        # Should map [-10, 10] to [1, 5]
        assert np.allclose(scaled, [1, 2, 3, 4, 5])
