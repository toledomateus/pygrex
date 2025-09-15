import pytest
from typing import List, Any

from pygrex.utils.sliding_window import SlidingWindow


class TestSlidingWindow:
    """Test suite for the SlidingWindow class."""

    def test_basic_functionality(self):
        """Test the basic window sliding functionality."""
        data = [1, 2, 3, 4, 5]
        window_size = 3
        sliding_window = SlidingWindow(data, window_size)

        # First window
        assert sliding_window.get_next_window() == [1, 2, 3]
        # Second window
        assert sliding_window.get_next_window() == [2, 3, 4]
        # Third window
        assert sliding_window.get_next_window() == [3, 4, 5]
        # No more windows
        assert sliding_window.get_next_window() is None

    def test_window_equal_to_sequence_length(self):
        """Test when window size equals the sequence length."""
        data = [1, 2, 3]
        window_size = 3
        sliding_window = SlidingWindow(data, window_size)

        assert sliding_window.get_next_window() == [1, 2, 3]
        assert sliding_window.get_next_window() is None

    def test_window_larger_than_sequence(self):
        """Test when window size is larger than the sequence length."""
        data = [1, 2, 3]
        window_size = 4
        sliding_window = SlidingWindow(data, window_size)

        assert sliding_window.get_next_window() is None

    def test_empty_sequence(self):
        """Test with an empty sequence."""
        data: List[Any] = []
        window_size = 2
        sliding_window = SlidingWindow(data, window_size)

        assert sliding_window.get_next_window() is None

    def test_invalid_window_size(self):
        """Test with invalid window sizes."""
        data = [1, 2, 3, 4, 5]

        # Test with zero window size
        with pytest.raises(ValueError):
            SlidingWindow(data, 0)

        # Test with negative window size
        with pytest.raises(ValueError):
            SlidingWindow(data, -1)

    def test_non_iterable_sequence(self):
        """Test with a non-iterable object."""
        data = 123  # Integer is not iterable
        window_size = 2

        with pytest.raises(TypeError):
            SlidingWindow(data, window_size)

    def test_reset_functionality(self):
        """Test the reset functionality."""
        data = [1, 2, 3, 4]
        window_size = 2
        sliding_window = SlidingWindow(data, window_size)

        # Get first two windows
        assert sliding_window.get_next_window() == [1, 2]
        assert sliding_window.get_next_window() == [2, 3]

        # Reset and check if we get the first window again
        sliding_window.reset()
        assert sliding_window.get_next_window() == [1, 2]

    def test_has_next(self):
        """Test the has_next method."""
        data = [1, 2, 3]
        window_size = 2
        sliding_window = SlidingWindow(data, window_size)

        assert sliding_window.has_next() is True
        sliding_window.get_next_window()  # Get first window
        assert sliding_window.has_next() is True
        sliding_window.get_next_window()  # Get second window
        assert sliding_window.has_next() is False

    def test_iterator_protocol(self):
        """Test the iterator protocol implementation."""
        data = [1, 2, 3, 4]
        window_size = 2
        sliding_window = SlidingWindow(data, window_size)

        # Using the class in a for loop
        windows = []
        for window in sliding_window:
            windows.append(window)

        assert windows == [[1, 2], [2, 3], [3, 4]]

        # After iteration, the index should be at the end
        assert sliding_window.has_next() is False

        # Test that reset works after iteration
        sliding_window.reset()
        assert sliding_window.has_next() is True
        assert sliding_window.get_next_window() == [1, 2]

    def test_len_functionality(self):
        """Test the __len__ method."""
        # Normal case
        data = [1, 2, 3, 4, 5]
        window_size = 2
        sliding_window = SlidingWindow(data, window_size)
        assert len(sliding_window) == 4

        # Window size equals sequence length
        window_size = 5
        sliding_window = SlidingWindow(data, window_size)
        assert len(sliding_window) == 1

        # Window size greater than sequence length
        window_size = 6
        sliding_window = SlidingWindow(data, window_size)
        assert len(sliding_window) == 0

        # Empty sequence
        data = []
        window_size = 2
        sliding_window = SlidingWindow(data, window_size)
        assert len(sliding_window) == 0

    def test_with_string_data(self):
        """Test with string data to verify generic implementation."""
        data = "abcde"
        window_size = 3
        sliding_window = SlidingWindow(data, window_size)

        assert sliding_window.get_next_window() == "abc"
        assert sliding_window.get_next_window() == "bcd"
        assert sliding_window.get_next_window() == "cde"
        assert sliding_window.get_next_window() is None

    def test_multiple_data_types(self):
        """Test with a list containing multiple data types."""
        data = [1, "two", 3.0, [4, 5], {"six": 6}]
        window_size = 2
        sliding_window = SlidingWindow(data, window_size)

        assert sliding_window.get_next_window() == [1, "two"]
        assert sliding_window.get_next_window() == ["two", 3.0]
        assert sliding_window.get_next_window() == [3.0, [4, 5]]
        assert sliding_window.get_next_window() == [[4, 5], {"six": 6}]
        assert sliding_window.get_next_window() is None
