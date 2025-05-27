from typing import List, Optional, TypeVar, Generic, Iterator

T = TypeVar("T")


class SlidingWindow(Generic[T]):
    """Class for creating and managing sliding windows over a sequence.

    This class provides functionality to iterate through windows of a fixed size
    over a sequence of items.
    """

    def __init__(self, sequence: List[T], window_size: int):
        """Initialize the sliding window.

        Args:
            sequence: The sequence of items to slide over
            window_size: The size of each window (must be positive)

        Raises:
            ValueError: If window_size is less than 1
            TypeError: If sequence is not iterable
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1")

        if not hasattr(sequence, "__iter__"):
            raise TypeError("Sequence must be iterable")

        self.sequence = sequence
        self.window_size = window_size
        self.index = 0
        self.max_index = len(sequence) - window_size + 1 if sequence else 0

    def get_next_window(self) -> Optional[List[T]]:
        """Return the next window and advance the current position.

        Returns:
            A list containing the next window of items, or None if all windows
            have been processed.
        """
        if self.index >= self.max_index:
            return None

        window = self.sequence[self.index : self.index + self.window_size]
        self.index += 1
        return window

    def reset(self) -> None:
        """Reset the window position to the beginning of the sequence."""
        self.index = 0

    def has_next(self) -> bool:
        """Check if there are more windows available.

        Returns:
            True if there are more windows, False otherwise.
        """
        return self.index < self.max_index

    def __iter__(self) -> Iterator[List[T]]:
        """Make the class iterable.

        Returns:
            An iterator over all windows in the sequence.
        """
        self.reset()
        return self

    def __next__(self) -> List[T]:
        """Get the next window for iteration.

        Returns:
            The next window as a list.

        Raises:
            StopIteration: When all windows have been processed.
        """
        window = self.get_next_window()
        if window is None:
            raise StopIteration
        return window

    def __len__(self) -> int:
        """Return the total number of windows.

        Returns:
            The number of complete windows in the sequence.
        """
        return max(0, self.max_index)
