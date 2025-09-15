from typing import List, Union, Optional
import numpy as np
from scipy import stats


class Scale:
    """
    A class for scaling numerical values using different methods.

    Methods:
        quantile: Scale values using quantile-based ranking.
        linear: Scale values linearly to a target range with outlier handling.
    """

    @staticmethod
    def quantile(
        raw_predictions: Union[List[float], np.ndarray],
        target_min: float = 1,
        target_max: float = 5,
    ) -> np.ndarray:
        """
        Scale raw predictions to the target range using quantile-based ranking.

        Args:
            raw_predictions: The raw prediction values.
            target_min: Minimum of the target range (default: 1).
            target_max: Maximum of the target range (default: 5).

        Returns:
            numpy.ndarray: Scaled predictions.

        Raises:
            ValueError: If raw_predictions is empty.
        """
        if len(raw_predictions) == 0:
            raise ValueError("Raw predictions array is empty.")

        # Convert to numpy array if it's not already
        raw_predictions = np.array(raw_predictions)

        ranks = stats.rankdata(raw_predictions, method="average")
        if len(raw_predictions) == 1:
            # Handle single element case
            scaled_predictions = np.array([(target_min + target_max) / 2])
        else:
            scaled_predictions = target_min + (ranks - 1) * (
                target_max - target_min
            ) / (len(raw_predictions) - 1)

        # Ensure scaled predictions are within [target_min, target_max]
        scaled_predictions = np.clip(scaled_predictions, target_min, target_max)

        return scaled_predictions

    @staticmethod
    def linear(
        raw_predictions: Union[List[float], np.ndarray],
        target_min: float = 1,
        target_max: float = 5,
        ref_min: Optional[float] = None,
        ref_max: Optional[float] = None,
        handle_outliers: bool = True,
    ) -> np.ndarray:
        """
        Scale raw predictions to the target range [target_min, target_max].

        Args:
            raw_predictions: The raw prediction values.
            target_min: Minimum of the target range (default: 1).
            target_max: Maximum of the target range (default: 5).
            ref_min: Reference minimum for raw predictions. If None, will be calculated
                     from the data or from outlier bounds if handle_outliers=True.
            ref_max: Reference maximum for raw predictions. If None, will be calculated
                     from the data or from outlier bounds if handle_outliers=True.
            handle_outliers: Whether to handle outliers using IQR method (default: True).

        Returns:
            numpy.ndarray: Scaled predictions.

        Raises:
            ValueError: If raw_predictions is empty.
        """
        if len(raw_predictions) == 0:
            raise ValueError("Raw predictions array is empty.")

        # Convert to numpy array if it's not already
        raw_predictions = np.array(raw_predictions)

        # Handle single element case
        if len(raw_predictions) == 1:
            if ref_min is not None and ref_max is not None:
                # Scale based on provided reference range
                value = raw_predictions[0]
                scaled_value = (
                    target_min
                    + (value - ref_min)
                    * (target_max - target_min)
                    / (ref_max - ref_min)
                    if ref_max != ref_min
                    else (target_min + target_max) / 2
                )
                scaled_value = np.clip(scaled_value, target_min, target_max)
                return np.array([scaled_value])
            else:
                # Can't determine range from single value, return middle of target range
                return np.array([(target_min + target_max) / 2])

        clipped_predictions = raw_predictions.copy()

        # Handle outliers if requested
        if handle_outliers:
            q1, q3 = np.percentile(raw_predictions, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            clipped_predictions = np.clip(raw_predictions, lower_bound, upper_bound)

        # Determine min and max values
        min_raw = np.min(clipped_predictions)
        max_raw = np.max(clipped_predictions)

        # Use provided reference bounds if given, otherwise use data bounds
        actual_ref_min = ref_min if ref_min is not None else min_raw
        actual_ref_max = ref_max if ref_max is not None else max_raw

        # Scale to [target_min, target_max]
        if actual_ref_max == actual_ref_min:
            # Reference bounds are equal, return the middle of the target range
            return np.full_like(raw_predictions, (target_min + target_max) / 2)
        else:
            scaled_predictions = target_min + (raw_predictions - actual_ref_min) * (
                target_max - target_min
            ) / (actual_ref_max - actual_ref_min)

        # Ensure scaled predictions are within [target_min, target_max]
        scaled_predictions = np.clip(scaled_predictions, target_min, target_max)

        return scaled_predictions
