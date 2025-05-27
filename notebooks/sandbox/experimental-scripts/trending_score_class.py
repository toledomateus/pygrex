import pandas as pd
from scipy.signal import find_peaks, peak_widths
import numpy as np
import random


class UserTrendingScorer:
    """
    Calculates a "trending score" for users based on how often they rate movies
    during identified hype periods. Hype periods are determined by analyzing
    local peaks in monthly movie rating counts, using normalized data for robustness.
    """

    def __init__(
        self,
        df_path: str,
        peak_norm_min_height: float = 0.3,
        peak_min_distance: int = 9,
        peak_norm_min_prominence: float = 0.2,
        peak_width_rel_height: float = 0.6,
    ):
        """
        Initializes the UserTrendingScorer.

        Args:
            df_path (str): Path to the ratings CSV file.
            peak_norm_min_height (float): Minimum normalized height for a peak (on [0,1] scale).
            peak_min_distance (int): Minimum number of months separating peaks.
            peak_norm_min_prominence (float): Minimum normalized prominence for a peak (on [0,1] scale).
            peak_width_rel_height (float): Relative height at which to measure peak width (on [0,1] scale
                                           relative to peak's prominence-based height).
        """
        self.df_path = df_path
        self.peak_norm_min_height = peak_norm_min_height
        self.peak_min_distance = peak_min_distance
        self.peak_norm_min_prominence = peak_norm_min_prominence
        self.peak_width_rel_height = peak_width_rel_height

        self._df = self._load_and_preprocess_data()
        if self._df is not None:
            self._movie_ratings_per_month = (
                self._df.groupby(["itemId", "year_month"], observed=False)
                .size()
                .reset_index(name="rating_count")
            )
            self._movie_hype_periods_df = self._calculate_all_movie_hype_periods()
        else:
            self._movie_ratings_per_month = (
                pd.DataFrame()
            )  # Ensure it's an empty DataFrame
            self._movie_hype_periods_df = (
                pd.DataFrame()
            )  # Ensure it's an empty DataFrame

    def _load_and_preprocess_data(self) -> pd.DataFrame | None:
        """
        Loads data from the CSV file and preprocesses timestamps.
        Returns a DataFrame or None if loading fails.
        """
        try:
            df = pd.read_csv(self.df_path)
        except FileNotFoundError:
            print(f"Error: The file '{self.df_path}' was not found.")
            return None
        except Exception as e:
            print(f"Error loading CSV from '{self.df_path}': {e}")
            return None

        if df.empty:
            print("Error: The DataFrame is empty after loading.")
            return None

        required_columns = [
            "userId",
            "itemId",
            "timestamp",
        ]  # 'rating' column not strictly needed for this score
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(
                f"Error: Missing required columns in CSV: {', '.join(missing_columns)}"
            )
            return None

        try:
            # Avoid re-adding if already present (e.g., if df is passed around and processed)
            if "timestamp_dt" not in df.columns or df["timestamp_dt"].isnull().all():
                df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s")
            if "year_month" not in df.columns or df["year_month"].isnull().all():
                df["year_month"] = df["timestamp_dt"].dt.to_period("M")
        except Exception as e:
            print(f"Error during timestamp conversion or year-month extraction: {e}")
            return None
        return df

    def _calculate_all_movie_hype_periods(self) -> pd.DataFrame:
        """
        Calculates hype period durations for all movies using normalized rating counts.
        Returns a DataFrame of hype periods.
        """
        if self._movie_ratings_per_month.empty:
            print(
                "Warning: Monthly rating data is empty. Cannot calculate hype periods."
            )
            return pd.DataFrame(
                columns=[
                    "itemId",
                    "hype_start_month",
                    "hype_end_month",
                    "peak_month",
                    "peak_rating_count_original",
                    "peak_rating_count_normalized",
                ]
            )

        all_hype_periods_list = []
        for item_id, group in self._movie_ratings_per_month.groupby("itemId"):
            group_sorted = group.sort_values("year_month").reset_index(drop=True)
            original_ratings = group_sorted["rating_count"].values

            if len(original_ratings) == 0:  # Should not happen if group is not empty
                continue

            min_rating = original_ratings.min()
            max_rating = original_ratings.max()

            normalized_ratings = np.array(
                [0.0] * len(original_ratings)
            )  # Default to all zeros
            if max_rating > min_rating:
                normalized_ratings = (original_ratings - min_rating) / (
                    max_rating - min_rating
                )
            elif (
                len(original_ratings) == 1 and original_ratings[0] > 0
            ):  # Single rating point
                # For now, if min_rating == max_rating, normalized_ratings remains all zeros.
                pass

            # Peak detection on normalized data
            peaks_indices, _ = find_peaks(
                normalized_ratings,
                height=self.peak_norm_min_height,
                distance=self.peak_min_distance,
                prominence=self.peak_norm_min_prominence,
            )

            if len(peaks_indices) > 0:
                widths, _, left_ips, right_ips = peak_widths(
                    normalized_ratings,
                    peaks_indices,
                    rel_height=self.peak_width_rel_height,
                )

                for i, peak_idx in enumerate(peaks_indices):
                    start_idx = max(0, int(round(left_ips[i])))
                    end_idx = min(len(group_sorted) - 1, int(round(right_ips[i])))

                    if start_idx <= end_idx:
                        start_month = group_sorted.iloc[start_idx]["year_month"]
                        end_month = group_sorted.iloc[end_idx]["year_month"]

                        all_hype_periods_list.append(
                            {
                                "itemId": item_id,
                                "hype_start_month": start_month,
                                "hype_end_month": end_month,
                                "peak_month": group_sorted.iloc[peak_idx]["year_month"],
                                "peak_rating_count_original": original_ratings[
                                    peak_idx
                                ],
                                "peak_rating_count_normalized": normalized_ratings[
                                    peak_idx
                                ],
                            }
                        )

        if not all_hype_periods_list:
            print(
                f"Info: No significant hype periods found for any movie with current NORMALIZED parameters (norm_height={self.peak_norm_min_height}, min_dist={self.peak_min_distance}, norm_prominence={self.peak_norm_min_prominence})."
            )
            return pd.DataFrame(
                columns=[
                    "itemId",
                    "hype_start_month",
                    "hype_end_month",
                    "peak_month",
                    "peak_rating_count_original",
                    "peak_rating_count_normalized",
                ]
            )

        return pd.DataFrame(all_hype_periods_list)

    def get_user_trending_score(self, user_id: int) -> float:
        """
        Calculates the trending score for a given user.

        Args:
            user_id (int): The ID of the user.

        Returns:
            float: The user's trending score (0.0 to 1.0).
        """
        if self._df is None or self._df.empty:
            print("Error: DataFrame not loaded or empty. Cannot calculate score.")
            return 0.0

        if self._movie_hype_periods_df.empty:
            user_ratings_check = self._df[self._df["userId"] == user_id]
            if user_ratings_check.empty:
                print(f"User {user_id} has no ratings. Trending score is 0.")
            else:
                print(
                    f"User {user_id} has ratings, but no global movie hype periods. Trending score is 0."
                )
            return 0.0

        user_ratings = self._df[self._df["userId"] == user_id].copy()

        if user_ratings.empty:
            print(f"User {user_id} has no ratings in the dataset. Trending score is 0.")
            return 0.0

        user_ratings_merged = pd.merge(
            user_ratings, self._movie_hype_periods_df, on="itemId", how="left"
        )

        user_ratings_merged["is_match"] = (
            (
                user_ratings_merged["year_month"]
                >= user_ratings_merged["hype_start_month"]
            )
            & (
                user_ratings_merged["year_month"]
                <= user_ratings_merged["hype_end_month"]
            )
            & user_ratings_merged[
                "hype_start_month"
            ].notna()  # Ensure there was a hype period to match
        )

        num_trending_ratings = 0
        total_unique_rating_events = 0

        if not user_ratings_merged.empty and "is_match" in user_ratings_merged.columns:
            # A user's single rating event is (userId, itemId, timestamp_dt)
            # It's trending if it matches *any* hype period for that itemId.
            is_event_trending = user_ratings_merged.groupby(
                ["userId", "itemId", "timestamp_dt"]
            )["is_match"].any()
            num_trending_ratings = is_event_trending.sum()
            total_unique_rating_events = len(is_event_trending)
        else:
            # This case can happen if user_ratings exist but movie_hype_periods_df was empty,
            # or if the merge results in an empty frame for other reasons.
            total_unique_rating_events = len(
                user_ratings.drop_duplicates(
                    subset=["userId", "itemId", "timestamp_dt"]
                )
            )
            # num_trending_ratings remains 0

        if total_unique_rating_events == 0:
            return 0.0

        trending_score = num_trending_ratings / total_unique_rating_events
        return trending_score

    def get_movie_hype_periods(self, item_id: int) -> pd.DataFrame | None:
        """
        Retrieves the calculated hype periods for a specific movie.

        Args:
            item_id (int): The ID of the movie.

        Returns:
            pd.DataFrame | None: A DataFrame with hype periods for the movie, or None if not found.
        """
        if self._movie_hype_periods_df.empty:
            # print(f"Info: No hype periods calculated for any movie.")
            return None

        specific_movie_hypes = self._movie_hype_periods_df[
            self._movie_hype_periods_df["itemId"] == item_id
        ]
        if specific_movie_hypes.empty:
            # print(f"Info: No hype periods found for movie ID {item_id} with current parameters.")
            return None
        return specific_movie_hypes


# --- Example Usage ---
if __name__ == "__main__":
    csv_file_path = "ratings32m.csv"  # Make sure this path is correct

    print("Initializing UserTrendingScorer...")
    scorer = UserTrendingScorer(df_path=csv_file_path)

    if scorer._df is None:  # Check if scorer initialized correctly
        print("Scorer initialization failed. Exiting.")
    else:
        # --- Test with a list of specific users or random users ---
        # users_to_test = [86, 401, 32] # Example specific users

        # Or, select random users if dataset is large enough
        num_random_users_to_test = 3
        all_user_ids = scorer._df["userId"].unique().tolist()
        users_to_test = []
        if len(all_user_ids) >= num_random_users_to_test:
            users_to_test = random.sample(all_user_ids, num_random_users_to_test)
        elif all_user_ids:
            users_to_test = all_user_ids  # Test all available if fewer than requested

        if users_to_test:
            print(f"\nCalculating scores for users: {users_to_test}")
            for user_id in users_to_test:
                score = scorer.get_user_trending_score(user_id)
                print(f"The trending score for user {user_id} is: {score:.4f}")
        else:
            print("No users selected for testing.")

        # --- Example: Get and display hype periods for a specific movie ---
        movie_id_to_inspect = 32
        print(f"\nInspecting hype periods for movie ID: {movie_id_to_inspect}")
        hype_periods = scorer.get_movie_hype_periods(movie_id_to_inspect)
        if hype_periods is not None and not hype_periods.empty:
            print(
                f"Identified hype periods for movie {movie_id_to_inspect} (using NORMALIZED params: norm_height={scorer.peak_norm_min_height}, dist={scorer.peak_min_distance}, norm_prom={scorer.peak_norm_min_prominence}, rel_width_h={scorer.peak_width_rel_height}):"
            )
            # Display original peak rating count for context
            print(
                hype_periods[
                    [
                        "hype_start_month",
                        "hype_end_month",
                        "peak_month",
                        "peak_rating_count_original",
                    ]
                ]
            )
        else:
            # This message might be redundant if the general "No significant hype periods" was already printed.
            print(
                f"No hype periods were identified for movie {movie_id_to_inspect} with the current parameters or it had no ratings."
            )

        # Example for a movie that might have very few ratings (like movie_189071 from user's example)
        movie_id_low_volume = 189071
        print(
            f"\nInspecting hype periods for (potentially low-volume) movie ID: {movie_id_low_volume}"
        )
        hype_periods_low_volume = scorer.get_movie_hype_periods(movie_id_low_volume)
        if hype_periods_low_volume is not None and not hype_periods_low_volume.empty:
            print(f"Identified hype periods for movie {movie_id_low_volume}:")
            print(
                hype_periods_low_volume[
                    [
                        "hype_start_month",
                        "hype_end_month",
                        "peak_month",
                        "peak_rating_count_original",
                    ]
                ]
            )
        else:
            print(
                f"No hype periods were identified for movie {movie_id_low_volume} with the current parameters or it had no ratings."
            )
