import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from scipy.signal import (
    find_peaks,
    peak_widths,
)
import numpy as np


def plot_movie_rating_trends(
    movie_ratings_df: pd.DataFrame,
    item_ids_to_plot: list = None,
    top_n_movies: int = 5,
    output_dir: str = "movie_plots",
    hype_periods_df: pd.DataFrame = None,
):  # Optional: to overlay hype periods
    """
    Generates and saves line plots for movie rating counts over time.
    Optionally overlays identified hype periods on the plots.
    Plots original rating counts, hype periods are derived from (potentially normalized) analysis.
    """
    if movie_ratings_df.empty:
        print(
            "Movie ratings data (monthly aggregates) is empty. Cannot generate plots."
        )
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    actual_item_ids_to_plot = []
    if item_ids_to_plot is not None:
        actual_item_ids_to_plot = item_ids_to_plot
        if not actual_item_ids_to_plot:
            print(
                "Warning: item_ids_to_plot was provided but is empty. No plots will be generated."
            )
            return
    else:
        if (
            "rating_count" not in movie_ratings_df.columns
            or "itemId" not in movie_ratings_df.columns
        ):
            print(
                "Error: 'rating_count' or 'itemId' column missing in movie_ratings_df for global top N."
            )
            return

        total_ratings_per_movie = movie_ratings_df.groupby("itemId")[
            "rating_count"
        ].sum()
        actual_item_ids_to_plot = total_ratings_per_movie.nlargest(
            top_n_movies
        ).index.tolist()
        if not actual_item_ids_to_plot:
            print(
                f"Could not determine global top {top_n_movies} movies to plot (fallback)."
            )
            return
        print(
            f"Plotting for global top {len(actual_item_ids_to_plot)} movies with most ratings (fallback): {actual_item_ids_to_plot}"
        )

    if not actual_item_ids_to_plot:
        print("No movie IDs to plot.")
        return

    for item_id_idx, item_id in enumerate(actual_item_ids_to_plot):
        # movie_data contains original rating_counts for plotting
        movie_data = movie_ratings_df[movie_ratings_df["itemId"] == item_id].copy()

        if movie_data.empty:
            print(f"No monthly rating data found for itemId {item_id}. Skipping plot.")
            continue

        movie_data.sort_values("year_month", inplace=True)
        movie_data["time_axis"] = movie_data["year_month"].dt.to_timestamp()

        plt.figure(figsize=(12, 6))
        plt.plot(
            movie_data["time_axis"],
            movie_data["rating_count"],
            marker="o",
            linestyle="-",
            zorder=1,
            label="Monthly Ratings",
        )

        is_first_hype_label_for_plot = True
        if hype_periods_df is not None and not hype_periods_df.empty:
            item_hype_periods = hype_periods_df[hype_periods_df["itemId"] == item_id]
            for _, row in item_hype_periods.iterrows():
                start_date = row["hype_start_month"].to_timestamp()
                end_date = row["hype_end_month"].to_timestamp()
                label_to_use = None
                if is_first_hype_label_for_plot:
                    label_to_use = "Hype Period"
                    is_first_hype_label_for_plot = False
                plt.axvspan(
                    start_date,
                    end_date,
                    color="red",
                    alpha=0.2,
                    zorder=0,
                    label=label_to_use,
                )

        plt.title(f"Monthly Rating Counts for Movie ID: {item_id}")
        plt.xlabel("Month")
        plt.ylabel(
            "Number of Ratings (Original Scale)"
        )  # Clarify y-axis is original scale
        plt.grid(True)
        plt.xticks(rotation=45)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()

        plot_filename = os.path.join(
            output_dir, f"movie_{item_id}_ratings_trend_hype.png"
        )
        try:
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close()


def calculate_user_trending_score(
    df_path: str,
    user_id: int,
    generate_plots: bool = False,
    plot_item_ids: list = None,
    plot_top_n: int = 5,
    full_df_for_user_top_n: pd.DataFrame = None,
    # Peak parameters now apply to NORMALIZED data [0,1]
    peak_norm_min_height: float = 0.1,
    peak_min_distance: int = 3,  # Stays in months
    peak_norm_min_prominence: float = 0.05,
    peak_width_rel_height: float = 0.5,
):
    """
    Calculates a trending score for a user, using normalized data for hype period detection.
    peak_norm_min_height and peak_norm_min_prominence are for normalized [0,1] data.
    """
    _df = None
    if full_df_for_user_top_n is not None and isinstance(
        full_df_for_user_top_n, pd.DataFrame
    ):
        _df = full_df_for_user_top_n.copy()
    else:
        # Fallback logic for loading _df
        if full_df_for_user_top_n is not None:
            print(
                f"Warning: full_df_for_user_top_n was provided but is not a DataFrame (type: {type(type(full_df_for_user_top_n))}). Will attempt to load from df_path."
            )
        try:
            _df = pd.read_csv(df_path)
        except FileNotFoundError:
            print(f"Error: The file '{df_path}' was not found.")
            return 0.0, None
        except Exception as e:
            print(f"Error loading CSV from '{df_path}': {e}")
            return 0.0, None

    if _df.empty:
        print("Error: The DataFrame (_df) is empty. Cannot calculate score or plot.")
        return 0.0, None

    required_columns = [
        "userId",
        "itemId",
        "rating",
        "timestamp",
    ]  # Ensure 'rating' is checked if used by other parts, though not directly for score
    missing_columns = [col for col in required_columns if col not in _df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in CSV: {', '.join(missing_columns)}")
        return 0.0, None

    try:
        if "timestamp_dt" not in _df.columns or _df["timestamp_dt"].isnull().all():
            _df["timestamp_dt"] = pd.to_datetime(_df["timestamp"], unit="s")
        if "year_month" not in _df.columns or _df["year_month"].isnull().all():
            _df["year_month"] = _df["timestamp_dt"].dt.to_period("M")
    except Exception as e:
        print(f"Error during timestamp conversion or year-month extraction: {e}")
        return 0.0, None

    # movie_ratings_per_month contains original rating counts
    movie_ratings_per_month = (
        _df.groupby(["itemId", "year_month"], observed=False)
        .size()
        .reset_index(name="rating_count")
    )

    if movie_ratings_per_month.empty:
        print(
            "Warning: No monthly rating data could be generated for any movie. Cannot determine trending periods."
        )
        return 0.0, None

    all_hype_periods_list = []
    for item_id, group in movie_ratings_per_month.groupby("itemId"):
        group_sorted = group.sort_values("year_month").reset_index(drop=True)
        original_ratings = group_sorted["rating_count"].values

        # --- Normalization Step ---
        min_rating = original_ratings.min()
        max_rating = original_ratings.max()

        normalized_ratings = None
        if (
            max_rating > min_rating
        ):  # Avoid division by zero if all ratings are the same
            normalized_ratings = (original_ratings - min_rating) / (
                max_rating - min_rating
            )
        elif len(original_ratings) > 0:  # All ratings are same, or only one rating
            # If all ratings are the same (and non-zero), conceptually it's a flat line.
            # We can set normalized to all 0s or 0.5s. If 0s, no peaks will be found if height > 0.
            # If it's a single point, it might be a "peak" if height=0, but prominence/width are tricky.
            # For simplicity, if flat, treat as no relative peaks.
            normalized_ratings = np.zeros_like(original_ratings, dtype=float)
        else:  # No ratings for this item in group_sorted (should not happen if groupby is correct)
            continue

        # --- Peak Detection on Normalized Data ---
        # Ensure there are enough data points for peak finding with distance
        if (
            len(normalized_ratings) < peak_min_distance * 2
            and len(normalized_ratings) > 0
        ):
            pass  # Let find_peaks handle it; it might find 0 peaks.

        # Parameters peak_norm_min_height and peak_norm_min_prominence apply to normalized_ratings
        peaks_indices, properties = find_peaks(
            normalized_ratings,
            height=peak_norm_min_height,
            distance=peak_min_distance,
            prominence=peak_norm_min_prominence,
        )

        if len(peaks_indices) > 0:
            # Widths are also calculated on normalized data to determine extent relative to normalized shape
            widths, _, left_ips, right_ips = peak_widths(
                normalized_ratings, peaks_indices, rel_height=peak_width_rel_height
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
                            ],  # Store original peak height for info
                            "peak_rating_count_normalized": normalized_ratings[
                                peak_idx
                            ],  # Store normalized peak height
                        }
                    )

    movie_hype_periods_df = pd.DataFrame()
    if all_hype_periods_list:
        movie_hype_periods_df = pd.DataFrame(all_hype_periods_list)

    if movie_hype_periods_df.empty:
        print(
            f"Info: No significant hype periods found for any movie with current NORMALIZED parameters (norm_min_height={peak_norm_min_height}, min_dist={peak_min_distance}, norm_min_prominence={peak_norm_min_prominence})."
        )
        user_has_ratings_check = not _df[_df["userId"] == user_id].empty
        if user_has_ratings_check:
            print(
                f"User {user_id} has ratings, but no movie hype periods could be established. Score will be 0."
            )
        else:
            print(
                f"User {user_id} has no ratings, and no movie hype periods. Score is 0."
            )
        return 0.0, movie_hype_periods_df

    if generate_plots:
        print(
            "\n--- Generating Movie Rating Trend Plots (with Hype Periods from Normalized Analysis) ---"
        )
        ids_for_plotting = plot_item_ids
        if ids_for_plotting is None:
            if _df is None or not isinstance(_df, pd.DataFrame):
                print(
                    "Error: A valid DataFrame (_df) is required to determine user's top N movies for plotting."
                )
                ids_for_plotting = []
            else:
                user_specific_ratings = _df[_df["userId"] == user_id]
                if not user_specific_ratings.empty:
                    user_movie_counts = user_specific_ratings["itemId"].value_counts()
                    ids_for_plotting = user_movie_counts.nlargest(
                        plot_top_n
                    ).index.tolist()
                    if not ids_for_plotting:
                        print(
                            f"User {user_id} has rated movies, but couldn't determine their top {plot_top_n} to plot."
                        )
                    else:
                        print(
                            f"Plotting for top {len(ids_for_plotting)} movies most rated by user {user_id}: {ids_for_plotting}"
                        )
                else:
                    print(
                        f"User {user_id} has no ratings. Cannot determine their top movies to plot."
                    )
                    ids_for_plotting = []

        if ids_for_plotting:
            # Plotting function uses original counts from movie_ratings_per_month
            plot_movie_rating_trends(
                movie_ratings_df=movie_ratings_per_month.copy(),
                item_ids_to_plot=ids_for_plotting,
                hype_periods_df=movie_hype_periods_df.copy(),
            )
        else:
            print("No specific movies selected for plotting.")
        print("--- Finished Generating Plots ---\n")

    user_ratings = _df[_df["userId"] == user_id].copy()

    if user_ratings.empty:
        print(f"User {user_id} has no ratings in the dataset. Trending score is 0.")
        return 0.0, movie_hype_periods_df

    user_ratings_merged = pd.merge(
        user_ratings, movie_hype_periods_df, on="itemId", how="left"
    )

    user_ratings_merged["is_match"] = (
        (user_ratings_merged["year_month"] >= user_ratings_merged["hype_start_month"])
        & (user_ratings_merged["year_month"] <= user_ratings_merged["hype_end_month"])
        & user_ratings_merged["hype_start_month"].notna()
    )

    if not user_ratings_merged.empty and "is_match" in user_ratings_merged.columns:
        is_event_trending = user_ratings_merged.groupby(
            ["userId", "itemId", "timestamp_dt"]
        )["is_match"].any()
        num_trending_ratings = is_event_trending.sum()
        total_unique_rating_events = len(is_event_trending)
    else:
        num_trending_ratings = 0
        total_unique_rating_events = len(
            user_ratings.drop_duplicates(subset=["userId", "itemId", "timestamp_dt"])
        )

    if total_unique_rating_events == 0:
        return 0.0, movie_hype_periods_df

    trending_score = num_trending_ratings / total_unique_rating_events
    return trending_score, movie_hype_periods_df


if __name__ == "__main__":
    csv_file_path = "datasets/ml-32m/ratings.csv"
    num_random_users_to_test = 2

    # Parameters for local peak detection (APPLY TO NORMALIZED DATA [0,1])
    param_peak_norm_min_height = (
        0.3  # Peak must be at least 20% of the movie's normalized range
    )
    param_peak_min_distance = (
        9  # Peaks must be at least 6 months apart (same as before, absolute)
    )
    param_peak_norm_min_prominence = (
        0.2  # Peak must stand out by 10% of normalized range
    )
    param_peak_width_rel_height = (
        0.6  # Measure width at 50% of normalized prominence-relative height
    )

    main_df_for_script = None
    try:
        main_df_for_script = pd.read_csv(csv_file_path)
        if main_df_for_script.empty:
            print(f"Exiting: Main DataFrame loaded from '{csv_file_path}' is empty.")
            exit()
        if "timestamp" in main_df_for_script.columns:
            main_df_for_script["timestamp_dt"] = pd.to_datetime(
                main_df_for_script["timestamp"], unit="s"
            )
            main_df_for_script["year_month"] = main_df_for_script[
                "timestamp_dt"
            ].dt.to_period("M")
        else:
            print("Exiting: 'timestamp' column missing in the main DataFrame.")
            exit()
    except FileNotFoundError:
        print(f"Exiting: The file '{csv_file_path}' was not found.")
        exit()
    except Exception as e:
        print(f"Exiting: Error loading or pre-processing CSV '{csv_file_path}': {e}")
        exit()

    if "userId" in main_df_for_script.columns:
        all_user_ids = main_df_for_script["userId"].unique().tolist()
        if len(all_user_ids) >= num_random_users_to_test:
            users_to_test = random.sample(all_user_ids, num_random_users_to_test)
            print(f"Selected random users for testing: {users_to_test}")
        elif all_user_ids:
            users_to_test = all_user_ids
            print(
                f"Warning: Fewer users in dataset than requested. Testing with all {len(all_user_ids)} users: {users_to_test}"
            )
        else:
            print(
                "Error: No user IDs found in the dataset. Cannot select random users."
            )
            users_to_test = []
    else:
        print(
            "Error: 'userId' column not found in the dataset. Cannot select random users."
        )
        users_to_test = []

    first_run_hype_periods = None

    for i, user in enumerate(users_to_test):
        print(f"\nProcessing user {user}...")
        score, calculated_hype_periods = calculate_user_trending_score(
            df_path=csv_file_path,
            user_id=user,
            generate_plots=True,
            plot_item_ids=None,
            plot_top_n=2,
            full_df_for_user_top_n=main_df_for_script.copy(),
            peak_norm_min_height=param_peak_norm_min_height,
            peak_min_distance=param_peak_min_distance,
            peak_norm_min_prominence=param_peak_norm_min_prominence,
            peak_width_rel_height=param_peak_width_rel_height,
        )
        print(
            f"The trending score for user {user} (hype durations, normalized) is: {score:.4f}"
        )
        if i == 0 and calculated_hype_periods is not None:
            first_run_hype_periods = calculated_hype_periods
