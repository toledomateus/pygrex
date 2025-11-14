import os
import re
import pickle
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Iterable, Optional, Set, Union
from contextlib import redirect_stdout, redirect_stderr
import plotly.graph_objects as go

# Library Imports
from pygrex.config import cfg
from pygrex.evaluator import ExplanationEvaluator
from pygrex.explain import (
    LORE4GroupsExplainer,
    SlidingWindowExplainer,
    RuleBasedGroupRecExplainer,
)

# Required to load the config for the explainer

# Page and State Configuration
st.set_page_config(
    page_title="Explanation & Evaluation",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Explanation & Evaluation")


#  Session State Checks
if not st.session_state.get("recommended_items", False):
    st.warning(
        "Please generate recommendations on the 'Group Recommendation' page first."
    )
    st.stop()
st.markdown("")

#  Retrieve objects from session state
data_reader = st.session_state.data_reader
group_handler = st.session_state.group_handler
model = st.session_state.trained_model
group_recommender = st.session_state.group_recommender
recommended_items = st.session_state.recommended_items
group_members = group_recommender.get_group_members()
aggregation_strategy = group_recommender.get_aggregation_strategy()

#  Caching for Expensive Functions


@st.cache_data
def load_cached_data_rules(min_support, min_confidence, rating_threshold):
    """
    Loads pre-computed association rules from the cached_rules folder.
    Returns the rules if found, None otherwise.
    """
    # Format the filename according to the pattern
    filename = f"rules_sup{min_support:.2f}_conf{min_confidence:.1f}_rating{rating_threshold:.0f}"

    # Try different file extensions
    cached_rules_dir = "cached_rules"
    possible_extensions = [".pkl", ".pickle", ".json"]

    for ext in possible_extensions:
        filepath = os.path.join(cached_rules_dir, filename + ext)
        if os.path.exists(filepath):
            try:
                data = None
                if ext in [".pkl", ".pickle"]:
                    with open(filepath, "rb") as f:
                        data = pickle.load(f)
                elif ext == ".json":
                    import json

                    with open(filepath, "r") as f:
                        data = json.load(f)
                return data
            except Exception as e:
                st.error(f"Error loading cached rules from {filepath}: {e}")
                continue

    return None


@st.cache_data
def get_user_history(rating_threshold):
    """
    Generates the user interaction history based only on the rating threshold.
    The keys of the returned dictionary are the ORIGINAL user IDs.
    """
    df_filtered = data_reader.dataset[data_reader.dataset["rating"] >= rating_threshold]

    # Group by the 'userId' column (which contains the new, consecutive IDs)
    history_by_new_id = df_filtered.groupby("userId")["itemId"].apply(set).to_dict()

    # Create the final dictionary mapping original user IDs to sets of new item IDs
    history_by_original_id = {}
    for new_id, item_set in history_by_new_id.items():
        try:
            original_id = data_reader.get_original_user_id(int(new_id))
            # The explainer needs the item IDs to be strings to match the rules
            history_by_original_id[original_id] = {str(item) for item in item_set}
        except (ValueError, KeyError):
            continue

    return history_by_original_id


@st.cache_data
def summarize_explanation_rules(
    explained_rules_info: list, all_rules_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates a list of rule dictionaries by looking up their full details
    in the main rules DataFrame.
    """
    if not explained_rules_info:
        return pd.DataFrame()

    # Get a list of all the antecedent frozensets that were found by the explainer
    found_antecedents = [rule["antecedent"] for rule in explained_rules_info]

    # Use collections.Counter to efficiently count the frequency of each unique antecedent
    antecedent_counts = Counter(found_antecedents)

    # Convert the counter object to a DataFrame: [antecedent, num_rules]
    summary = pd.DataFrame(
        antecedent_counts.items(), columns=["antecedent", "num_rules"]
    )

    # Filter the main rules DataFrame to get the full details for our found antecedents
    metrics_df = all_rules_df[all_rules_df["antecedents"].isin(summary["antecedent"])]

    # Calculate the average metrics for each antecedent
    avg_metrics = (
        metrics_df.groupby("antecedents")
        .agg(avg_confidence=("confidence", "mean"), avg_lift=("lift", "mean"))
        .reset_index()
    )

    # Merge the counts and the average metrics together
    final_summary = pd.merge(
        summary, avg_metrics, left_on="antecedent", right_on="antecedents"
    )

    # Clean up and sort the final result
    final_summary = final_summary.sort_values(by="num_rules", ascending=False)
    final_summary = final_summary.rename(
        columns={"antecedent": "because_they_interacted_with"}
    )

    # Select and reorder the final columns to be displayed
    return final_summary[
        ["because_they_interacted_with", "num_rules", "avg_confidence", "avg_lift"]
    ]


def generate_pills(id_list, max_display=10, pill_type="default"):
    """Generates styled HTML pills for a list of IDs with theme adaptation."""

    # Different styles for different pill types with better contrast
    if pill_type == "users":
        base_color = "#4285f4"  # Google Blue
        hover_color = "#5a95f5"
        shadow_color = "rgba(66, 133, 244, 0.3)"
    elif pill_type == "items":
        base_color = "#ff6b35"  # Vibrant Orange
        hover_color = "#ff8659"
        shadow_color = "rgba(255, 107, 53, 0.3)"
    else:
        base_color = "#6366f1"  # Indigo
        hover_color = "#8b5cf6"
        shadow_color = "rgba(99, 102, 241, 0.3)"

    pill_style = f"""
        display: inline-block;
        padding: 0.35rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 600;
        color: #FFFFFF;
        background: linear-gradient(135deg, {base_color} 0%, {hover_color} 100%);
        border-radius: 1.5rem;
        margin: 0.25rem 0.15rem;
        border: none;
        box-shadow: 0 2px 8px {shadow_color};
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        user-select: none;
    """

    # Enhanced hover and animation styles
    hover_style = f"""
    <style>
    .pill-container {{
        line-height: 1.6;
        margin: 0.5rem 0;
    }}
    .pill-container span:hover {{
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 6px 16px {shadow_color};
        background: linear-gradient(135deg, {hover_color} 0%, {base_color} 100%);
    }}
    .pill-container span:active {{
        transform: translateY(0) scale(1.02);
        transition: all 0.1s ease;
    }}
    </style>
    """

    # Show only the specified number, add "..." if there are more
    display_list = id_list[:max_display]
    pills_html = "".join(
        [
            f'<span style="{pill_style}" title="ID: {id}">{id}</span>'
            for id in display_list
        ]
    )

    if len(id_list) > max_display:
        more_count = len(id_list) - max_display
        more_style = (
            pill_style.replace(base_color, "#64748b")
            .replace(hover_color, "#94a3b8")
            .replace(shadow_color, "rgba(100, 116, 139, 0.2)")
        )
        more_pill = f'<span style="{more_style}" title="View all {len(id_list)} items">+{more_count} more</span>'
        pills_html += more_pill

    return hover_style + f'<div class="pill-container">{pills_html}</div>'


# A dictionary to get a friendlier name for the model class
model_name = type(model).__name__

st.markdown(
    """
<style>
    /* This style only applies to hr tags inside an element with the class "compact-list" */
    .compact-list .stElementContainer {
        margin-bottom: 1rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Context and Summary Section
with st.expander("🎯 **Recommendation Context & Results**", expanded=True):
    st.markdown(
        """
    <style>
    /* Mobile tweaks */
    @media (max-width: 768px) {
        .metric-container {
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-start;
            gap: 1.5rem !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        .custom-metric {
            margin: 0 !important;
        }

        /* Remove gap before the first metric */
        .metric-container > .custom-metric:first-child {
            margin-top: 0 !important;
        }

        /* Remove gap after the last metric */
        .metric-container > .custom-metric:last-child {
            margin-bottom: 0 !important;
        }

        .custom-metric-label {
            font-size: 0.9rem !important;
            margin-bottom: 0.3rem !important;
        }

        .custom-metric-value,
        .model-badge,
        .strategy-badge {
            font-size: 1.4rem !important;
            line-height: 1.3 !important;
        }
    }

    /* Desktop (force grid) */
    @media (min-width: 769px) {
        .metric-container {
            display: grid !important;
            grid-template-columns: repeat(4, 1fr) !important;
            gap: 0.5rem ;

            margin: 0 !important;
            padding: 0 !important;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
    .metric-container {
        padding: 1rem 0;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-color, #374151);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .custom-metric {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    .custom-metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-color-secondary, #6b7280);
        margin: 0;
        padding: 0;
        line-height: 1.25;
    }
    .custom-metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
        line-height: 1;
    }
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        font-size: 2rem;
        line-height: 1;
        margin: 0;
        padding: 0;
    }
    .strategy-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        font-size: 2rem;
        line-height: 1;
        margin: 0;
        padding: 0;
    }
    .metric-description {
        font-size: 0.75rem;
        color: var(--text-color-secondary, #9ca3af);
        margin-top: 0.25rem;
        line-height: 1.25;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="metric-container">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1.2, 1.5, 1, 1])

    with col1:
        st.markdown(
            f"""
        <div class="custom-metric">
            <div class="custom-metric">🧠 Model</div>
            <div class="model-badge"> {model_name}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        strategy_display = aggregation_strategy.name.replace("_", " ").title()
        st.markdown(
            f"""
        <div class="custom-metric">
            <div class="custom-metric">📊 Aggregation Strategy</div>
            <div class="strategy-badge">{strategy_display}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="custom-metric">
            <div class="custom-metric">👥 Group Size</div>
            <div class="custom-metric-value">{len(group_members)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="custom-metric">
            <div class="custom-metric">📝 Items Recommended</div>
            <div class="custom-metric-value">{len(recommended_items)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    list_col1, list_col2 = st.columns(2)

    with list_col1:
        st.markdown(
            '<div class="custom-metric">👥 Group Members</div>', unsafe_allow_html=True
        )
        pills_html = generate_pills(group_members, max_display=18, pill_type="users")
        st.markdown(pills_html, unsafe_allow_html=True)

    with list_col2:
        st.markdown(
            '<div class="custom-metric">🎯 Top Recommendations</div>',
            unsafe_allow_html=True,
        )
        pills_html = generate_pills(
            recommended_items, max_display=18, pill_type="items"
        )
        st.markdown(pills_html, unsafe_allow_html=True)

explainer_options = [
    "Counterfactual Explanation (Sliding Window)",
    "Rule-Based Explanation (EXPGRS)",
    "Local Model-Agnostic (LORE4GroupRS)",
]

chosen_explainer = st.selectbox(
    "**Select an explanation method:**",
    options=explainer_options,
    help="Choose the type of explanation you want to generate for the recommendation.",
)

st.markdown("")

#  Content for "Counterfactual Explanation (Sliding Window)"
if chosen_explainer == explainer_options[0]:
    st.header("Counterfactual Explanation (Sliding Window)")
    st.markdown(
        """
        This method answers the question: *"Which minimal set of items, if removed from the group's history, would cause the recommended item to disappear from the list?"* It first ranks the group's previously seen items by a composite score and then iteratively removes them to find an explanation.
        """
    )

    st.markdown("##### Configuration")
    col1, col2 = st.columns(2)
    with col1:
        target_item = st.selectbox(
            "Choose a recommended item to explain:",
            options=recommended_items,
            key="sw_target_item",
        )
    with col2:
        window_size = st.slider("Sliding Window Size", 1, 10, 3, key="sw_window_size")

    #  Component Weights for Item Ranking
    with st.expander("Configure Item Ranking Weights"):
        st.write(
            "These weights determine how to rank the items from the group's history before attempting to remove them."
        )
        weights = {}
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            weights["popularity"] = st.slider(
                "Popularity", 0.0, 2.0, 1.0, 0.1, key="w_pop"
            )
        with c2:
            weights["intensity"] = st.slider(
                "Intensity", 0.0, 2.0, 1.0, 0.1, key="w_int"
            )
        with c3:
            weights["rating"] = st.slider("Rating", 0.0, 2.0, 1.0, 0.1, key="w_rat")
        with c4:
            weights["relevance"] = st.slider(
                "Relevance", 0.0, 2.0, 1.0, 0.1, key="w_rel"
            )
        with c5:
            weights["trend"] = st.slider("Trend", 0.0, 2.0, 1.0, 0.1, key="w_tre")

    if st.button("Generate Counterfactual Explanation", key="sw_button"):
        with st.spinner(
            "Ranking items and running Sliding Window explainer... This may take a while."
        ):
            try:
                # 1. Get all items rated by the group
                items_rated_by_group = (
                    group_handler.get_rated_items_by_all_group_members(
                        group=group_members, original_data=data_reader
                    )
                )

                explainer = SlidingWindowExplainer(
                    config=None,
                    data=data_reader,
                    group_handler=group_handler,
                    members=group_members,
                    target_item=target_item,
                    aggregation_strategy=aggregation_strategy,
                    model=model,
                    window_size=window_size,
                )

                # 5. Find explanations and capture the results
                stdout_buffer = StringIO()
                stderr_buffer = StringIO()
                # Use context managers to redirect both streams to our text buffers
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    explanations = explainer.find_explanation(
                        items_rated_by_group=items_rated_by_group,
                        group_predictions=group_recommender.get_individual_predictions(),
                        top_recommendation=group_recommender.get_top_recommendation(),
                        ranking_weights=weights,
                    )

                st.session_state.sw_explanations = explanations

                st.success("✅ Counterfactual explanation process finished.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    #  Display Sliding Window Results
    if "sw_explanations" in st.session_state:
        st.markdown("")
        st.header("Explanation Results")
        explanations = st.session_state.sw_explanations

        st.markdown("")

        if not explanations:
            st.warning(
                "No counterfactual explanation could be found with the current configuration."
            )
        else:
            st.write(
                "Found the following minimal sets of items that act as explanations:"
            )

            metric_helpers = {
                "Popularity": "Measures the item's overall appeal across ALL users in the dataset. A high score indicates the item is globally popular.",
                "Intensity": "The proportion (percentage) of group members who have interacted with this item. A high score means it is a widely shared experience within the group.",
                "Rating": "The group's average rating for this item, normalized to a 0-1 scale. A high score indicates the group collectively enjoyed this item.",
                "Relevance": "Measures how influential this item was for the recommendation of the target item, based on the model's prediction scores for the group members.",
                "Trend": "The proportion of the group's interactions with this item that occurred during its detected 'hype periods' (i.e., peaks in its popularity over time).",
                "Composite Score": "The final weighted sum of all other metrics. This score is used to rank items to find the most likely counterfactual explanation.",
            }

            for call, exp_data in explanations.items():
                exp_items = exp_data["items"]  # type: ignore
                new_rec = exp_data["new_rec"]  # type: ignore
                metrics = exp_data.get("metrics", {})
                st.info(
                    f"If the group had NOT interacted with **{exp_items}**, the recommended item **{target_item}** would have been removed from the list. In its place, item **{new_rec}** would have been recommended instead."
                )
                if metrics:
                    with st.container(border=True):
                        st.markdown("##### 🔎 Contributing Item Metrics")

                        # Loop through each item in the explanation
                        for item_id, item_scores in metrics.items():
                            st.subheader(f"Item: `{item_id}`")
                            cols = st.columns(3)
                            # Define a consistent order for displaying metrics
                            display_order = [
                                "Popularity",
                                "Intensity",
                                "Rating",
                                "Relevance",
                                "Trend",
                                "Composite Score",
                            ]

                            col_index = 0
                            for metric_name in display_order:
                                if metric_name in item_scores:
                                    # Place each metric in the next available column
                                    with cols[col_index % 3]:
                                        st.metric(
                                            label=metric_name,
                                            value=f"{item_scores[metric_name]:.3f}",
                                            help=metric_helpers.get(metric_name),
                                        )
                                    col_index += 1
                st.markdown("")  # Add a separator for the next explanation

#  Content for "Rule-Based Explanation (EXPGRS)"
elif chosen_explainer == explainer_options[1]:
    st.header("Rule-Based Explanation (EXPGRS)")
    st.markdown(
        """
        This method calculates the **Model Fidelity**: the percentage of the Top-N list that can be explained by pre-computed association rules from cached files.
        """
    )
    st.markdown("##### Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_support = st.slider("Minimum Support", 0.08, 0.30, 0.10, key="rb_support")
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence", 0.1, 1.0, 0.1, key="rb_confidence"
        )
    with col3:
        rating_threshold = st.slider(
            "Rating Threshold", 0.5, 5.0, 1.0, step=0.5, key="rb_rating_threshold"
        )
    with col4:
        min_members = st.slider(
            "Minimum Satisfied Members",
            1,
            len(group_members) if group_members else 1,
            2,
            key="rb_members",
        )

    # Show the expected filename pattern
    expected_filename = f"rules_sup{min_support:.2f}_conf{min_confidence:.1f}_rating{rating_threshold:.0f}"

    if st.button("Generate Rule-Based Explanation", key="rb_button"):
        cached_rules = None
        with st.spinner("Loading cached association rules and finding explanations..."):
            try:
                # Try to load cached rules first
                cached_data_rules = load_cached_data_rules(
                    min_support, min_confidence, rating_threshold
                )
                if cached_data_rules:
                    cached_rules = cached_data_rules.get("rules")

                if cached_rules is None:
                    st.error(
                        f"⚠️ **Cached rules not found!**\n\n"
                        f"Could not find a cached rules file with the pattern: `{expected_filename}` "
                        f"in the `cached_rules/` folder.\n\n"
                        f"Please ensure the file exists with the correct naming pattern and try again."
                    )
                    st.stop()

                st.success(
                    f"✅ Successfully loaded cached rules from: `{expected_filename}`"
                )

                # Get user history
                user_history = get_user_history(rating_threshold)

                # Create explainer with cached rules
                explainer = RuleBasedGroupRecExplainer(
                    rules=cached_rules,
                    data=data_reader,
                    pool_recommendations=recommended_items,
                    members=group_members,
                    user_history=user_history,
                    min_members_threshold=min_members,
                )

                fidelity_score = explainer.find_explanation()
                advanced_fidelity_score = explainer.compute_group_fidelity_advanced()
                explanation_details = explainer.get_explanation_details()

                explanation_results = {
                    "fidelity": fidelity_score,
                    "advanced_fidelity": advanced_fidelity_score,
                    "details": explanation_details,
                }
                st.session_state.expgrs_results = explanation_results

                st.success(
                    "✅ Rule-based explanations generated successfully using cached rules!"
                )

            except Exception as e:
                st.error(f"An error occurred while processing cached rules: {e}")

        #  Display Rule-Based Results
        if "expgrs_results" in st.session_state:
            st.markdown("")
            st.header("Explanation Results")
            evaluator = ExplanationEvaluator()
            metrics = evaluator.evaluate(
                st.session_state.expgrs_results, explainer_type="EXPGRS"
            )

            #  Display Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Explanation Fidelity",
                    f"{metrics['fidelity']:.2%}",
                    help="The percentage of recommended items that could be explained by at least one rule satisfying the minimum member threshold.",
                )
            with col2:
                st.metric(
                    "Advanced Explanation Fidelity",
                    f"{st.session_state.expgrs_results['advanced_fidelity']:.2%}",
                    help="A stricter fidelity where every group member must have seen at least one item from the rule's antecedent.",
                )
            with col3:
                st.metric(
                    "Explanation Diversity (GILD)",
                    f"{metrics['gild']:.4f}",
                    help="Gaussian Intra-List Diversity. Measures how varied the explanation   are. Higher is better.",
                )

            with st.expander(
                "🔍 View Detailed Explanations for the Group", expanded=True
            ):
                details = st.session_state.get("expgrs_details", {})
                explained_items_dict = {
                    item: exp for item, exp in details.items() if exp
                }

                if not explained_items_dict:
                    st.write(
                        "No items in the list could be explained with the current settings."
                    )
                else:
                    st.write(
                        "The following items were explained by summarizing the found rules:"
                    )

                    for item_id, rules_list in explained_items_dict.items():
                        with st.container(border=True):
                            st.subheader(f"Item {item_id} was recommended because...")

                            # Ensure cached_rules is a DataFrame before passing
                            if isinstance(cached_rules, pd.DataFrame):
                                summary_df = summarize_explanation_rules(
                                    rules_list, cached_rules
                                )
                                if not summary_df.empty:
                                    summary_df["because_they_interacted_with"] = (
                                        summary_df[
                                            "because_they_interacted_with"
                                        ].apply(lambda fs: sorted(list(fs)))
                                    )

                                    # Display the clean summary table
                                    st.dataframe(
                                        summary_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "because_they_interacted_with": st.column_config.ListColumn(
                                                "Interaction Pattern (Antecedent)",
                                                help="The set of items the group previously interacted with.",
                                            ),
                                            "num_rules": st.column_config.NumberColumn(
                                                "Pattern Frequency",
                                                help="How many rules were based on this specific pattern.",
                                            ),
                                            "avg_confidence": st.column_config.ProgressColumn(
                                                "Avg. Confidence",
                                                help="The average confidence that this pattern leads to the recommendation.",
                                                format="%.2f",
                                                min_value=0,
                                                max_value=1,
                                            ),
                                            "avg_lift": st.column_config.NumberColumn(
                                                "Avg. Lift",
                                                help="How much more likely this pattern is than random chance. (Lift > 1 is good).",
                                                format="%.2f",
                                            ),
                                        },
                                    )
                            else:
                                st.warning(
                                    "Cached rules are not available as a DataFrame, so explanation summary cannot be displayed."
                                )

#  Content for "Local Model-Agnostic (LORE4GroupRS)"
elif chosen_explainer == explainer_options[2]:
    # Helper functions
    def diagnose_and_create_aligned_profiles():
        """Create properly aligned item profiles for LORE4Groups"""

        # Import required modules

        with st.spinner("🔍 Diagnosing data alignment and creating item profiles..."):
            # Step 1: Diagnose ID mismatch
            tags_df = pd.read_csv(cfg.data.tags.tags_file)

            # Step 2: Create mapping from consecutive to original IDs
            consecutive_items = set(data_reader.dataset["itemId"].unique())
            consecutive_to_original = {}
            original_to_consecutive = {}

            for consecutive_id in consecutive_items:
                try:
                    original_id = data_reader.get_original_item_id(consecutive_id)
                    consecutive_to_original[consecutive_id] = original_id
                    original_to_consecutive[original_id] = consecutive_id
                except (ValueError, KeyError):
                    continue

            # Step 3: Filter and process tags
            tags_df_filtered = tags_df[
                tags_df["movieId"].isin(original_to_consecutive.keys())
            ]

            if len(tags_df_filtered) == 0:
                st.error("❌ No tag data matches items in ratings dataset!")
                return None, None, None

            # Process tags - tokenize labels
            tag_sequences = []
            movie_ids = []

            for _, row in tags_df_filtered.iterrows():
                # Keep the full label as a single tag, just normalize case
                label = str(row["label"]).lower().strip()
                tag_sequences.append(label)
                movie_ids.append(row["movieId"])

            # Create processed dataframe
            tags_processed = pd.DataFrame(
                {"original_movieId": movie_ids, "label": tag_sequences}
            )

            # Convert to consecutive IDs
            tags_processed["movieId"] = tags_processed["original_movieId"].map(
                original_to_consecutive
            )
            tags_processed = tags_processed.dropna(subset=["movieId"])
            tags_processed["movieId"] = tags_processed["movieId"].astype(int)

            # Step 4: Create item profiles
            top_n_labels = cfg.explainer.lore4groups.top_n_labels
            top_labels = (
                tags_processed["label"]
                .value_counts()
                .nlargest(top_n_labels)
                .index.tolist()
            )
            tags_final = tags_processed[tags_processed["label"].isin(top_labels)]

            # Create item profiles (using consecutive IDs as keys)
            item_profiles = tags_final.groupby("movieId")["label"].apply(set).to_dict()
            item_profiles = {str(k): v for k, v in item_profiles.items()}

            # Create item-label matrix
            item_label_matrix = tags_final.assign(value=1).pivot_table(
                index="movieId", columns="label", values="value", fill_value=0
            )
            item_label_matrix.index = item_label_matrix.index.astype(str)

            # Step 5: Validation
            # Calculate coverage
            total_rating_items = len(consecutive_items)
            tagged_items = len(item_profiles)
            coverage = tagged_items / total_rating_items

            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", f"{total_rating_items:,}")
            with col2:
                st.metric("Tagged Items", f"{tagged_items:,}")
            with col3:
                st.metric("Coverage", f"{coverage:.1%}")

            if coverage < 0.1:
                st.warning(
                    "⚠️ Low tag coverage detected. Consider increasing `top_n_labels` parameter."
                )
            else:
                st.success("✅ Good tag coverage for explanations!")

            return item_profiles, item_label_matrix, tags_final

    def validate_user_coverage(item_profiles):
        """Validate that group members have sufficient tag coverage"""

        with st.spinner("🔍 Validating user tag coverage..."):
            coverage_stats = []

            for member_id in group_members:
                try:
                    # Convert to consecutive ID
                    user_idx = data_reader.get_new_user_id(member_id)
                    user_data = data_reader.dataset[
                        data_reader.dataset["userId"] == user_idx
                    ]

                    total_items = len(user_data)
                    tagged_items = sum(
                        1
                        for item_id in user_data["itemId"]
                        if str(item_id) in item_profiles
                    )
                    coverage = tagged_items / total_items if total_items > 0 else 0

                    coverage_stats.append(
                        {
                            "member_id": member_id,
                            "total_items": total_items,
                            "tagged_items": tagged_items,
                            "coverage": coverage,
                        }
                    )

                except Exception as e:
                    st.warning(f"Could not validate coverage for user {member_id}: {e}")
                    continue

            if coverage_stats:
                # Display coverage table
                coverage_df = pd.DataFrame(coverage_stats)
                mean_coverage = coverage_df["coverage"].mean()

                # Format the dataframe for display
                display_df = coverage_df.copy()
                display_df["coverage"] = display_df["coverage"].apply(
                    lambda x: f"{x:.1%}"
                )
                display_df.columns = [
                    "Member ID",
                    "Total Ratings",
                    "Tagged Items",
                    "Coverage",
                ]

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Show summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Coverage", f"{mean_coverage:.1%}")
                with col2:
                    good_coverage_users = sum(
                        1 for stat in coverage_stats if stat["coverage"] > 0.1
                    )
                    st.metric(
                        "Users with >10% coverage",
                        f"{good_coverage_users}/{len(coverage_stats)}",
                    )

                if mean_coverage < 0.1:
                    st.warning(
                        "⚠️ Low average tag coverage. Explanations may be limited."
                    )
                    return False
                else:
                    st.success("✅ Sufficient tag coverage for explanations!")
                    return True
            else:
                st.error("❌ Could not validate any users' tag coverage.")
                return False

    def prepare_user_history(item_profiles):
        """Prepare user history in the format expected by LORE4Groups"""

        user_hist = {}
        for user_id_orig in group_members:
            try:
                user_id_consecutive = data_reader.get_new_user_id(user_id_orig)
                # Get items (consecutive IDs) for this user
                hist_items = set(
                    data_reader.dataset[
                        data_reader.dataset["userId"] == user_id_consecutive
                    ]["itemId"].astype(str)
                )
                user_hist[user_id_orig] = hist_items
            except Exception as e:
                st.warning(f"Could not prepare history for user {user_id_orig}: {e}")
                user_hist[user_id_orig] = set()

        return user_hist

    def parse_movie_genres(
        movies_dat_path: Optional[str] = None,
        movies_data: Optional[str] = None,
    ) -> Dict[str, Dict[str, Union[str, Set[str]]]]:
        """
        Parses movie genres from a file or string in the `movies.dat` format.

        This improved version avoids code duplication, uses an efficient generator,
        and includes robust handling for malformed lines.

        The format is expected to be: MovieID::Title::Genres
        e.g., `1::Toy Story (1995)::Animation|Children's|Comedy`

        Args:
            movies_dat_path: Optional path to the `movies.dat` file.
            movies_data: Optional string containing movie data.

        Returns:
            A dictionary mapping movie IDs to their title and a set of genres.
            Returns an empty dictionary if no data source is provided or found.
        """

        # Define a generator function to process lines from any source.
        # This keeps the parsing logic in one place (DRY principle).
        def process_lines(
            lines: Iterable[str],
        ) -> Dict[str, Dict[str, Union[str, Set[str]]]]:
            return {
                parts[0]: {"title": parts[1], "genres": set(parts[2].split("|"))}
                for line in lines
                # The walrus operator (:=) assigns and checks in one step.
                # This ensures we only process well-formed lines.
                if (parts := line.strip().split("::")) and len(parts) == 3
            }

        # Handle file path source
        if movies_dat_path and os.path.exists(movies_dat_path):
            with open(movies_dat_path, "r", encoding="latin-1") as f:
                # The file object 'f' is an iterator itself, which is memory-efficient
                # as it reads the file line by line instead of all at once.
                return process_lines(f)
        # Handle direct string data source
        elif movies_data:
            return process_lines(movies_data.strip().split("\n"))

        # Return an empty dict if no valid source is provided
        return {}

    def store_genre_profiles(movies_content):
        """Helper to store movie genre data in session state"""
        st.session_state.movie_profiles = movies_content
        return movies_content

    def create_decision_tree_visualization(
        clf, feature_names, item_id, item_title="Item"
    ):
        """Create an actual decision tree visualization using plotly"""

        if not clf:
            st.warning("No decision tree available for visualization")
            return

        st.subheader(f"Decision Tree for {item_title}")

        MIN_NODE_SIZE = 30
        MAX_NODE_SIZE = 40
        # Get tree structure
        tree = clf.tree_

        # Create nodes and edges for visualization
        total_samples = tree.n_node_samples[0]
        nodes = []
        edges = []
        positions = {}

        def add_node(node_id, depth=0, pos_x=0.0):
            """Recursively add nodes and calculate positions"""

            num_samples = tree.n_node_samples[node_id]
            proportion = num_samples / total_samples
            scaled_size = MIN_NODE_SIZE + (proportion * (MAX_NODE_SIZE - MIN_NODE_SIZE))

            # Node information
            if tree.feature[node_id] != -2:  # Not a leaf
                feature_name = feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                node_text = f"{feature_name}<br>≤ {threshold:.2f}"
                node_color = "#e3f2fd"  # Light blue for internal nodes
            else:  # Leaf node
                values = tree.value[node_id][0]
                predicted_class = np.argmax(values)
                confidence = values[predicted_class] / np.sum(values)
                node_text = f"Predict: {'LIKE' if predicted_class == 1 else 'DISLIKE'}<br>Conf: {confidence:.2f}"
                node_color = (
                    "#c8e6c9" if predicted_class == 1 else "#ffcdd2"
                )  # Green for like, red for dislike

            nodes.append(
                {
                    "id": node_id,
                    "text": node_text,
                    "x": pos_x,
                    "y": -depth,
                    "color": node_color,
                    "size": scaled_size,
                }
            )

            positions[node_id] = (pos_x, -depth)

            # Add children
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            if left_child != -1:
                # Calculate positions for children
                left_x = pos_x - 1.5 / (depth + 1)
                right_x = pos_x + 1.5 / (depth + 1)

                add_node(left_child, depth + 1, left_x)
                add_node(right_child, depth + 1, right_x)

                # Add edges
                edges.extend(
                    [
                        {"from": node_id, "to": left_child, "label": "Yes"},
                        {"from": node_id, "to": right_child, "label": "No"},
                    ]
                )

        # Build the tree structure
        add_node(0)

        # Create plotly figure
        fig = go.Figure()

        # Add edges
        for edge in edges:
            from_pos = positions[edge["from"]]
            to_pos = positions[edge["to"]]

            # Add line
            fig.add_trace(
                go.Scatter(
                    x=[from_pos[0], to_pos[0]],
                    y=[from_pos[1], to_pos[1]],
                    mode="lines",
                    line=dict(color="gray", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add edge label
            mid_x = (from_pos[0] + to_pos[0]) / 2
            mid_y = (from_pos[1] + to_pos[1]) / 2
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=edge["label"],
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            )

        # Add nodes
        for node in nodes:
            fig.add_trace(
                go.Scatter(
                    x=[node["x"]],
                    y=[node["y"]],
                    mode="markers+text",
                    marker=dict(
                        size=node["size"],
                        color=node["color"],
                        line=dict(color="black", width=2),
                    ),
                    text=node["text"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=node["text"],
                )
            )

        # Update layout
        fig.update_layout(
            # title=f"Decision Tree Structure for {item_title}",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor="white",  # TODO: ADAPT TO DARK MODE
        )

        st.plotly_chart(fig, width="stretch", key=f"decision_tree_{item_id}")

    def evaluate_rule_for_item(
        rule: str, item_profiles: Dict, item_id: str, movie_genres: Optional[set] = None
    ) -> bool:
        """
        Evaluate if a rule condition is satisfied for a specific item.

        Args:
            rule: The technical rule string (e.g., "classic > 0.50" or "genre: `Comedy`")
            item_profiles: Dictionary of item profiles with features/tags
            item_id: The specific item ID being evaluated
            movie_genres: Set of genres for the movie

        Returns:
            Boolean indicating if the rule condition is satisfied
        """

        # Get item's features/profile
        item_features = item_profiles.get(str(item_id), set())

        # Handle genre rules
        genre_match = re.search(r"genre: `([^`]+)`", rule)
        if genre_match:
            target_genre = genre_match.group(1).lower()
            if movie_genres:
                movie_genres_lower = {g.lower() for g in movie_genres}
                if "Does NOT have" in rule:
                    return target_genre not in movie_genres_lower
                else:  # "Has genre:"
                    return target_genre in movie_genres_lower
            return False

        # Handle threshold rules (e.g., "classic > 0.50")
        threshold_match = re.match(r"(.+?)\s*([<>]=?)\s*(\d+\.?\d*)", rule)
        if threshold_match:
            feature, operator, threshold_str = threshold_match.groups()
            feature = feature.strip()
            # Check if item has this feature (assuming binary features for now)
            has_feature = feature in item_features

            if operator in (">", ">="):
                # Rule expects feature to be present (> 0.5 typically means "has feature")
                return has_feature
            else:  # "<", "<="
                # Rule expects feature to be absent (<= 0.5 typically means "doesn't have feature")
                return not has_feature

        # Handle tag rules
        if "Has tag:" in rule and "`" in rule:
            tag = rule.split("`")[1]
            return tag in item_features
        elif "Does NOT have tag:" in rule and "`" in rule:
            tag = rule.split("`")[1]
            return tag not in item_features

        # Fallback - if we can't parse the rule, assume it's not satisfied
        return False

    def create_decision_path(
        factual_rules, item_id, item_profiles, item_title="Item", movie_genres=None
    ):
        """Fixed decision path that properly evaluates rule conditions"""

        if not factual_rules:
            st.warning("No factual rules found for this item")
            return

        st.subheader(f"Why {item_title} is recommended:")

        # Show movie genres if available
        if movie_genres:
            st.markdown(f"**Movie Genres:** {', '.join(movie_genres)}")
            st.markdown("")

        # Create path visualization
        path_container = st.container()

        with path_container:
            # 1. Start Node: "Analyzing Item"
            st.markdown(
                """
                <div style='background-color: #e3f2fd; border-radius: 10px; padding: 15px; text-align: center; border: 2px solid #2196f3; max-width: 400px; margin: auto;'>
                    <strong>Analyzing Item</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Arrow pointing down
            st.markdown(
                "<div style='text-align: center; font-size: 30px; margin: 5px 0;'>↓</div>",
                unsafe_allow_html=True,
            )

            # 3. Process each rule and display it vertically
            st.markdown(
                "<h5 style='text-align: center;'>Evaluating Rules</h5>",
                unsafe_allow_html=True,
            )

            for i, rule in enumerate(factual_rules):
                formatted_rule = format_rule_with_context(rule, movie_genres or set())

                # Evaluate the rule for the specific item
                rule_satisfied = evaluate_rule_for_item(
                    rule, item_profiles, item_id, movie_genres
                )

                if rule_satisfied:
                    bg_color = "#c8e6c9"  # Light green for satisfied
                    border_color = "#4caf50"
                    result = "YES ✓"
                else:
                    bg_color = "#ffcdd2"  # Light red for unsatisfied
                    border_color = "#f44336"
                    result = "NO ✗"

                # Each call to st.markdown will add a new element below the previous one
                st.markdown(
                    f"""
                    <div style='background-color: {bg_color}; border-radius: 10px; padding: 15px; margin: 10px auto; border: 2px solid {border_color}; max-width: 600px;'>
                        <strong>Rule {i + 1}:</strong> {formatted_rule["readable"]}<br>
                        <small style='color: #666;'>{formatted_rule["context"]}</small><br>
                        <span style='color: {border_color}; font-weight: bold;'>Result: {result}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 4. Arrow pointing down to the final result
            st.markdown(
                "<div style='text-align: center; font-size: 30px; margin: 5px 0;'>↓</div>",
                unsafe_allow_html=True,
            )
            # 5. Final Recommendation

            # 5. Final Recommendation
            st.markdown(
                """
                <div style='background-color: #4caf50; color: white; border-radius: 10px; padding: 20px 15px; text-align: center; font-weight: bold; font-size: 18px; max-width: 400px; margin: auto;'>
                    🎬 RECOMMEND TO GROUP
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.divider(width=1)

    def create_counterfactuals(counterfactuals, item_title="Item", movie_genres=None):
        """Enhanced counterfactual explanations that consider genres"""

        if not counterfactuals:
            st.info("No counterfactual scenarios available")
            return

        st.subheader(f"What would make {item_title} NOT recommended:")

        # Show current genres context
        if movie_genres:
            st.markdown(f"**Current Genres:** {', '.join(movie_genres)}")
            st.markdown(
                "*Consider how changing these characteristics would affect the recommendation:*"
            )
            st.markdown("")

        for member_id, scenarios in counterfactuals.items():
            with st.expander(
                f"Member **{member_id}** would reject if:", expanded=False
            ):
                for scenario_idx, scenario in enumerate(
                    scenarios, 1
                ):  # Max 2 scenarios
                    st.markdown(f"**Counterfactual {scenario_idx}:**")

                    scenario_text = "If "
                    conditions = []

                    for rule in scenario:
                        formatted_rule = format_rule_with_context(
                            rule, movie_genres or set()
                        )
                        conditions.append(formatted_rule["readable"].lower())

                    scenario_text += " **AND** ".join(conditions)

                    st.markdown(f"- {scenario_text}")
                    st.markdown("  → **Result: DO NOT RECOMMEND** ✗")

                    # Add genre-based insight if available
                    if movie_genres:
                        genre_insight = get_genre_counterfactual_insight(
                            scenario, movie_genres
                        )
                        if genre_insight:
                            st.markdown(f"  💡 *Genre Insight: {genre_insight}*")

                    st.markdown("")

    GENRE_INSIGHT_PATTERNS = {
        "adventure": {
            "action": "consistent with adventure themes",
            "thriller": "consistent with adventure themes",
            "quest": "central to adventure narratives",
        },
        "comedy": {
            "family": "aligns with comedy preferences",
            "children": "aligns with comedy preferences",
            "humor": "a key element of comedy",
        },
        "drama": {
            "romance": "fits dramatic storytelling",
            "emotional": "fits dramatic storytelling",
            "relationships": "focuses on character drama",
        },
        "romance": {
            "love story": "the central focus of the genre",
            "relationships": "explores romantic dynamics",
        },
        #  New Additions
        "action": {
            "stunts": "a hallmark of the action genre",
            "explosions": "common in action films",
            "superhero": "a popular subgenre of action",
        },
        "sci-fi": {
            "space": "aligns with sci-fi settings",
            "aliens": "a common sci-fi theme",
            "futuristic": "characteristic of sci-fi worlds",
            "dystopia": "a popular sci-fi subgenre",
        },
        "horror": {
            "scary": "defines the horror experience",
            "suspense": "builds tension in horror films",
            "monster": "a classic horror element",
            "supernatural": "common in horror narratives",
        },
        "thriller": {
            "suspense": "essential for building tension",
            "mystery": "central to the thriller plot",
            "crime": "a common element in thrillers",
        },
        "fantasy": {
            "magic": "a core component of fantasy",
            "mythology": "often inspires fantasy worlds",
            "epic": "relates to the scale of fantasy stories",
        },
        "mystery": {
            "detective": "central to the investigation",
            "whodunit": "defines the mystery plot",
            "crime": "often the basis of a mystery",
        },
    }

    def format_rule_with_context(rule: str, movie_genres: Optional[set] = None) -> dict:
        """
        Enhanced rule formatting that considers movie genres using a more structured
        and scalable approach.

        Args:
            rule: The technical rule string (e.g., "classic > 0.50").
            movie_genres: A set of genres for the movie being explained.

        Returns:
            A dictionary with the formatted, human-readable rule and context.
        """
        # Get the base formatting from the simple formatter
        formatted = format_rule_with_context_simple(rule)

        # Early exit if there are no genres or the rule is not formatted
        if not movie_genres or not formatted:
            return formatted

        # Prepare data once for efficient searching
        movie_genres_lower = {g.lower() for g in movie_genres}
        rule_lower = rule.lower()

        found_contexts = set()

        # Add context by checking for relationships
        for genre in movie_genres_lower:
            # a. Check if the rule is directly about one of the movie's genres
            if genre in rule_lower:
                found_contexts.add(f"relates to the movie's '{genre.title()}' genre")

            # b. Check for pre-defined insight patterns based on the movie's genres
            if genre in GENRE_INSIGHT_PATTERNS:
                for tag, insight in GENRE_INSIGHT_PATTERNS[genre].items():
                    if tag in rule_lower:
                        found_contexts.add(insight)

        # Append all unique, found contexts to the original context string
        if found_contexts:
            # Sort for consistent output order
            additional_context = ". ".join(sorted(list(found_contexts)))
            formatted["context"] += f" ({additional_context})"

        return formatted

    def format_rule_with_context_simple(rule: str) -> dict:
        """
        Translates a technical rule string into a human-readable dictionary
        with a clear 'readable' text and 'context'.
        """

        # First, strip the member support count, if it exists, to analyze the core rule
        rule_only = re.sub(r"\s*\(\d+/\d+ members\)", "", rule).strip()

        #  Enhanced Genre Rules
        match = re.search(r"genre: `([^`]+)`", rule_only)
        if match:
            genre = match.group(1)
            if "Does NOT have" in rule_only:
                return {
                    "readable": f"the movie is NOT in the '{genre}' genre",
                    "context": f"This item is not classified as '{genre}'.",
                    "type": "negative",
                }
            else:  # "Has genre:"
                return {
                    "readable": f"the movie is in the '{genre}' genre",
                    "context": f"This item is classified as '{genre}'.",
                    "type": "positive",
                }

        #  Enhanced Tag/Feature Rules (e.g., "classic > 0.50")
        match = re.match(r"(.+?)\s*([<>]=?)\s*(\d+\.?\d*)", rule_only)
        if match:
            feature, operator, _ = match.groups()
            feature = feature.strip()

            if operator in (">", ">="):
                return {
                    "readable": f"it has the '{feature}' characteristic",
                    "context": f"The item is associated with the '{feature}' tag.",
                    "type": "positive",
                }
            else:  # "<", "<="
                return {
                    "readable": f"it does NOT have the '{feature}' characteristic",
                    "context": f"The item is not associated with the '{feature}' tag.",
                    "type": "negative",
                }

        # Fallback for any rule format that doesn't match the above
        return {
            "readable": rule,
            "context": "A technical condition was met.",
            "type": "unknown",
        }

    def get_genre_counterfactual_insight(scenario_rules, movie_genres):
        """Generate genre-based insights for counterfactual scenarios"""
        if not movie_genres:
            return None

        genre_insights = {
            "action": "More action-oriented content might be preferred.",
            "adventure": "More exciting, journey-based stories are preferred.",
            "animation": "Animated stories could capture a wider demographic.",
            "children's": "Content suitable for a younger audience is a key area for growth.",
            "comedy": "Lighter, humorous content would be better.",
            "crime": "Gritty, crime-focused narratives are a strong draw.",
            "documentary": "Factual, informative content would attract viewers interested in real-world stories.",
            "drama": "More serious, character-driven stories are favored.",
            "fantasy": "Imaginative, fantastical worlds and magic systems would be well-received.",
            "film-noir": "Stylized, cynical film-noir narratives would appeal to a niche audience.",
            "horror": "Content that creates a sense of dread and fear would be effective.",
            "musical": "Incorporating musical numbers could broaden the audience.",
            "mystery": "Intriguing mysteries that keep the audience guessing are popular.",
            "romance": "Romantic elements would improve appeal.",
            "sci-fi": "Science fiction elements would be welcome.",
            "thriller": "Suspenseful content would be more engaging.",
            "war": "Stories set against the backdrop of historical conflicts would be compelling.",
            "western": "Classic western themes of frontier justice would resonate well.",
        }

        # Look for genre-related patterns in the counterfactual rules
        for genre in movie_genres:
            genre_lower = genre.lower()
            if genre_lower in genre_insights:
                # Check if any rule contradicts this genre
                for rule in scenario_rules:
                    if genre_lower in rule.lower() and (
                        "not" in rule.lower() or "<=" in rule
                    ):
                        return genre_insights[genre_lower]

        return None

    def create_genre_group_analysis(
        factual_rules_categorized, counterfactuals, movie_genres=None
    ):
        """Enhanced group consensus analysis with genre insights"""

        st.subheader("Group Consensus Analysis")

        # Show genre context at the top
        if movie_genres:
            st.markdown(f"**Analyzing:** Movie with genres: {', '.join(movie_genres)}")
            st.markdown("")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### **Why the GROUP agrees:**")
            # print(factual_rules_categorized)
            # Display Unanimous Rules
            if factual_rules_categorized.get("unanimous"):
                st.markdown("##### Unanimous Agreement")
                for rule in factual_rules_categorized["unanimous"]:
                    st.success(f"✓ {rule}")  # Using success box for emphasis

            # Display Majority Rules
            if factual_rules_categorized.get("majority"):
                st.markdown("##### Majority Consensus")
                for rule in factual_rules_categorized["majority"]:
                    st.info(f"• {rule}")  # Using info box for emphasis

            # Display Influential Minority Rules
            if factual_rules_categorized.get("minority"):
                st.markdown("##### Minority  Factors")
                for rule in factual_rules_categorized["minority"]:
                    st.markdown(f"• {rule}")

            if not any(factual_rules_categorized.values()):
                st.markdown("No clear group consensus found.")
        with col2:
            st.markdown("#### Individual member concerns:")

            if counterfactuals:
                member_count = len(counterfactuals)
                st.markdown(f"*{member_count} members have specific concerns (φ):*")
                st.markdown('<div class="compact-list">', unsafe_allow_html=True)

                for member_id in counterfactuals.keys():
                    scenarios_count = len(counterfactuals[member_id])
                    st.markdown(
                        f"- Member {member_id}: {scenarios_count} Counterfactual(s)"
                    )

                    # Add genre-specific member insights
                    if movie_genres:
                        member_genre_insight = get_member_genre_preference(
                            counterfactuals[member_id], movie_genres
                        )
                        if member_genre_insight:
                            st.badge(f"💡 *{member_genre_insight}*")
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown("No individual concerns identified")

    def get_member_genre_preference(member_scenarios, movie_genres):
        """Get genre preference insight for individual member"""
        if not movie_genres:
            return None

        # Look at what this member would prefer instead
        rejected_features = []
        for scenario in member_scenarios:
            for rule in scenario:
                if any(genre.lower() in rule.lower() for genre in movie_genres):
                    rejected_features.append(rule)

        if rejected_features:
            return "Prefers different genre characteristics"
        return None

    def display_lore4groups(domain_name="movie", item_type="Movie"):
        """LORE4Groups display with tree visualization and genre integration"""

        if "lore_explanation" not in st.session_state:
            st.warning(
                "No LORE4Groups explanations available. Please generate explanations first."
            )
            return

        results = st.session_state.lore_explanation
        evaluator = ExplanationEvaluator()
        metrics = evaluator.evaluate(results, explainer_type="LORE4Groups")
        details = results.get("details", {})

        # Fidelity metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{item_type}s Explained", len(details))
        with col2:
            fidelity_score = metrics.get("fidelity", 0.0)
            if fidelity_score > 0.7:
                font_color = "rgb(52, 142, 79)"
                background_color = "rgba(224, 240, 229, 0.7)"
                quality_text = "High"
            elif fidelity_score > 0.3:
                font_color = "rgb(209, 126, 32)"
                background_color = "rgba(251, 236, 219, 0.7)"
                quality_text = "Moderate"
            else:
                font_color = "rgb(217, 72, 63)"
                background_color = "rgba(252, 228, 226, 0.7)"
                quality_text = "Low"

            #  2. Create the HTML string for the tag
            tag_html = f"""
            <div style="
                background-color: {background_color};
                color: {font_color};
                padding: 0.25rem 0.5rem;
                border-radius: 0.5rem;
                font-weight: 600;
                font-size: 1.5rem;
                display: inline-block;
            ">
                {quality_text}
            </div>
            """

            percentage_value = f"{fidelity_score * 100:.1f}%"

            st.markdown(
                f"""
            <div style="display: flex; flex-direction: column">
                <div style="font-size: 0.875rem; padding-top: 0.25rem">
                    Fidelity score
                </div>
                <div style="display: flex; align-items: baseline; gap: 8px; padding-bottom: 1rem;">
                    <div style="font-size: 2.25rem; font-weight: 400;">{percentage_value} </div>
                    {tag_html}
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col3:
            st.metric(
                "Explanation Diversity (GILD)",
                f"{metrics.get('gild', 0.0):.4f}",
                help="Gaussian Intra-List Diversity. Measures how varied the generated factual rules are. Higher is better.",
            )

        # Individual item explanations
        if details:
            for item_id, exp_data in details.items():
                decision_path_rules = exp_data.get("decision_path", [])
                factual_rules = exp_data.get("group_factual_rule", [])
                counterfactuals = exp_data.get("individual_counterfactuals", {})

                # Get item title and genres
                item_title, movie_genres = get_item_metadata(
                    item_id, item_type=item_type
                )

                # Main explanation container
                with st.expander(
                    f"**{item_id}** - {item_title} - Complete Explanation",
                    expanded=True,
                ):
                    threshold_info = exp_data.get("threshold_info")
                    if threshold_info and threshold_info.get("was_overridden"):
                        original_thresh = threshold_info.get(
                            "original_threshold", "N/A"
                        )
                        final_thresh = threshold_info.get("final_threshold", 0)

                        st.info(
                            f"**Note:** Your selected 'Like' threshold of `{original_thresh}` "
                            f"did not create enough class diversity for this item's local neighborhood. "
                            f"To generate a valid explanation, the threshold was automatically adjusted "
                            f"to the local mean rating of **`{final_thresh:.2f}`**.",
                            icon="ℹ️",
                        )

                    # Four enhanced tabs
                    tab1, tab2, tab3, tab4 = st.tabs(
                        [
                            "🌳 Decision Tree",
                            "📋 Decision Path",
                            "🔄 Alternatives",
                            "👥 Group Analysis",
                        ]
                    )

                    with tab1:
                        # Show the actual decision tree if available
                        if "decision_tree" in exp_data and exp_data["decision_tree"]:
                            clf = exp_data["decision_tree"]
                            feature_names = exp_data.get("feature_names", [])
                            create_decision_tree_visualization(
                                clf, feature_names, item_id, item_title
                            )
                        else:
                            st.info(
                                "Decision tree visualization not available. This requires storing the trained classifier."
                            )
                            st.markdown("""
                            **To enable tree visualization:**
                            1. Modify the LORE4Groups explainer to store the trained DecisionTreeClassifier
                            2. Include feature names in the explanation results
                            3. The tree will show the actual decision nodes and paths
                            """)

                    with tab2:
                        create_decision_path(
                            decision_path_rules,
                            item_id,
                            st.session_state.lore_item_profiles,
                            item_title,
                            movie_genres,
                        )

                    with tab3:
                        create_counterfactuals(
                            counterfactuals, item_title, movie_genres
                        )

                    with tab4:
                        create_genre_group_analysis(
                            factual_rules, counterfactuals, movie_genres
                        )
        else:
            st.info(
                "No explanations were generated. This may indicate insufficient data overlap between group members' preferences."
            )

    def get_item_metadata(item_id, data_reader=None, item_type="Item"):
        """Enhanced metadata retrieval that includes genre information"""
        try:
            # Try to get movie title and genres from session state
            if "movie_profiles" in st.session_state:
                movie_info = st.session_state.movie_profiles.get(str(item_id), {})
                if "title" in movie_info and "genres" in movie_info:
                    return movie_info["title"], movie_info["genres"]
                elif "title" in movie_info:
                    return movie_info["title"], set()

            if data_reader and hasattr(data_reader, "get_original_item_id"):
                original_id = data_reader.get_original_item_id(int(item_id))
                return f"{item_type} {original_id}", set()

            return f"{item_type} {item_id}", set()
        except Exception as e:
            st.warning(f"Metadata retrieval failed for item {item_id}: {e}")
            return f"{item_type} {item_id}", set()

    def is_genre_related_rule(rule):
        """Check if a rule relates to movie genres"""
        genre_keywords = [
            "comedy",
            "fantasy",
            "thriller",
            "romance",
            "drama",
            "musical",
            "crime",
            "war",
            "mystery",
            "sci-fi",
            "western",
            "horror",
            "children's",
            "animation",
            "film-noir",
            "adventure",
            "action",
            "documentary",
        ]
        return any(keyword in rule.lower() for keyword in genre_keywords)

    def analyze_genre_usage_in_explanations(details):
        """Analyze how genres are used across all explanations"""
        all_genres = set()
        genre_based_rules = 0
        total_rules = 0
        genre_frequency = Counter()

        for exp_data in details.values():
            rules = exp_data.get("group_factual_rule", [])
            total_rules += len(rules)

            for rule in rules:
                if is_genre_related_rule(rule):
                    genre_based_rules += 1
                    # Extract genre from rule (simplified)
                    for genre in [
                        "comedy",
                        "fantasy",
                        "thriller",
                        "romance",
                        "drama",
                        "musical",
                        "crime",
                        "war",
                        "mystery",
                        "sci-fi",
                        "western",
                        "horror",
                        "children's",
                        "animation",
                        "film-noir",
                        "adventure",
                        "action",
                        "documentary",
                    ]:
                        if genre in rule.lower():
                            all_genres.add(genre)
                            genre_frequency[genre] += 1
                            break

        return {
            "total_genres": len(all_genres),
            "genre_based_rules": int(
                (genre_based_rules / total_rules * 100) if total_rules > 0 else 0
            ),
            "top_genres": [genre for genre, _ in genre_frequency.most_common(3)],
        }

    def analyze_rule_features(factual_rules, movie_genres):
        """Analyze the features used in rules"""
        if not factual_rules:
            return {}

        feature_types = {
            "genre_related": 0,
            "tag_related": 0,
            "threshold_based": 0,
            "other": 0,
        }

        for rule in factual_rules:
            if any(genre.lower() in rule.lower() for genre in movie_genres):
                feature_types["genre_related"] += 1
            elif "tag" in rule.lower():
                feature_types["tag_related"] += 1
            elif ">" in rule or "<" in rule:
                feature_types["threshold_based"] += 1
            else:
                feature_types["other"] += 1

        return feature_types

    # Main LORE4Groups interface
    st.header("Local Rule-Based Explanations (LORE4Groups)")
    st.markdown("""
        **LORE4Groups** generates explanations by building local decision trees around each recommended item:
        
        1. **Local Neighborhood**: For each recommended item, find similar items that group members have rated
        2. **Local Model**: Build a decision tree using item features to predict ratings in this neighborhood  
        3. **Rule Extraction**: Extract interpretable rules explaining why the item was recommended
        4. **Group Consensus**: Aggregate individual explanations into group-level rules
        
        The method provides both **factual** rules (why recommended) and **counterfactual** rules (what would change the recommendation).
        """)

    # Configuration section
    st.markdown("##### Configuration")

    col1, col2 = st.columns(2)
    with col1:
        rating_threshold = st.slider(
            "Rating Threshold for 'Like'",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Ratings above this threshold are considered 'liked'",
        )
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Minimum Jaccard similarity to consider items as neighbors",
        )

    with col2:
        n_similar_for_tree = st.slider(
            "Similar Items for Tree",
            min_value=20,
            max_value=200,
            value=50,
            step=10,
            help="Number of similar items to use for building local decision tree",
        )
        min_neighbors = st.slider(
            "Minimum Neighbors",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="Minimum number of similar items needed to build explanation",
        )

    # Initialize session state for LORE profiles
    if "lore_item_profiles" not in st.session_state:
        st.session_state.lore_item_profiles = None
        st.session_state.lore_item_matrix = None
        st.session_state.lore_tags_final = None

    # Button to prepare data
    if st.button("🔧 Prepare Item Profiles", key="prepare_profiles_btn"):
        profiles_result = diagnose_and_create_aligned_profiles()
        if profiles_result[0] is not None:
            st.session_state.lore_item_profiles = profiles_result[0]
            st.session_state.lore_item_matrix = profiles_result[1]
            st.session_state.lore_tags_final = profiles_result[2]
            st.success("✅ Item profiles prepared successfully!")
        else:
            st.error("❌ Failed to prepare item profiles. Check data compatibility.")

    # Main explanation generation
    if st.session_state.lore_item_profiles is not None:
        # Validate user coverage

        with st.expander("🔍 Validate Group Coverage", expanded=False):
            coverage_ok = validate_user_coverage(st.session_state.lore_item_profiles)
            st.session_state.lore_coverage_validated = coverage_ok

        # Generate explanations button
        if st.button("Generate LORE4Groups Explanations", key="generate_lore_btn"):
            # Check if we have recommendations
            if not recommended_items:
                st.error(
                    "❌ No recommendations available. Please generate recommendations first."
                )
                st.stop()
                if st.session_state.lore_item_matrix is None:
                    st.error(
                        "❌ Item profiles are not fully prepared. Please click 'Prepare Item Profiles' again."
                    )
                    st.stop()
            with st.spinner(
                "Building local decision trees and generating explanations..."
            ):
                try:
                    cfg.explainer.lore4groups.rating_threshold_for_like = (
                        rating_threshold
                    )
                    cfg.explainer.lore4groups.similarity_threshold = (
                        similarity_threshold
                    )
                    cfg.explainer.lore4groups.n_similar_for_tree = n_similar_for_tree
                    cfg.explainer.lore4groups.min_neighbors = min_neighbors

                    # Filter recommendations to those with tag profiles
                    explainable_recs = [
                        str(item)
                        for item in recommended_items
                        if str(item) in st.session_state.lore_item_profiles
                    ]

                    if not explainable_recs:
                        st.error(
                            "❌ No recommended items have sufficient tag data for explanation!"
                        )
                        st.stop()

                    # Prepare user history
                    user_hist = prepare_user_history(
                        st.session_state.lore_item_profiles
                    )
                    movies_content = parse_movie_genres(
                        movies_dat_path="datasets/ml-1m/movies.dat"
                    )
                    store_genre_profiles(movies_content)
                    genre_profiles = {
                        movie_id: data["genres"]
                        if isinstance(data["genres"], set)
                        else set(data["genres"].split("|"))
                        if isinstance(data["genres"], str)
                        else set()
                        for movie_id, data in movies_content.items()
                    }

                    if st.session_state.lore_item_matrix is None:
                        st.error(
                            "❌ Item label matrix is not prepared. Please click 'Prepare Item Profiles' again."
                        )
                        st.stop()

                    explainer = LORE4GroupsExplainer(
                        st.session_state.lore_item_profiles,
                        st.session_state.lore_item_matrix,
                        cfg,
                        genre_profiles=genre_profiles,
                    )

                    # Prepare data structures

                    results = explainer.find_explanation(
                        st.session_state.recommended_items,
                        group_members,
                        user_hist,
                        data_reader.dataset,
                        model=model,
                        data_reader=data_reader,
                    )
                    # Store results
                    st.session_state.lore_explanation = results
                    st.success("🎉 LORE4Groups explanations generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error generating explanations: {str(e)}")
                    st.exception(e)

    else:
        st.info("👆 Please prepare item profiles first before generating explanations.")

    display_lore4groups(domain_name="movie", item_type="Movie")
