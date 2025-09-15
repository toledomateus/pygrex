from pathlib import Path
import streamlit as st
import pandas as pd

from pygrex.recommender.group_recommender import GroupRecommender
from pygrex.utils.aggregation_strategy import AggregationStrategy

st.set_page_config(page_title="Group Recommendation", page_icon="🎯", layout="wide")
st.title("🎯 Group Recommendation")

#  Session State Checks
# Ensure data is loaded and a model is trained before proceeding.
if not st.session_state.get("data_loaded", False):
    st.warning("⚠️ Please load data on the **📄 Data Preparation** page first.")
    st.stop()
if not st.session_state.get("trained_model", False):
    st.warning("⚠️ Please train a model on the **🧠 Model Training** page first.")
    st.stop()

#  Retrieve objects from session state
data_reader = st.session_state.data_reader
group_handler = st.session_state.group_handler
model = st.session_state.trained_model
model_name = st.session_state.model_name

#  Recommendation Setup
st.header("1. Select a Group and Strategy")

group_filename = st.session_state.group_filename

try:
    available_groups = group_handler.read_groups(filename=group_filename)

    col1, col2 = st.columns(2)
    with col1:
        selected_group_id = st.selectbox(
            "Choose a group:",
            options=available_groups,
            help="These groups were loaded from your group data file.",
        )

    # Parse and display members of the selected group
    if selected_group_id:
        group_members = group_handler.parse_group_members(selected_group_id)
        st.write("👥 **Group Members:**", ", ".join(map(str, group_members)))

    with col2:
        # Use the AggregationStrategy Enum to populate the selectbox
        agg_strategy_enum = st.selectbox(
            "Choose an aggregation strategy:",
            options=list(AggregationStrategy),
            format_func=lambda x: x.name.replace("_", " ").title(),
            help="Select the method for combining individual member preferences.",
        )

    # Conditional Input for Most Respected Person
    mrp_id = None
    if agg_strategy_enum == AggregationStrategy.MOST_RESPECTED_PERSON:
        mrp_id = st.selectbox(
            "Select the Most Respected Person:",
            options=group_members,  # type: ignore
            help="This user's preferences will solely determine the group recommendation.",
        )

except Exception as e:
    st.error(f"Could not read groups from file '{group_filename}'. Error: {e}")
    st.stop()

# Top-K Configuration
st.header("2. Specify Number of Recommendations")
top_k = st.slider(
    "Number of items to recommend (Top-K):",
    min_value=1,
    max_value=50,
    value=10,
    help="Adjust the slider to change the length of the final recommendation list.",
)

# Generate Recommendations
st.header("3. Generate and View Recommendations")

if st.button("Generate Group Recommendations", type="primary"):
    if not selected_group_id:
        st.warning("Please select a group first.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                # 1. Instantiate the GroupRecommender
                group_recommender = GroupRecommender(data=data_reader)

                # 2. Setup the recommendation process
                group_recommender.setup_recommendation(
                    model=model,
                    members=group_members,  # type: ignore
                    data=data_reader,
                    aggregation_strategy=agg_strategy_enum,
                    most_respected_person=mrp_id,
                )

                # 3. Get the final recommendation list
                recommended_items = group_recommender.get_group_recommendations(
                    top_k=top_k
                )

                # Store the recommender instance for the explanation page
                st.session_state.group_recommender = group_recommender
                st.session_state.recommended_items = recommended_items

                st.success("✅ Recommendations generated successfully!")

            except Exception as e:
                st.error(f"An error occurred while generating recommendations: {e}")


#  Display Results
if "recommended_items" in st.session_state:
    st.markdown("")
    st.subheader(f"Top {top_k} Recommended Items")

    recommender = st.session_state.group_recommender
    scores = recommender.get_recommendation_scores()

    # Create a DataFrame for nice display
    rec_data = []
    for i, item_id in enumerate(st.session_state.recommended_items):  # type: ignore
        rec_data.append(
            {
                "Rank": i + 1,
                "Item ID": item_id,
                "Aggregated Score": scores.get(item_id, 0.0),
            }
        )

    if not rec_data:
        st.info("No recommendations were generated for this group.")
    else:
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

        #  Show detailed individual predictions
        with st.expander("🔍 View Individual Predictions"):
            individual_preds = recommender.get_individual_predictions()
            if individual_preds:
                # Convert to a more readable DataFrame
                df_preds = pd.DataFrame(
                    individual_preds
                ).T  # Transpose to have users as rows
                df_preds.index.name = "User ID"
                st.write(
                    "Predicted scores (1-5 scale) for each user on items in the candidate pool:"
                )
                st.dataframe(df_preds.head(10))
            else:
                st.write("No individual predictions available.")

    st.info(
        "Navigate to the **💬 Explanation & Evaluation** page to analyze these recommendations."
    )
