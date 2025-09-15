import streamlit as st
import pandas as pd
import os
from io import StringIO

#  Library Imports
from pygrex.data_reader import DataReader, GroupInteractionHandler

#  Page Configuration
st.set_page_config(page_title="Data Preparation", page_icon="📄", layout="wide")

st.title("📄 Data Preparation")

#  Default File Paths
DEFAULT_RATINGS_PATH = "datasets/stratigis/ratings.csv"
DEFAULT_GROUPS_PATH = "datasets/stratigis/groupsWithHighRatings5.txt"

#  Session State Initialization
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data_reader = None
    st.session_state.group_handler = None
    st.session_state.num_groups = 0

#  DATA INPUT SECTION

#  Ratings Input
st.header("1. Ratings Data")
st.markdown(
    "You can upload your own ratings file or use the default **MovieLens 100k** dataset."
)
ratings_file_buffer = st.file_uploader(
    "Upload Your Ratings Data (Optional)", type=["csv"]
)

#  Group Input
st.header("2. Group Data")
group_input_method = st.radio(
    "Choose group input method:",
    ("Enter groups manually", "Upload a group file"),
    horizontal=True,
)

#  Load default group data for the text area
default_group_text = ""
if os.path.exists(DEFAULT_GROUPS_PATH) and ratings_file_buffer is None:
    with open(DEFAULT_GROUPS_PATH, "r") as f:
        default_group_text = f.read()

if group_input_method == "Enter groups manually":
    group_text_input = st.text_area(
        "Enter group members (one group per line, members separated by '_')",
        value=default_group_text,
        height=150,
    )
else:
    groups_file_buffer = st.file_uploader(
        "Upload Your Group Data (Optional)", type=["txt"]
    )

#  Preprocessing Options
st.header("3. Preprocessing")
binarize_data = st.checkbox(
    "Binarize ratings (for implicit feedback models)", value=True
)
if binarize_data:
    binary_threshold = st.number_input(
        "Rating threshold for binarization", min_value=0.0, value=1.0, step=0.5
    )

#  Main Loading Logic
st.header("4. Load and Process")
if st.button("Load and Process Data", type="primary"):
    with st.spinner("Processing data..."):
        try:
            desired_columns = ["userId", "itemId", "rating", "timestamp"]
            #  Determine which ratings file to use
            if ratings_file_buffer:
                ratings_df = pd.read_csv(
                    StringIO(ratings_file_buffer.getvalue().decode("utf-8")),
                    sep=",",
                    usecols=lambda column: column in desired_columns,
                )
            else:
                if not os.path.exists(DEFAULT_RATINGS_PATH):
                    st.error(
                        f"Default ratings file not found at: `{DEFAULT_RATINGS_PATH}`"
                    )
                    st.stop()
                ratings_df = pd.read_csv(
                    DEFAULT_RATINGS_PATH,
                    sep=",",
                    names=desired_columns,
                    skiprows=1,
                )
            ratings_df = ratings_df[desired_columns]

            #  Determine which group data to use and prepare it for the handler
            temp_dir = "temp/group_data"
            os.makedirs(temp_dir, exist_ok=True)
            groups_filepath = os.path.join(temp_dir, "current_groups.txt")

            if group_input_method == "Enter groups manually":
                with open(groups_filepath, "w") as f:
                    f.write(group_text_input)  # type: ignore
                st.session_state.group_filename = os.path.basename(groups_filepath)
            else:  # File upload method
                if groups_file_buffer:  # type: ignore
                    with open(groups_filepath, "wb") as f:
                        f.write(groups_file_buffer.getbuffer())
                    st.session_state.group_filename = groups_file_buffer.name
                else:  # Fallback to default if no file is uploaded
                    if not os.path.exists(DEFAULT_GROUPS_PATH):
                        st.error(
                            f"Default groups file not found at: `{DEFAULT_GROUPS_PATH}`"
                        )
                        st.stop()
                    groups_filepath = DEFAULT_GROUPS_PATH
                    st.session_state.group_filename = os.path.basename(groups_filepath)

            #  Instantiate library classes and process data
            data_reader = DataReader(dataframe=ratings_df)
            group_handler = GroupInteractionHandler(filepath_or_buffer=groups_filepath)

            if binarize_data:
                data_reader.binarize(binary_threshold=binary_threshold)  # type: ignore
            data_reader.make_consecutive_ids_in_dataset()

            available_groups = group_handler.read_groups(
                filename=st.session_state.group_filename
            )

            #  Store results in session state
            st.session_state.data_reader = data_reader
            st.session_state.group_handler = group_handler
            st.session_state.num_groups = len(available_groups)
            st.session_state.data_loaded = True

            st.success("✅ Data loaded and processed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.data_loaded = False


#  Enhanced Data Summary
if st.session_state.data_loaded:
    st.markdown("")
    st.header("Data Summary")

    dr = st.session_state.data_reader

    col1, col2 = st.columns(2)
    with col1:
        st.metric("👥 Unique Users", f"{dr.num_user:,}")  # type: ignore
        st.metric("📦 Unique Items", f"{dr.num_item:,}")  # type: ignore
    with col2:
        st.metric("⭐ Total Ratings", f"{len(dr.get_raw_dataset()):,}")  # type: ignore
        st.metric("👨‍👩‍👧‍👦 Number of Groups", f"{st.session_state.num_groups:,}")

    with st.expander("Processed Ratings DataFrame Head:", expanded=True):
        st.dataframe(dr.dataset.head(), hide_index=True)  # type: ignore
