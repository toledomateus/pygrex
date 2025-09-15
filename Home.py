import streamlit as st

# Set the page configuration
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="PYGREX: Pyhton Explainable Group Recommendation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Welcome Page Content ---
st.title("Welcome to PY-GREX! 👋")
st.header("A Novel Library for Supporting Explanations in Group Recommendation")

st.markdown("""
Welcome to the interactive application for the **GREX** library. This tool allows you to step through the entire process of generating and explaining group recommendations.

**GREX** is designed with a modular architecture, allowing you to flexibly combine different components like data handling, recommendation models, group aggregation strategies, and explanation methods.

### How to Use This App:

1.  **📄 Data Preparation**: Use the sidebar to navigate to the Data Preparation page to upload your datasets.
2.  **🧠 Model Training**: Select and train a recommender model.
3.  **🎯 Group Recommendation**: Choose a group and an aggregation strategy to generate recommendations.
4.  **💬 Explanation & Evaluation**: View the generated explanations and analyze their quality.

Use the navigation on the left to begin!
# """)

# st.sidebar.header("⚙️ App Controls")
# st.sidebar.write("If results seem inconsistent, clear the app's memory.")

# if st.sidebar.button("⚠️ Clear Cache & Rerun"):
#     # Clears all items from the session state (like data, models, etc.)
#     st.session_state.clear()
#     # Reruns the script from the top
#     st.rerun()


st.sidebar.success("Select a page above to start.")
