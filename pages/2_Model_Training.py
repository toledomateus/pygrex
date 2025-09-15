import streamlit as st
import time

#  Library Imports
from pygrex.models import (
    ALS,
    BPR,
    ExplAutoencoderTorch,
    EMFModel,
    GMFModel,
    MLPModel,
    SVD,
    KNNBasic,
)
from pygrex.evaluator.evaluation_utils import (
    run_leave_one_out_evaluation_improved,
    run_evaluation_with_proper_split,
)

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Model Selection & Training")

#  Check if data is loaded
if not st.session_state.get("data_loaded", False):
    st.warning("⚠️ Please load data on the **📄 Data Preparation** page first.")
    st.stop()  # Stop execution if no data is loaded

#  Model Selection
st.header("1. Select a Model")
# As you add more models to your library, you can add them to this list.
model_option = st.selectbox(
    "Choose a recommendation model:",
    ("ALS", "BPR", "Autoencoder", "EMF", "GMF", "MLP", "KNN", "SVD"),
)

#  Hyperparameter Configuration
st.header("2. Configure Hyperparameters")

if model_option == "ALS":
    st.subheader("ALS (Alternating Least Squares) Parameters")

    # Create columns for a cleaner layout
    col1, col2, col3 = st.columns(3)
    with col1:
        latent_dim = st.number_input(
            "Latent Dimensions (factors)",
            min_value=1,
            max_value=500,
            value=100,
            step=10,
            help="The number of latent factors to compute.",
        )
    with col2:
        reg_term = st.number_input(
            "Regularization Term",
            min_value=0.001,
            max_value=1.0,
            value=0.001,
            step=0.001,
            format="%.3f",
            help="The regularization factor.",
        )
    with col3:
        epochs = st.number_input(
            "Epochs (iterations)",
            min_value=1,
            max_value=200,
            value=10,
            step=5,
            help="The number of ALS iterations.",
        )

elif model_option == "BPR":
    st.subheader("BPR (Bayesian Personalised Ranking) Parameters")

    # First Row
    col1_r1, col2_r1, col3_r1 = st.columns(3)
    with col1_r1:
        latent_dim = st.number_input(
            "Latent Dimensions (factors)",
            min_value=1,
            max_value=500,
            value=100,
            step=10,
            help="The number of latent factors to compute.",
        )
    with col2_r1:
        reg_term = st.number_input(
            "Regularization Term",
            min_value=0.001,
            max_value=1.0,
            value=0.001,
            step=0.001,
            format="%.3f",
            help="The regularization factor.",
        )
    with col3_r1:
        epochs = st.number_input(
            "Epochs (iterations)",
            min_value=1,
            max_value=200,
            value=10,
            step=5,
            help="The number of ALS iterations.",
        )

    # Second Row
    col1_r2, col2_r2, col3_r2 = st.columns(3)

    with col1_r2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.01,
            format="%.2f",
            help="The step size at each iteration while moving toward a minimum of the loss function.",
        )

elif model_option == "Autoencoder":
    st.subheader("Autoencoder Parameters")
    # First Row
    col1_r1, col2_r1, col3_r1 = st.columns(3)

    with col1_r1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.005,
            step=0.001,
            format="%.4f",
            help="The step size at each iteration while moving toward a minimum of the loss function.",
        )

    with col2_r1:
        weight_decay = st.number_input(
            "Weight Decay",
            min_value=0.0000001,
            max_value=0.0001,
            value=0.0000001,
            step=0.0000001,
            format="%.7f",
            help="The regularization factor to prevent overfitting by penalizing large weights.",
        )

    with col3_r1:
        hidden_layer_features = st.number_input(
            "Hidden Layer Features",
            min_value=4,
            max_value=128,
            value=8,
            step=4,
            help="The number of features in the hidden layers of the neural network.",
        )

    # Second Row
    col1_r2, col2_r2, col3_r2 = st.columns(3)

    with col1_r2:
        epochs = st.number_input(
            "Epochs (iterations)",
            min_value=1,
            max_value=200,
            value=30,
            step=5,
            help="The number of complete passes through the entire training dataset.",
        )

    with col2_r2:
        cuda = st.checkbox(
            "Use CUDA (GPU)",
            value=False,
            help="Check to use NVIDIA CUDA for GPU acceleration if available.",
        )

    with col3_r2:
        optimizer_name = st.selectbox(
            "Optimizer",
            options=["adam", "sgd", "rmsprop"],
            index=0,  # 'adam'
            help="The optimization algorithm to use for training the model.",
        )

    # Third Row
    col1_r3, col2_r3, col3_r3 = st.columns(3)

    with col1_r3:
        positive_threshold = st.number_input(
            "Positive Threshold",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="The minimum rating value considered as a 'positive' interaction.",
        )

    with col2_r3:
        knn = st.number_input(
            "K-Nearest Neighbors (KNN)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="The number of nearest neighbors to consider for KNN-based models.",
        )

    with col3_r3:
        expl = st.checkbox(
            "Enable Explanations",
            value=True,
            help="Check to enable model explanations or interpretability features.",
        )

elif model_option == "EMF":
    st.subheader("EMF (Explainable Matrix Factorisation) Parameters")

    # First Row
    col1_r1, col2_r1, col3_r1 = st.columns(3)

    with col1_r1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.4f",
            help="The step size at each iteration for the EMF model.",
        )

    with col2_r1:
        reg_term = st.number_input(
            "Regularization Term",
            min_value=0.0001,
            max_value=1.0,
            value=0.001,
            step=0.001,
            format="%.4f",
            help="The regularization factor for the main matrix factorization components.",
        )

    with col3_r1:
        expl_reg_term = st.number_input(
            "Explanation Regularization Term",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.001,
            format="%.4f",
            help="The regularization factor for the explanation components in EMF.",
        )

    # Second Row
    col1_r2, col2_r2, col3_r2 = st.columns(3)

    with col1_r2:
        latent_dim = st.number_input(
            "Latent Dimension",
            min_value=10,
            max_value=200,
            value=80,
            step=10,
            help="The number of latent factors used in the matrix factorization.",
        )

    with col2_r2:
        epochs = st.number_input(
            "Epochs (iterations)",
            min_value=1,
            max_value=200,
            value=10,
            step=5,
            help="The number of complete passes through the entire training dataset for EMF.",
        )

    with col3_r2:
        positive_threshold = st.number_input(
            "Positive Threshold",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="The minimum rating value considered as a 'positive' interaction for EMF.",
        )

    # Third Row
    col1_r3, col2_r3, col3_r3 = st.columns(3)

    with col1_r3:
        knn = st.number_input(
            "K-Nearest Neighbors (KNN)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="The number of nearest neighbors to consider for KNN-based aspects of EMF.",
        )

elif model_option == "GMF":
    st.subheader("GMF (Generalised Matrix Factorisation) Parameters")

    # First Row
    col1_r1, col2_r1, col3_r1 = st.columns(3)

    with col1_r1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.005,
            step=0.001,
            format="%.4f",
            help="The step size at each iteration for the GMF model.",
        )

    with col2_r1:
        weight_decay = st.number_input(
            "Weight Decay",
            min_value=0.0000001,
            max_value=0.0001,
            value=0.0000001,
            step=0.0000001,
            format="%.7f",
            help="The regularization factor to prevent overfitting in GMF.",
        )

    with col3_r1:
        latent_dim = st.number_input(
            "Latent Dimension",
            min_value=4,
            max_value=128,
            value=8,
            step=4,
            help="The number of latent factors for users and items in GMF.",
        )

    # Second Row
    col1_r2, col2_r2, col3_r2 = st.columns(3)

    with col1_r2:
        epochs = st.number_input(
            "Epochs (iterations)",
            min_value=1,
            max_value=200,
            value=30,
            step=5,
            help="The number of complete passes through the training data for GMF.",
        )

    with col2_r2:
        num_negative = st.number_input(
            "Number of Negative Samples",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="The number of negative samples per positive interaction during training.",
        )

    with col3_r2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=64,
            max_value=4096,
            value=1024,
            step=64,
            help="The number of samples per gradient update.",
        )

    # Third Row
    col1_r3, col2_r3, col3_r3 = st.columns(3)

    with col1_r3:
        cuda = st.checkbox(
            "Use CUDA (GPU)",
            value=False,
            help="Check to use NVIDIA CUDA for GPU acceleration if available for GMF.",
        )

    with col2_r3:
        optimizer_name = st.selectbox(
            "Optimizer",
            options=["adam", "sgd", "rmsprop"],
            index=0,  # 'adam'
            help="The optimization algorithm to use for training the GMF model.",
        )

    # col3_r3 is left empty here if no further parameters for GMF

elif model_option == "MLP":
    st.subheader("MLP (Multi-Layer Perceptron) Parameters")

    # First Row
    col1_r1, col2_r1, col3_r1 = st.columns(3)

    with col1_r1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.005,
            step=0.001,
            format="%.4f",
            help="The step size at each iteration for the MLP model.",
        )

    with col2_r1:
        weight_decay = st.number_input(
            "Weight Decay",
            min_value=0.0000001,
            max_value=0.0001,
            value=0.0000001,
            step=0.0000001,
            format="%.7f",
            help="The regularization factor to prevent overfitting in MLP.",
        )

    with col3_r1:
        latent_dim = st.number_input(
            "Latent Dimension",
            min_value=4,
            max_value=128,
            value=8,
            step=4,
            help="The number of latent factors for users and items in MLP.",
        )

    # Second Row
    col1_r2, col2_r2, col3_r2 = st.columns(3)

    with col1_r2:
        epochs = st.number_input(
            "Epochs (iterations)",
            min_value=1,
            max_value=200,
            value=30,
            step=5,
            help="The number of complete passes through the training data for MLP.",
        )

    with col2_r2:
        num_negative = st.number_input(
            "Number of Negative Samples",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="The number of negative samples per positive interaction during MLP training.",
        )

    with col3_r2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=64,
            max_value=4096,
            value=1024,
            step=64,
            help="The number of samples per gradient update for MLP.",
        )

    # Third Row
    col1_r3, col2_r3, col3_r3 = st.columns(3)

    with col1_r3:
        cuda = st.checkbox(
            "Use CUDA (GPU)",
            value=False,
            help="Check to use NVIDIA CUDA for GPU acceleration if available for MLP.",
        )

    with col2_r3:
        optimizer_name = st.selectbox(
            "Optimizer",
            options=["adam", "sgd", "rmsprop"],
            index=0,  # 'adam'
            help="The optimization algorithm to use for training the MLP model.",
        )

    # col3_r3 is left empty here if no further parameters for MLP

elif model_option == "KNN":
    st.subheader("KNN (K-Nearest Neighbors) Parameters")

    # First Row
    col1_r1, col2_r1, col3_r1 = st.columns(3)

    with col1_r1:
        k_neighbors = st.number_input(
            "Number of Neighbors (k)",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="The number of nearest neighbors to consider for making predictions.",
        )

    with col2_r1:
        min_k_neighbors = st.number_input(
            "Minimum Number of Neighbors",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="The minimum number of neighbors required to make a prediction.",
        )

    with col3_r1:
        similarity_type = st.selectbox(
            "Similarity Metric",
            options=["cosine", "pearson"],
            index=1,  # 'pearson'
            help="The similarity metric to use for finding nearest neighbors.",
        )

    # Second Row
    col1_r2, col2_r2, col3_r2 = st.columns(3)

    with col1_r2:
        boolean_user_based = st.checkbox(
            "User-Based Collaborative Filtering",
            value=True,
            help="Check to use user-based collaborative filtering; uncheck for item-based.",
        )

    # col2_r2 and col3_r2 is left empty here if no further parameters for KNN

elif model_option == "SVD":
    st.subheader("SVD Parameters")
    N_FACTORS = 64  # Sweet spot for MovieLens 1M
    N_EPOCHS = 30  # Slightly more epochs with good regularization
    LEARNING_RATE = 0.005  # Slightly lower for stability
    REGULARIZATION = 0.08  # Slightly reduced for better fit
    RANDOM_STATE = 42
    # First Row
    col1_r1, col2_r1, col3_r1 = st.columns(3)
    with col1_r1:
        n_factors = st.number_input(
            "Latent Dimensions (factors)",
            min_value=1,
            max_value=100,
            value=64,
            step=1,
            help="The number of latent factors to compute.",
        )

    with col2_r1:
        n_epochs = st.number_input(
            "Epochs (iterations)",
            min_value=1,
            max_value=35,
            value=30,
            step=1,
            help="The number of model iterations.",
        )

    with col3_r1:
        learning_rater = st.number_input(
            "Learning Rate",
            min_value=0.001,
            max_value=0.050,
            value=0.005,
            help="The step size at each iteration while moving toward a minimum of the loss function.",
        )

    # Second Row
    col1_r2, col2_r2, col3_r2 = st.columns(3)

    with col1_r2:
        early_stopping = st.checkbox(
            "Enable Early Stopping",
            value=False,
            help="Check to stop training when validation performance degrades.",
        )
    with col2_r2:
        reg = st.number_input(
            "Regularization Term",
            min_value=0.01,
            max_value=0.50,
            value=0.08,
            step=0.01,
            format="%.3f",
            help="The regularization factor.",
        )
    with col3_r2:
        init_mean = st.number_input(
            "Initialization Mean",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="The mean for initializing latent factors.",
        )
    # Third Row

    col1_r3, col2_r3, col3_r3 = st.columns(3)
    with col1_r3:
        init_std = st.number_input(
            "Initialization Standard Deviation",
            min_value=0.00,
            max_value=0.5,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="The standard deviation for initializing latent factors.",
        )
    with col2_r3:
        random_state = st.number_input(
            "Random State (Seed)",
            min_value=1,
            max_value=100,
            value=42,
            step=1,
            help="The seed for random number generation to ensure reproducibility.",
        )
    # col3_r3 is left empty here if no further parameters for SVD
else:
    st.info(f"Configuration for **{model_option}** is not yet implemented.")
    st.stop()

#  Model Training
st.header("3. Train the Model")

if st.button("Train Model", type="primary"):
    with st.spinner(f"Training **{model_option}** model... This may take a moment."):
        try:
            # Retrieve the data_reader object from session state
            data_reader = st.session_state.data_reader

            # 1. Instantiate the model with user-defined hyperparameters

            if model_option == "ALS":
                model = ALS(
                    latent_dim=latent_dim,  # type: ignore
                    reg_term=reg_term,  # type: ignore
                    epochs=epochs,
                )
            if model_option == "BPR":
                model = BPR(
                    latent_dim=latent_dim,  # type: ignore
                    reg_term=reg_term,  # type: ignore
                    epochs=epochs,  # type: ignore
                    learning_rate=learning_rate,  # type: ignore
                )
            if model_option == "Autoencoder":
                model = ExplAutoencoderTorch(
                    learning_rate=learning_rate,  # type: ignore
                    weight_decay=weight_decay,  # type: ignore
                    hidden_layer_features=hidden_layer_features,  # type: ignore
                    epochs=epochs,
                    cuda=cuda,  # type: ignore
                    optimizer_name=optimizer_name,  # type: ignore
                    positive_threshold=positive_threshold,  # type: ignore
                    knn=knn,  # type: ignore
                    expl=expl,  # type: ignore
                )
            if model_option == "EMF":
                model = EMFModel(
                    learning_rate=learning_rate,  # type: ignore
                    reg_term=reg_term,  # type: ignore
                    expl_reg_term=expl_reg_term,  # type: ignore
                    latent_dim=latent_dim,  # type: ignore
                    epochs=epochs,
                    positive_threshold=positive_threshold,  # type: ignore
                    knn=knn,  # type: ignore
                )
            if model_option == "GMF":
                model = GMFModel(
                    learning_rate=learning_rate,  # type: ignore
                    weight_decay=weight_decay,  # type: ignore
                    latent_dim=latent_dim,  # type: ignore
                    epochs=epochs,
                    num_negative=num_negative,  # type: ignore
                    batch_size=batch_size,  # type: ignore
                    cuda=cuda,  # type: ignore
                    optimizer_name=optimizer_name,  # type: ignore
                )
            if model_option == "MLP":
                model = MLPModel(
                    learning_rate=learning_rate,  # type: ignore
                    weight_decay=weight_decay,  # type: ignore
                    latent_dim=latent_dim,  # type: ignore
                    epochs=epochs,
                    num_negative=num_negative,  # type: ignore
                    batch_size=batch_size,  # type: ignore
                    cuda=cuda,  # type: ignore
                    optimizer_name=optimizer_name,  # type: ignore
                )
            if model_option == "KNN":
                model = KNNBasic(
                    k=k_neighbors,  # type: ignore
                    min_k=min_k_neighbors,  # type: ignore
                    sim_options={
                        "name": similarity_type,
                        "user_based": boolean_user_based,
                    },  # type: ignore
                )
            if model_option == "SVD":
                model = SVD(
                    n_factors=n_factors,  # type: ignore
                    n_epochs=n_epochs,  # type: ignore
                    lr=learning_rater,  # type: ignore
                    reg=reg,  # type: ignore
                    init_mean=init_mean,  # type: ignore
                    init_std=init_std,  # type: ignore
                    random_state=random_state,  # type: ignore
                    early_stopping=early_stopping,  # type: ignore
                )

            start_time = time.time()
            # 2. Fit the model using the processed dataset
            model.fit(data_reader)  # type: ignore
            end_time = time.time()
            training_time = end_time - start_time

            # 3. Store the trained model in session state for the next page

            st.session_state.trained_model = model  # type: ignore
            st.session_state.model_name = model_option

            st.success(
                f"✅ **{model_option}** model trained successfully in {training_time:.2f} seconds!"
            )

        except Exception as e:
            st.error(f"An error occurred during model training: {e}")
            if "trained_model" in st.session_state:
                del st.session_state.trained_model

if "trained_model" in st.session_state:
    st.markdown("")
    st.header("4. Offline Model Evaluation")

    with st.expander("🔬 Run Model Evaluation", expanded=True):
        st.markdown("""
        Choose your evaluation method:
        - **Leave-One-Out**: More thorough but slower (recommended for final evaluation)
        - **Train/Test Split**: Faster and practical for iterative testing
        
        **Metrics Explained:**
        - **Hit Ratio @10**: Percentage of users for whom we found at least one relevant item in top-10
        - **NDCG @10**: Measures ranking quality - higher values mean better ranking of relevant items
        """)

        # Evaluation method selection
        eval_method = st.radio(
            "Select Evaluation Method:",
            ["Train/Test Split (Fast)", "Leave-One-Out (Thorough)"],
            index=0,
        )

        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            if eval_method == "Train/Test Split (Fast)":
                test_size = st.slider("Test Set Size (%)", 10, 30, 20) / 100
            eval_top_n = st.number_input("Top-N for evaluation", 1, 20, 10)

        with col2:
            if eval_method == "Leave-One-Out (Thorough)":
                st.info("Leave-one-out will use 1 item per user for testing")

        # Run evaluation button
        eval_button_key = f"run_eval_{eval_method.replace(' ', '_').replace('(', '').replace(')', '')}"

        if st.button("Run Evaluation", key=eval_button_key, type="primary"):
            with st.spinner(
                f"Running {eval_method.lower()} evaluation... Please wait."
            ):
                try:
                    # Get the model configuration for re-instantiation
                    model_name = st.session_state.model_name

                    # Re-instantiate model with same parameters
                    if model_name == "ALS":
                        eval_model = ALS(
                            latent_dim=latent_dim, reg_term=reg_term, epochs=epochs
                        )
                    elif model_name == "BPR":
                        eval_model = BPR(
                            latent_dim=latent_dim,
                            reg_term=reg_term,
                            epochs=epochs,
                            learning_rate=learning_rate,
                        )
                    elif model_name == "Autoencoder":
                        eval_model = ExplAutoencoderTorch(
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            hidden_layer_features=hidden_layer_features,
                            epochs=epochs,
                            cuda=cuda,
                            optimizer_name=optimizer_name,
                            positive_threshold=positive_threshold,
                            knn=knn,
                            expl=expl,
                        )
                    elif model_name == "EMF":
                        eval_model = EMFModel(
                            learning_rate=learning_rate,
                            reg_term=reg_term,
                            expl_reg_term=expl_reg_term,
                            latent_dim=latent_dim,
                            epochs=epochs,
                            positive_threshold=positive_threshold,
                            knn=knn,
                        )
                    elif model_name == "GMF":
                        eval_model = GMFModel(
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            latent_dim=latent_dim,
                            epochs=epochs,
                            num_negative=num_negative,
                            batch_size=batch_size,
                            cuda=cuda,
                            optimizer_name=optimizer_name,
                        )
                    elif model_name == "MLP":
                        eval_model = MLPModel(
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            latent_dim=latent_dim,
                            epochs=epochs,
                            num_negative=num_negative,
                            batch_size=batch_size,
                            cuda=cuda,
                            optimizer_name=optimizer_name,
                        )
                    elif model_name == "KNN":
                        eval_model = KNNBasic(
                            k=k_neighbors,
                            min_k=min_k_neighbors,
                            sim_options={
                                "name": similarity_type,
                                "user_based": boolean_user_based,
                            },
                        )
                    elif model_name == "SVD":
                        eval_model = SVD(
                            n_factors=n_factors,
                            n_epochs=n_epochs,
                            lr=learning_rater,
                            reg=reg,
                            init_mean=init_mean,
                            init_std=init_std,
                            random_state=random_state,
                            early_stopping=early_stopping,
                        )
                    else:
                        st.error(f"Evaluation not implemented for {model_name}")
                        st.stop()

                    # Run the appropriate evaluation
                    if eval_method == "Leave-One-Out (Thorough)":
                        evaluation_scores = run_leave_one_out_evaluation_improved(
                            data_reader=st.session_state.data_reader,
                            model=eval_model,
                            top_n=eval_top_n,
                        )
                    else:  # Train/Test Split
                        evaluation_scores = run_evaluation_with_proper_split(
                            data_reader=st.session_state.data_reader,
                            model=eval_model,
                            test_size=test_size,
                            top_n=eval_top_n,
                        )

                    # Store results
                    st.session_state.evaluation_scores = evaluation_scores
                    st.session_state.eval_method = eval_method

                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    st.exception(e)

        # Display results if available
        if "evaluation_scores" in st.session_state:
            st.markdown("")
            st.subheader("📊 Evaluation Results")

            scores = st.session_state.evaluation_scores
            method = st.session_state.get("eval_method", "")

            # Metrics display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label=f"Hit Ratio @{eval_top_n}",
                    value=f"{scores.get('Hit Ratio', 0.0):.2%}",
                    help="Percentage of test users for whom at least one relevant item was found in top-10",
                )

            with col2:
                ndcg_value = scores.get("NDCG", scores.get("eNDCG", 0.0))
                st.metric(
                    label=f"NDCG @{eval_top_n}",
                    value=f"{ndcg_value:.4f}",
                    help="Normalized Discounted Cumulative Gain - measures ranking quality",
                )

            with col3:
                st.metric(
                    label="Evaluation Time",
                    value=f"{scores.get('evaluation_time', 0):.1f}s",
                    help="Time taken to complete the evaluation",
                )

            # Additional info
            if "test_interactions" in scores:
                st.info(
                    f"📈 Evaluated on {scores['test_interactions']:,} test interactions using {method}"
                )

            # Performance interpretation
            hit_ratio = scores.get("Hit Ratio", 0.0)
            ndcg = ndcg_value

            st.markdown("### 🎯 Performance Interpretation")

            if hit_ratio > 0.15 and ndcg > 0.08:
                st.success(
                    "🎉 Excellent performance! Your model shows strong recommendation capability."
                )
            elif hit_ratio > 0.08 and ndcg > 0.04:
                st.success("✅ Good performance! Your model is working well.")
            elif hit_ratio > 0.03 and ndcg > 0.02:
                st.warning(
                    "⚠️ Moderate performance. Consider tuning hyperparameters or trying a different model."
                )
            else:
                st.error(
                    "❌ Poor performance. The model may need significant improvements."
                )

    st.info("Navigate to the **🎯 Group Recommendation** page to continue.")
