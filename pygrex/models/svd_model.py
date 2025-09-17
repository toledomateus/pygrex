from math import sqrt
import numpy as np
from pygrex.data_reader.data_reader import DataReader
from pygrex.models.recommender_model import RecommenderModel


class SVD(RecommenderModel):
    def __init__(
        self,
        n_factors=50,
        n_epochs=25,
        lr=0.007,
        reg=0.1,
        init_mean=0.0,
        init_std=0.1,
        random_state=42,
        early_stopping=True,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.init_mean = init_mean
        self.init_std = init_std
        self.random_state = random_state
        self.early_stopping = early_stopping

        # Model parameters
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None

        # Training history
        self.training_rmse = []

    def fit(self, data: DataReader, validation_data=None):
        df = data.dataset
        if data._num_user is None or data._num_item is None:
            raise ValueError("The number of users and items cannot be None.")
        num_users, num_items = data._num_user, data._num_item

        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)

        # Initialize parameters with better scaling
        scale = 1.0 / sqrt(self.n_factors)
        self.user_factors = rng.normal(
            self.init_mean, scale, (num_users, self.n_factors)
        )  # type: ignore
        self.item_factors = rng.normal(
            self.init_mean, scale, (num_items, self.n_factors)
        )  # type: ignore
        self.user_biases = np.zeros(num_users)
        self.item_biases = np.zeros(num_items)
        self.global_mean = df["rating"].mean()

        # Convert to list of tuples for faster iteration
        ratings_tuple = list(
            df[["userId", "itemId", "rating"]].itertuples(index=False, name=None)
        )

        # Training loop with early stopping
        best_rmse = float("inf")
        patience = 3
        patience_counter = 0

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}...")

            # Shuffle training data
            rng.shuffle(ratings_tuple)

            # SGD updates
            for user, item, rating in ratings_tuple:
                # Predict rating
                dot_product = np.dot(self.user_factors[user], self.item_factors[item])
                prediction = (
                    self.global_mean
                    + self.user_biases[user]
                    + self.item_biases[item]
                    + dot_product
                )

                # Compute error
                error = rating - prediction

                # Update biases
                self.user_biases[user] += self.lr * (
                    error - self.reg * self.user_biases[user]
                )
                self.item_biases[item] += self.lr * (
                    error - self.reg * self.item_biases[item]
                )

                # Update factors
                uf_temp = self.user_factors[user].copy()
                self.user_factors[user] += self.lr * (
                    error * self.item_factors[item] - self.reg * self.user_factors[user]
                )
                self.item_factors[item] += self.lr * (
                    error * uf_temp - self.reg * self.item_factors[item]
                )

            # Calculate training RMSE
            if epoch % 5 == 0 or epoch == self.n_epochs - 1:
                train_rmse = self.calculate_rmse(ratings_tuple)
                self.training_rmse.append(train_rmse)
                print(f"  Training RMSE: {train_rmse:.4f}")

                # Early stopping
                if self.early_stopping and validation_data is not None:
                    val_rmse = self.calculate_rmse(validation_data)
                    print(f"  Validation RMSE: {val_rmse:.4f}")

                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        print("Fit complete.")

    def calculate_rmse(self, ratings_data):
        """Calculate RMSE for given ratings data."""
        total_error = 0
        count = 0

        for user, item, rating in ratings_data:
            prediction = self.predict(user, item)
            total_error += (rating - prediction) ** 2
            count += 1

        return sqrt(total_error / count) if count > 0 else 0

    def predict(self, user_id: int | str, item_id: int | str) -> float:
        # Check that all model components are initialized
        if (
            self.user_factors is None
            or self.item_factors is None
            or self.user_biases is None
            or self.item_biases is None
            or self.global_mean is None
        ):
            raise RuntimeError("The model has not been trained yet.")

        try:
            user_id = int(user_id)
            item_id = int(item_id)
        except (ValueError, TypeError):
            # If conversion fails, return the global mean rating
            return self.global_mean

        # Make prediction
        dot_product = np.dot(self.user_factors[user_id], self.item_factors[item_id])
        prediction = (
            self.global_mean
            + self.user_biases[user_id]
            + self.item_biases[item_id]
            + dot_product
        )

        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
