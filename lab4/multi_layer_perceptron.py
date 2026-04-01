"""Multilayer perceptron.

Solve a classification problem using the data you worked with in Lab No. 2.
Compare the result with the previously obtained one.
"""
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_ml_model(ml_model, parameters, X_train, y_train, X_test, y_test, model_name):
    """
    Train and evaluate a machine learning model with hyperparameter tuning.

    Parameters:
    -----------
    ml_model : estimator object
        The machine learning model to train
    parameters : dict
        Grid search parameters
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    model_name : str
        Name of the model for saving figures

    Returns:
    --------
    ml_model : GridSearchCV object
        Trained model with best parameters
    """
    # Create pipeline with scaling and model
    steps = [
        ("scaler", StandardScaler()),  # Feature scaling
        ("model", ml_model),
    ]
    model_pipe = Pipeline(steps)

    # Perform grid search with cross-validation
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}...")
    print(f"{'=' * 60}")

    ml_model = GridSearchCV(model_pipe, parameters, cv=3, n_jobs=-1, verbose=1)
    ml_model = ml_model.fit(X_train, y_train.ravel())

    # Predictions
    y_pred_train = ml_model.predict(X_train)
    y_pred_test = ml_model.predict(X_test)

    # Calculate accuracies
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # Display results
    print(f"\nTraining set accuracy: {accuracy_train:.4f}")
    print(f"Test set accuracy: {accuracy_test:.4f}")
    print(f"\nBest parameters: {ml_model.best_params_}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(
        cm, annot=True, cmap="viridis", fmt=".0f", ax=ax, cbar_kws={"label": "Count"}
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    df = pd.read_csv("bird.csv")
    df = df.fillna(df.mean(numeric_only=True))
    df = df.drop(columns=["id"])
    feature_columns = [
        "huml",
        "humw",
        "ulnal",
        "ulnaw",
        "feml",
        "femw",
        "tibl",
        "tibw",
        "tarl",
        "tarw",
    ]

    X = df.drop("type", axis=1).values
    y = df["type"].values.reshape(-1, 1)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    run_ml_model(
        MLPClassifier(),
        {
            "model__activation": ["identity", "logistic", "tanh", "relu"],
            "model__hidden_layer_sizes": [1000, 2000],
            "model__solver": ["sgd", "adam"],
            "model__learning_rate": ["constant", "invscaling", "adaptive"],
            "model__random_state": [0],
        },
        X_train,
        y_train,
        X_test,
        y_test,
        "Multilayer Perceptron",
    )


if __name__ == "__main__":
    main()
