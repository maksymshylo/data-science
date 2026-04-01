"""Logistic Regression (2D).

1. Generate two synthetic datasets on a plane and add class labels (2 classes).
3. Split the data into a training and a test set. Implement logistic regression.
4. Check the accuracy of the resulting algorithm using the accuracy metric (on the test set).
5. Visualize the decision boundary and compare it with the one from sklearn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

def generate_synthetic_data(n_samples=100, noise=0.9, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    labels = np.random.randint(0, 2, n_samples)
    x = (np.random.rand(n_samples) + labels) / 2
    y = x + np.random.rand(n_samples) * noise

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    labels = labels.reshape(-1, 1)

    return np.hstack([np.ones_like(x), x, y, labels])


def split_train_test(data, train_size=70):
    np.random.shuffle(data)
    train_x = data[:, [0, 1, 2]][:train_size]
    train_y = data[:, 3][:train_size]
    test_x = data[:, [0, 1, 2]][train_size:]
    test_y = data[:, 3][train_size:]
    return train_x, train_y, test_x, test_y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_logistic_regression(X, y, learning_rate=0.01, n_iters=5000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for _ in range(n_iters):
        y_pred = sigmoid(X @ weights)
        gradient = (X.T @ (y_pred - y)) / n_samples
        weights -= learning_rate * gradient

    return weights


def predict(X, weights):
    probabilities = sigmoid(X @ weights)
    return (probabilities > 0.5).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def decision_boundary(weights, x_values):
    c, a, b = weights
    return -(a * x_values + c) / b


def plot_both_boundaries(data, my_weights, sklearn_model):
    x_values = np.linspace(0, 1, 100)

    my_y = decision_boundary(my_weights, x_values)
    sk_c = sklearn_model.intercept_[0]
    sk_a, sk_b = sklearn_model.coef_[0]
    sk_y = -(sk_a * x_values + sk_c) / sk_b

    plt.scatter(data[:, 1][data[:, 3] == 1], data[:, 2][data[:, 3] == 1], label="Class 1")
    plt.scatter(data[:, 1][data[:, 3] == 0], data[:, 2][data[:, 3] == 0], label="Class 0")
    plt.plot(x_values, my_y, label="My solution", linewidth=2, color="red")
    plt.plot(x_values, sk_y, label="sklearn", linestyle="--", linewidth=2, color="green")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("logistic_regression.png")


def main():
    data = generate_synthetic_data(random_state=42)
    train_x, train_y, test_x, test_y = split_train_test(data)

    my_weights = train_logistic_regression(train_x, train_y)
    my_pred = predict(test_x, my_weights)

    print("Custom model accuracy:", accuracy(test_y, my_pred))
    print("Custom model weights:", my_weights)

    sklearn_model = SklearnLogisticRegression().fit(train_x[:, 1:], train_y)
    print("sklearn coefficients:", sklearn_model.coef_)
    print("sklearn intercept:", sklearn_model.intercept_)
    print("sklearn accuracy:", sklearn_model.score(test_x[:, 1:], test_y))

    plot_both_boundaries(data, my_weights, sklearn_model)


if __name__ == "__main__":
    main()