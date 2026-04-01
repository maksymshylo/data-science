"""Linear regression.

1. Generate a synthetic dataset of the form y = ax + b + noise.
2. Solve the linear regression problem for the set of points (x, y).
3. Find the parameters using gradient descent.
4. Visualize the fitted line and compare it with the one from sklearn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


def generate_data(
        a=2,
        b=5,
        n_samples=100,
        x_low=0, x_high=20, noise_low=0, noise_high=5, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.uniform(low=x_low, high=x_high, size=n_samples).reshape(-1, 1)
    noise = np.random.uniform(low=noise_low, high=noise_high, size=n_samples).reshape(-1, 1)
    y = a * X + b + noise
    return X, y


def add_bias_column(X):
    return np.hstack([np.ones_like(X), X])


def linear_regression_gd(X, y, learning_rate=0.01, iterations=4000):
    X_b = add_bias_column(X)
    m = len(y)
    theta = np.random.rand(X_b.shape[1])

    print("Initial weights:", theta)

    for _ in range(iterations):
        y_pred = X_b @ theta
        error = y.flatten() - y_pred
        gradient = -(1.0 / m) * (error @ X_b)
        theta -= learning_rate * gradient

    return theta


def plot_regression_line(X, y, theta):
    X_b = add_bias_column(X)
    y_pred = X_b @ theta

    plt.scatter(X, y, c="blue", label="Data points")
    plt.plot(X, y_pred, c="green", label="Regression line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("linear_regression.png")
    plt.close()


def main():
    # Generate and visualize data
    X, y = generate_data(random_state=42)

    # Train custom gradient descent model
    theta = linear_regression_gd(X, y)
    print("Final weights:", theta)
    print("Real weights:", [5, 2])

    # Plot fitted line
    plot_regression_line(X, y, theta)

    # Compare with sklearn
    model = SklearnLinearRegression().fit(X, y)
    print("sklearn coef_:", model.coef_.flatten())
    print("sklearn intercept_:", model.intercept_)


if __name__ == "__main__":
    main()