"""Decision Tree (2D).

1. Generate two synthetic datasets on a plane and add class labels (2 classes).
2. Divide the points into classes using the CART (gini criterion) algorithm.
3. Limit the depth of the tree (optional).
4. Check the accuracy of the resulting algorithm using the accuracy metric on the test set.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


def generate_synthetic_data(n_samples=200, noise=0.6, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    labels = np.random.randint(0, 2, n_samples)
    x = (np.random.rand(n_samples) + labels) / 2
    y = -x + np.random.rand(n_samples) * noise

    x = (x.reshape(-1, 1) * 200).astype(int)
    y = (y.reshape(-1, 1) * 200).astype(int)
    labels = labels.reshape(-1, 1)

    return np.hstack([x, y, labels])


def plot_dataset(data):
    plt.scatter(data[:, 0][data[:, 2] == 1], data[:, 1][data[:, 2] == 1], label="Class 1")
    plt.scatter(data[:, 0][data[:, 2] == 0], data[:, 1][data[:, 2] == 0], label="Class 0")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("decision_tree.png")


def split_train_test(data, train_size=70):
    np.random.shuffle(data)
    train_x = data[:, [0, 1]][:train_size]
    train_y = data[:, 2][:train_size]
    test_x = data[:, [0, 1]][train_size:]
    test_y = data[:, 2][train_size:]
    return train_x, train_y, test_x, test_y


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if self.n_feats is None else min(self.n_feats, X.shape[1])
        self.root = self._create_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _create_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._create_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._create_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feat_idx], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        child_entropy = (n_l / n) * entropy(y[left_idxs]) + (n_r / n) * entropy(y[right_idxs])
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]


def plot_data_and_predictions(data, my_clf, sklearn_clf):
    X = data[:, :2]
    y_true = data[:, 2]

    my_pred = my_clf.predict(X)
    sk_pred = sklearn_clf.predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm", edgecolor="k")
    axes[0].set_title("Original data")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].scatter(X[:, 0], X[:, 1], c=my_pred, cmap="coolwarm", edgecolor="k")
    axes[1].set_title("My solution predictions")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    axes[2].scatter(X[:, 0], X[:, 1], c=sk_pred, cmap="coolwarm", edgecolor="k")
    axes[2].set_title("sklearn predictions")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    plt.tight_layout()
    plt.savefig("decision_tree.png")

def main():
    data = generate_synthetic_data(random_state=42)

    train_x, train_y, test_x, test_y = split_train_test(data)

    my_clf = DecisionTree(max_depth=10)
    my_clf.fit(train_x, train_y)
    my_pred = my_clf.predict(test_x)
    print("Custom model accuracy:", accuracy(test_y, my_pred))

    sklearn_clf = DecisionTreeClassifier(random_state=32).fit(train_x, train_y)
    print("sklearn accuracy:", sklearn_clf.score(test_x, test_y))

    plot_data_and_predictions(data, my_clf, sklearn_clf)


if __name__ == "__main__":
    main()