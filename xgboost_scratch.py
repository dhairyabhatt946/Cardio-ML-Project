import numpy as np
from collections import Counter

class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth):
        n_samples, n_feats = X.shape
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        best_feat, best_thresh = self._best_split(X, y)

        if best_feat is None: return np.mean(y)

        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh

        return {
            'feature': best_feat,
            'threshold': best_thresh,
            'left': self._grow_tree(X[left_idxs], y[left_idxs], depth + 1),
            'right': self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        }

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feat, best_thresh = None, None
        n_samples, n_feats = X.shape

        feat_idxs = np.random.choice(n_feats, min(n_feats, 10), replace=False)

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for thr in thresholds:
                left_y = y[X[:, feat_idx] <= thr]
                right_y = y[X[:, feat_idx] > thr]

                if len(left_y) == 0 or len(right_y) == 0: continue

                mse = (np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat_idx
                    best_thresh = thr
        return best_feat, best_thresh

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict): return node
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])

class GradientBoostingFromScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = 0.0

        y_pred_log_odds = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            probabilities = 1 / (1 + np.exp(-y_pred_log_odds))

            residuals = y - probabilities

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            update = tree.predict(X)
            y_pred_log_odds += self.learning_rate * update

            if _ % 10 == 0: print(f"Tree {_} built. Error range: {np.mean(np.abs(residuals)):.4f}")

    def predict(self, X):
        y_pred_log_odds = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            y_pred_log_odds += self.learning_rate * tree.predict(X)

        proba = 1 / (1 + np.exp(-y_pred_log_odds))

        return (proba > 0.5).astype(int)