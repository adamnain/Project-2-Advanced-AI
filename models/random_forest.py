import numpy as np

from models.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, 
                n_estimator: int = 100,
                max_depth: int | None = None, 
                min_samples_split: int = 2,
                max_features: int | str | None = "sqrt",
                random_state: int = 42,
                ) -> None:
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.classes: np.ndarray | None = None
        self.trees: list[DecisionTree] = []

    def _resolve_max_feature(self, n_features: int) -> int | None:
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))

        return self.max_features

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_features: list[int] | None = None) -> "RandomForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes = np.unique(y) # label
        n_samples, n_features = X.shape # ukuran data
        k = self._resolve_max_feature(n_features) # jumlah kolom per split

        rng = np.random.default_rng(self.random_state)

        self.trees = []

        for _ in range(self.n_estimator):
            index = rng.integers(0, n_samples, size = n_samples)
            seed = int(rng.integers(0, 2**31 - 1)) # seed nya random integer

            tree = DecisionTree(max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 max_features=k,
                                 random_state=seed)
            
            tree = tree.fit(X[index], y[index], categorical_features) # train tree

            self.trees.append(tree)

        return self

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        votes = np.zeros((len(X), len(self.classes)))

        for tree in self.trees:
            pred = tree.predict(X)

            for k, label in enumerate(self.classes):
                votes[:, k] += (pred == label) # cek apakah hasil prediksi sesuai label

        return votes / len(self.trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_probability(X), axis=1)]

