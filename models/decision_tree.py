import numpy as np

class Node:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value  # menyimpan proporsi tiap kelas
        self.feature: int | None = None # index kolom untuk split
        self.threshold: float | None = None # batas untuk split
        self.left: Node | None = None   # child node
        self.right: Node | None = None  # child node
        self.nan_go_left: bool = True # data NaN masuk ke kiri

    def is_leaf(self) -> bool:
        return self.left is None # leaf node tidak punya child


def gini(counts: np.ndarray) -> float:
    # parameternya berupa jumlah data tiap kelas
    n = counts.sum()
    if n == 0:
        return 0.0 # biar ga ada pembagian dengan 0

    p = counts / n

    return 1.0 - np.sum(p ** 2)

def weighted_gini(count_left: np.ndarray, count_right: np.ndarray) -> float:
    # parameter nya jumlah data di kiri dan kanan
    n_left = count_left.sum()
    n_right = count_right.sum()

    return (n_left * gini(count_left) + n_right * gini(count_right)) / (n_left + n_right)

class DecisionTree:
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, max_features: int | None = None, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.classes: np.ndarray | None = None # buat nyimpan label target
        self.root: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_features: list[int] | None = None) -> "DecissionTree":
        X = np.asarray(X, dtype=float)
        label, index = np.unique(y, return_inverse=True) # ambil target label secara unik dan index nya

        self.classes = label

        self._n_classes = len(self.classes) # menyimpan banyak kelas
        self._n_features = X.shape[1] # menyimpan banyak fitur

        self._is_categorical = np.zeros(self._n_features, dtype=bool)

        if categorical_features is not None:
            self._is_categorical[categorical_features] = True # categorical feature berupa list index

        self._rng = np.random.default_rng(self.random_state)

        self.root = self._build(X, index, depth=0)

        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int): # membangun tree
        counts = np.bincount(y, minlength=self._n_classes) # menghitung jumlah masing masing kelas
        proporsi_tiap_kelas = counts / len(y)

        node = Node(proporsi_tiap_kelas)

        if self.max_depth is not None and depth >= self.max_depth: # pohon mencapai kedalaman maksimal
            return node

        if len(y) < self.min_samples_split: # data sudah terlalu sedikit untuk di split
            return node

        if np.count_nonzero(counts) == 1: # kelas sudah murni, sehingga hanya ada 1 yang tidak 0
            return node

        feature, threshold, child_gini = self._best_split(X, y)

        if feature is None:
            return node # tidak ada kolom yang bisa di split

        if gini(counts) - child_gini <= 1e-12: # cek apakah split memperbaiki keadaan
            return node

        col = X[:, feature]
        is_nan = np.isnan(col)

        
        if self._is_categorical[feature]:
            go_left = col == threshold
        else:
            go_left = col <= threshold

        n_left = go_left.sum()
        n_right = (~go_left & ~is_nan).sum() # masuk ke node kanan untuk yang belum masuk ke kiri dan tidak none

        nan_go_left = n_left >= n_right # nan akan masuk ke node mayoritas
        go_left[is_nan] = nan_go_left # semua nan di update value nya ke nan_go_left

        node.feature = feature
        node.threshold = threshold
        node.nan_go_left = nan_go_left

        node.left = self._build(X[go_left], y[go_left], depth + 1)
        node.right = self._build(X[~go_left], y[~go_left], depth + 1)

        return node

    def _best_threshold(self, x: np.ndarray, y: np.ndarray) -> tuple[float | None, float]: # menentukan threshold
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        left = np.zeros(self._n_classes) # awalnya set 0 sebanyak kelas
        right = np.bincount(y, minlength=self._n_classes) # set dulu sesuai jumlah masing masing kelas

        best_threshold = None
        best_gini = np.inf

        for i in range(len(x_sorted) - 1):
            left[y_sorted[i]] += 1
            right[y_sorted[i]] -= 1

            if x_sorted[i] == x_sorted[i+1]:
                continue

            wg = weighted_gini(left, right)
            if wg < best_gini:
                best_threshold = (x_sorted[i] + x_sorted[i+1]) / 2
                best_gini = wg

        return best_threshold, best_gini

    def _best_category(self, x: np.ndarray, y: np.ndarray) -> tuple[float | None, float]:
        categories = np.unique(x)

        if len(categories) < 2:
            return None, np.inf

        total = np.bincount(y, minlength=self._n_classes) # hitung total tiap feature

        best_category = None
        best_gini = np.inf

        for category in categories:
            left = np.bincount(y[x == category], minlength=self._n_classes)
            right = total - left

            wg = weighted_gini(left, right)

            if wg < best_gini:
                best_category = category
                best_gini = wg

        return best_category, best_gini

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        features = range(self._n_features)

        # pilih feature secara acak sejumlah max feature
        if self.max_features is not None and self.max_features < self._n_features:
            features = self._rng.choice(self._n_features, size=self.max_features, replace=False)

        best_feature = None
        best_threshold = None
        best_gini = np.inf

        for feature in features:
            col = X[:, feature]

            # row tanpa Nan value
            not_nan = ~np.isnan(col)
            x_clean = col[not_nan]
            y_clean = y[not_nan]

            if self._is_categorical[feature]:
                threshold, wg = self._best_category(x_clean, y_clean)
            else:
                threshold, wg = self._best_threshold(x_clean, y_clean)

            if threshold is not None and wg < best_gini:
                best_feature = int(feature)
                best_threshold = threshold
                best_gini = wg

        return best_feature, best_threshold, best_gini

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        prob = np.zeros((len(X), self._n_classes)) # 1 baris per data dan 1 kolom per kelas

        for i, row in enumerate(X):
            node = self.root

            while not node.is_leaf():
                value = row[node.feature]

                if np.isnan(value):
                    go_left = node.nan_go_left
                elif self._is_categorical[node.feature]:
                    go_left = value == node.threshold
                else:
                    go_left = value <= node.threshold

                node = node.left if go_left else node.right

            prob[i] = node.value

        return prob

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_probability(X), axis=1)]
