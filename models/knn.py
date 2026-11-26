import numpy as np

# distance
def manhattan_distance(X_test, X_train) -> np.ndarray:
    return np.array([np.abs(q - X_train).sum(axis=1) for q in X_test])

def euclidean_distance(X_test, X_train) -> np.ndarray:
    return np.array([np.sqrt(((q - X_train) ** 2).sum(axis=1)) for q in X_test])

DISTANCE = {"manhattan": manhattan_distance, "euclidean": euclidean_distance}
NORMALISASI = (None, "minmax", "zscore")

class KNN:
    def __init__(self, k: int = 3,  distance_metric: str = "euclidean", normalisasi: str = None):
        if k < 1:
            raise ValueError("k harus >= 1")
        if distance_metric not in DISTANCE:
            raise ValueError(f"matriks tidak ada: {distance_metric}")
        if normalisasi not in NORMALISASI:
            raise ValueError(f"normalisasi tidak ada: {normalisasi}")
        self.k = k
        self.distance_metric = distance_metric
        self.normalisasi = normalisasi
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi (n sampel, n feature)")
        if y.ndim != 1 or len(X) != len(y):
            raise ValueError("y harus 1D dan jumlah barisnya sama dengan X")
        if self.k > len(X):
            raise ValueError(f"k ({self.k}) tidak boleh lebih besar dari jumlah data train ({len(X)})")

        self._fit_normalisasi(X)
        self.X_train = self._normalisasi_data(X)
        # simpan y sebagai index kelas, urutan kelas sama dengan model lain (np.unique)
        self.classes, self.y_train = np.unique(y, return_inverse=True)
        return self

    def _fit_normalisasi(self, X):
        if self.normalisasi == "minmax":
            self.center = np.nanmin(X, axis=0)
            self.scale = np.nanmax(X, axis=0) - self.center
        elif self.normalisasi == "zscore":
            self.center = np.nanmean(X, axis=0)
            self.scale = np.nanstd(X, axis=0)
        else:
            self.center = np.zeros(X.shape[1])
            self.scale = np.ones(X.shape[1])
        self.scale[self.scale == 0] = 1  # agar tidak 0

        # NaN diisi rata-rata kolom dari data train 
        self.isi_nan = np.nan_to_num(np.nanmean(X, axis=0))

    def _normalisasi_data(self, X):
        X = np.where(np.isnan(X), self.isi_nan, X)  # isi NaN dulu
        return (X - self.center) / self.scale

    def _compute_distances(self, X_test) -> np.ndarray:
        return DISTANCE[self.distance_metric](X_test, self.X_train)

    def _tetangga(self, X_test) -> np.ndarray:
        # return index kelas dari k tetangga, urut dari yang paling dekat
        if self.X_train is None:
            raise RuntimeError("model belum pernah di fit")

        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim != 2 or X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"X harus 2 dimensi dengan {self.X_train.shape[1]} kolom")

        X_test = self._normalisasi_data(X_test)
        D = self._compute_distances(X_test)
        idx = np.argsort(D, axis=1, kind="stable")[:, :self.k]
        return self.y_train[idx]

    def _hitung_vote(self, label_tetangga) -> np.ndarray:
        # jumlah vote tiap kelas
        return (label_tetangga[:, :, None] == np.arange(len(self.classes))).sum(axis=1)

    def predict(self, X_test) -> np.ndarray:
        label_tetangga = self._tetangga(X_test)
        vote = self._hitung_vote(label_tetangga)

        hasil = []
        for label, jumlah in zip(label_tetangga, vote):
            seri = jumlah == jumlah.max()
            # kalau vote seri, pilih kelas yang tetangganya paling dekat
            hasil.append(label[seri[label]][0])
        return self.classes[np.array(hasil)]

    def predict_probability(self, X_test) -> np.ndarray:
        vote = self._hitung_vote(self._tetangga(X_test))
        return vote / self.k
