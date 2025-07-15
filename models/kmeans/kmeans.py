import numpy as np
import tqdm
from tqdm import tqdm

from config import PROCESSED_DATA_DIR, MODELS_DIR


class KMeans(object):
    def __init__(self, n_clusters=2, random_state=0, max_iter=100, tol=0.0001):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None #will be ndarray
        self.labels = None
        self.n_features = None


    def fit(self, X):
        print(X.shape)
        n_samples, n_features = X.shape

        # aléatoire des centroid pour commencer
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]

        for _ in tqdm(range(self.max_iter)):
            #calcule des distances
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])
            shift = np.linalg.norm(new_centroids - self.centroids)
            if shift < self.tol :
                break

            self.centroids = new_centroids


    def get_centroids(self):
        return self.centroids

    def predict(self, x):
        if self.centroids is None:
            raise ValueError("Kmeans has not been trained")

        x = np.array(x)

        if x.ndim != 1 or x.shape[0] != self.centroids.shape[1]:
            raise ValueError(f"x doit avoir shape ({self.centroids.shape[1]},), reçu {x.shape}")

            # Calcul des distances entre x et chaque centroïde
        distances = np.linalg.norm(self.centroids - x, axis=1)
        return np.argmin(distances)

    def save(self, filename):
        data = {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'centroids': self.centroids,
            'labels': self.labels,
            'n_features': self.n_features
        }
        np.save(filename, data, allow_pickle=True)


    @classmethod
    def load(cls, filename):
        data = np.load(filename, allow_pickle=True).item()

        model = cls(n_clusters=data['n_clusters'],
                    random_state=data['random_state'],
                    max_iter=data['max_iter'],
                    tol=data['tol'])

        model.centroids = data['centroids']
        model.labels = data['labels']
        model.n_features = data['n_features']

        return model




if __name__ == '__main__':
    X = np.load(PROCESSED_DATA_DIR / "mnist_flattened.npy")
    kmean = KMeans(20, max_iter=100)
    kmean.fit(X)
    kmean.save(MODELS_DIR / "kmeans" / "weight" / "kmeans_200_mnist.npy")
    print(X.shape)