import numpy as np
import tqdm
from tqdm import tqdm
import tensorflow as tf

from config import PROCESSED_DATA_DIR, MODELS_DIR


class KohonenMaps(object):
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

    def fit_tf(self, X):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        n_samples = tf.shape(X)[0]
        n_features = tf.shape(X)[1]
        self.n_features = int(X.shape[1])

        # Initialisation aléatoire des centroïdes (indices numpy pour compatibilité)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = tf.Variable(tf.gather(X, indices), dtype=tf.float32)

        for _ in tqdm(range(self.max_iter)):
            # Étape 1 : assigner chaque point au centroïde le plus proche
            distances = tf.norm(tf.expand_dims(X, axis=1) - tf.expand_dims(self.centroids, axis=0), axis=2)  # (n, k)
            labels = tf.argmin(distances, axis=1)  # (n,)
            self.labels = labels.numpy()  # pour compatibilité .save()

            # Étape 2 : recalcul des centroïdes
            new_centroids = []
            for i in range(self.n_clusters):
                mask = tf.equal(labels, i)
                cluster_points = tf.boolean_mask(X, mask)
                if tf.shape(cluster_points)[0] > 0:
                    mean = tf.reduce_mean(cluster_points, axis=0)
                else:
                    mean = self.centroids[i]  # si cluster vide → on garde l’ancien
                new_centroids.append(mean)
            new_centroids = tf.stack(new_centroids)

            # Étape 3 : test de convergence
            shift = tf.norm(new_centroids - self.centroids)
            if shift.numpy() < self.tol:
                break

            self.centroids.assign(new_centroids)

    def fit_loop(self, X):
        n_samples, n_features = X.shape
        self.n_features = n_features

        # Initialisation aléatoire des centroïdes
        np.random.seed(self.random_state)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]

        for _ in tqdm(range(self.max_iter)):
            # Étape 1 : assigner les points au centroïde le plus proche
            labels = np.empty(n_samples, dtype=int)

            for i in range(n_samples):
                min_dist = float('inf')
                best_cluster = -1
                for j in range(self.n_clusters):
                    dist = np.linalg.norm(X[i] - self.centroids[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = j
                labels[i] = best_cluster

            self.labels = labels

            # Étape 2 : recalculer les centroïdes (moyenne des points dans chaque cluster)
            new_centroids = np.zeros_like(self.centroids)
            counts = np.zeros(self.n_clusters)

            for i in range(n_samples):
                cluster = labels[i]
                new_centroids[cluster] += X[i]
                counts[cluster] += 1

            for j in range(self.n_clusters):
                if counts[j] > 0:
                    new_centroids[j] /= counts[j]
                else:
                    new_centroids[j] = self.centroids[j]  # éviter NaN si cluster vide

            # Étape 3 : vérification convergence
            shift = np.linalg.norm(new_centroids - self.centroids)
            if shift < self.tol:
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
    kmean = KMeans(50, max_iter=100)
    kmean.fit_loop(X)
    kmean.save(MODELS_DIR / "kmeans" / "weight" / "kmeans_50_mnist.npy")
    print(X.shape)