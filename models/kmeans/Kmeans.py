import numpy as np

class KMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
    

    def _clusters_centers_(self):
        """
        Retourne les centroïdes appris (shape = (n_clusters, n_features)).
        """
        return self.centroids
    
    def _predict(self, X):
        """
        Pour chaque point de X, calcule la distance à chacun des centroïdes
        et renvoie l'indice du centroïde le plus proche.
        """
        # calcule les distances (shape = (n_samples, n_clusters))
        distances = self._compute_distances(X, self.centroids)
        # prend l'indice du centroïde le plus proche sur l'axe 1
        return np.argmin(distances, axis=1)

    def _fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Choisir k points aléatoires comme centroïdes initiaux
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[indices]

        for i in range(self.max_iter):
            # Calculer les distances de tous les points aux centroïdes
            distances = self._compute_distances(X, centroids)

            # Trouver le cluster le plus proche pour chaque point
            labels = np.argmin(distances, axis=1)

            #recalculer la position des centroïdes
            # en prenant la moyenne des points assignés à chaque cluster
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(self.n_clusters)
            ])

            diff = np.linalg.norm(new_centroids - centroids)
            if diff < self.tol:
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels_ = labels

    def _compute_distances(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances
    

