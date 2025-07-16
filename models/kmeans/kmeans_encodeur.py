from models.kmeans.kmeans import KMeans
import matplotlib.pyplot as plt
from config import PROCESSED_DATA_DIR, KM_DIR
import numpy as np
from sklearn.decomposition import PCA

def encode(kmeans: KMeans, x):
    k = kmeans.predict(x)
    return k

def decode(kmeans: KMeans, k):
    return kmeans.centroids[k]


def show_centroids_grid(centroids, image_shape=(28, 28), title="Centroid Images"):
    k = len(centroids)
    n_cols = int(np.ceil(np.sqrt(k)))      # largeur de la grille
    n_rows = int(np.ceil(k / n_cols))      # hauteur de la grille

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    axes = axes.flatten()

    for i, centroid in enumerate(centroids):
        img = centroid.reshape(image_shape)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Cluster {i}")
        axes[i].axis('off')

    # Masquer les cases vides
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_original_vs_decoded(X, model, encode_fn, decode_fn, n=10, image_shape=(28, 28)):
    indices = np.random.choice(len(X), size=n, replace=False)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(4, n * 2))

    for i, idx in enumerate(indices):
        original = X[idx].reshape(image_shape)
        encoded = encode_fn(model, X[idx])
        reconstructed = decode_fn(model, encoded).reshape(image_shape)

        axes[i, 0].imshow(original, cmap='gray')
        axes[i, 0].set_title(f"Original #{idx}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed, cmap='gray')
        axes[i, 1].set_title(f"Decoded #{idx}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Chargement
    model = KMeans.load(KM_DIR / "weight" / "kmeans_50_mnist.npy")
    X = np.load(PROCESSED_DATA_DIR / 'mnist_flattened.npy')

    # show_centroids_grid(model.centroids)
    # show_original_vs_decoded(X, model, encode, decode, n=10)

    #X.shape -> (60000, 784)
    #Z_2d = PCA(n_components=2).fit_transform(Z)

    #plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=model.labels, cmap='tab10', s=5)
    #plt.title("Projection 2D de l'espace latent")
    #plt.show()


    # img = X[1].reshape(28, 28)
    # plt.imshow(img, cmap='gray')
    # plt.title("Image 0")
    # plt.axis('off')
    # plt.show()
    #
    # encoded = encode(model, X[1])
    # print(f"encoded idx {encoded}")
    #
    # decoded = decode(model, encoded)
    #
    # img = decoded.reshape(28, 28)
    # plt.imshow(img, cmap='gray')
    # plt.title("Image 0")
    # plt.axis('off')
    # plt.show()