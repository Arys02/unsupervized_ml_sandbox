from models.kmeans.kmeans import KMeans
import matplotlib.pyplot as plt
from config import PROCESSED_DATA_DIR, KM_DIR
import numpy as np

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


if __name__ == '__main__':
    # Chargement
    model = KMeans.load(KM_DIR / "weight" / "kmeans.npy")
    X = np.load(PROCESSED_DATA_DIR / 'mnist_flattened.npy')

    show_centroids_grid(model.centroids)


    img = X[1].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title("Image 0")
    plt.axis('off')
    plt.show()

    encoded = encode(model, X[0])
    print(f"encoded idx {encoded}")

    decoded = decode(model, encoded)

    img = decoded.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title("Image 0")
    plt.axis('off')
    plt.show()