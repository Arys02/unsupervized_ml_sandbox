from torchvision.datasets import MNIST
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import numpy as np

if __name__ == "__main__":
    dataset = MNIST(root=RAW_DATA_DIR, download=True)
    print(dataset.data.shape)
    image = dataset.data.numpy()
    reshaped_img = image.reshape(60000, 784).astype(np.float32) / 255.0
    np.save(PROCESSED_DATA_DIR / "mnist_flattened.npy", reshaped_img)