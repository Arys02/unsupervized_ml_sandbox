from keras.datasets import mnist
from config import PROCESSED_DATA_DIR
import numpy as np

mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = x_train.reshape(60000, 784).astype(np.float32) / 255.0
np.save(PROCESSED_DATA_DIR / "mnist_flattened.npy", x)