import numpy as np
import pathlib

def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as data:
        images, labels = data['x_train'], data['y_train']
    images = images.astype('float32') / 255.0
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels