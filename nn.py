from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

images, labels = get_mnist()

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer

NN: 784 -> 30 -> 10
"""
n_i = 784
n_h = 30
n_o = 10

w_i_h = np.random.uniform(-1.0, 1.0, (n_h, n_i))
b_i_h = np.zeros((n_h, 1))
w_h_o = np.random.uniform(-1.0, 1.0, (n_o, n_h))
b_h_o = np.zeros((n_o, 1))

learn_rate = 0.02
nr_correct = 0
epochs = 30
for epoch in range(epochs):
    for img, l in zip(images, labels):

        # Changes shape from vector to matrix of dim 1
        img.shape += (1,)
        l.shape += (1,)

        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre)) # Activation function (sigmoid) -> hidden layer

        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre)) # Activation function (sigmoid) -> output layer

        # Cost / Error calculation (Mean Squared)
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0) # SSE
        nr_correct += int(np.argmax(o) == np.argmax(l)) # Check which output neuron is the largest

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"{o.argmax()}")
    plt.show()