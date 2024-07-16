import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_list=[1, 3, 1]) -> None:
        self.layer_list = layer_list
        self.parameters = self.initialize_weights_deep(self.layer_list)
        self.cache = []
        self.epoch_loss = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def initialize_weights_deep(self, layer_list):
        parameters = {}
        no_layers = len(layer_list)

        for l in range(1, no_layers):
            parameters["W" + str(l)] = np.random.randn(layer_list[l], layer_list[l-1]) * 0.1
            parameters["b" + str(l)] = np.random.randn(layer_list[l], 1) * 0.1

        return parameters

    def deep_forward(self, X):
        self.cache = []
        no_weights = len(self.parameters) // 2
        A_temp = X
        for l in range(1, no_weights):
            z = np.dot(self.parameters["W" + str(l)], A_temp) + self.parameters["b" + str(l)]
            a = self.relu(z)
            temp_cache = (z, self.parameters["W" + str(l)], self.parameters["b" + str(l)], A_temp)
            self.cache.append(temp_cache)
            A_temp = a

        Z = np.dot(self.parameters["W" + str(no_weights)], A_temp) + self.parameters["b" + str(no_weights)]
        A = self.sigmoid(Z)
        temp_cache = (Z, self.parameters["W" + str(no_weights)], self.parameters["b" + str(no_weights)], A_temp)
        self.cache.append(temp_cache)
        return A

    def backward_pass(self, Y_target, AL):
        grads = {}
        no_layers = len(self.cache)
        Y_target = Y_target.reshape(AL.shape)
        dAL = -(Y_target - AL)

        Z_cache, W_cache, b_cache, A_prev_cache = self.cache[-1]
        dZ = dAL * self.sigmoid_derivative(Z_cache)
        dW_temp = np.dot(dZ, A_prev_cache.T)
        db_temp = dZ
        dA_prev_temp = np.dot(W_cache.T, dZ)

        grads["dA" + str(no_layers - 1)] = dA_prev_temp
        grads["dW" + str(no_layers)] = dW_temp
        grads["db" + str(no_layers)] = db_temp

        for l in reversed(range(no_layers - 1)):
            Z_cache, W_cache, b_cache, A_prev_cache = self.cache[l]
            dZ = np.array(dA_prev_temp, copy=True)
            dZ[Z_cache <= 0] = 0
            dW_temp = np.dot(dZ, A_prev_cache.T)
            db_temp = dZ
            dA_prev_temp = np.dot(W_cache.T, dZ)

            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    def train(self, X, y, learning_rate, epochs=10, verbose_epoch=10, verbose=True):
        self.epoch_loss = []
        dataset_length = len(X)
        for epoch in range(epochs):
            cost = 0
            for iter in range(dataset_length):
                out = self.deep_forward(X[iter])
                cost += self.cost_function(Y=y[iter], AL=out)
                grad = self.backward_pass(Y_target=y[iter], AL=out)
                self.update_parameters(grads=grad, learning_rate=learning_rate)
            cost = cost / dataset_length
            if epoch % verbose_epoch == 0 and verbose:
                print(f"Epoch {epoch} Loss: {cost[0][0]}")
            self.epoch_loss.append(cost[0][0])

    def cost_function(self, Y, AL):
        error = Y - AL
        cost = (1/2) * (error)**2
        return cost

    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    def plot_epoch(self):
        plt.figure(figsize=(5, 3))
        plt.plot(self.epoch_loss, linestyle='-', color='b', label='Epoch Loss')
        plt.title('Epoch Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def clear_weights(self):
        self.parameters = self.initialize_weights_deep(self.layer_list)
