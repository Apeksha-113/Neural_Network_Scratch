import argparse
from model import NeuralNetwork
from utils import prepare_data, plot_predictions
import numpy as np

def main(layers, learning_rate, epochs, verbose_epoch):
    X, y = prepare_data()
    nn = NeuralNetwork(layer_list=layers)
    nn.train(X, y, learning_rate=learning_rate, epochs=epochs, verbose_epoch=verbose_epoch)
    nn.plot_epoch()

    y_pred = np.array([nn.deep_forward(x.reshape(-1, 1))[0][0] for x in X])
    plot_predictions(X, y, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple neural network")
    parser.add_argument("--layers", nargs='+', type=int, default=[1, 10, 10, 1], help="List of layer sizes")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--verbose_epoch", type=int, default=200, help="Verbose output every n epochs")

    args = parser.parse_args()
    main(layers=args.layers, learning_rate=args.learning_rate, epochs=args.epochs, verbose_epoch=args.verbose_epoch)
