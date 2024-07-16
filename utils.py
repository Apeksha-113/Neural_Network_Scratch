import numpy as np
import matplotlib.pyplot as plt

def prepare_data(num_samples=500):
    X = np.linspace(-2 * np.pi, 2 * np.pi, num_samples).reshape(num_samples, 1, 1)
    y = 0.5 * (np.sin(X) + 1)
    X_norm = (X - X.mean()) / X.std()
    return X_norm, y

def plot_predictions(X, y, y_pred):
    plt.figure(figsize=(10, 4))
    plt.plot(y[:, 0], label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle='dashed', color='r')
    plt.title('Actual vs Predicted')
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.legend()
    plt.grid(True)
    plt.show()
