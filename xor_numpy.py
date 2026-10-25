import numpy as np
import numpy.typing as npt
import math

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

weights_hidden = np.random.uniform(-math.sqrt(1/2), math.sqrt(1/2), (2, 2))
weights_output = np.random.uniform(-math.sqrt(1/2), math.sqrt(1/2), (1, 2))

biases_hidden = np.array([[0.1], [0.1]])
bias_output = np.array([[0.1]])

def train(epochs: int = 1000, learning_rate: float = 0.5) -> None:
    global weights_hidden, biases_hidden, weights_output, bias_output
    for epoch in range(epochs):
        dw_hidden = np.zeros((2, 2))
        db_hidden = np.zeros((2, 1))

        dw_output = np.zeros((1, 2))
        db_output = 0

        for i, (x, target) in enumerate(zip(inputs, targets)):
            x = x.reshape(1, 2)
            
            z_hidden = x @ weights_hidden.T + biases_hidden.T
            a_hidden = sigmoid(z_hidden)
                
            z_output = a_hidden @ weights_output.T + bias_output
            a_output = sigmoid(z_output)

            prediction = a_output
            d_error = (prediction - target) * (prediction * (1 - prediction))
            
            dw_hidden += (d_error * weights_output * a_hidden * (1 - a_hidden)).T @ x
            db_hidden += (d_error * weights_output * a_hidden * (1 - a_hidden)).T

            dw_output += d_error * a_hidden
            db_output += d_error

        weights_hidden -= learning_rate * dw_hidden
        biases_hidden -= learning_rate * db_hidden

        weights_output -= learning_rate * dw_output
        bias_output -= learning_rate * db_output
            

def sigmoid(x: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-x))

def predict() -> list[int]:
    predictions = sigmoid(sigmoid(inputs @ weights_hidden.T + biases_hidden.T) \
                  @ weights_output.T + bias_output)
    predictions = predictions.tolist()
    print(predictions)
    predictions = [1 if prediction[0] >= 0.5 else 0 for prediction in predictions]

    print(predictions)
    return predictions

def main() -> None:
    train(epochs=10000, learning_rate=0.2)
    predict()

if __name__ == "__main__":
    main()