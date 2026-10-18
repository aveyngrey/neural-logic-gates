import math
import random

inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
targets = (0, 1, 1, 0)

weights_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
weights_output = [random.uniform(-1, 1) for _ in range(2)]

biases_hidden = [0.1, 0.1]
bias_output = 0.1

def train(epochs: int = 1000, learning_rate: float = 0.1) -> None:
    global weights_hidden, weights_output, biases_hidden, bias_output
    for epoch in range(1, epochs + 1):
        dw_hidden = [[0, 0], # neuron 1
                     [0, 0]] # neuron 2
        db_hidden = [0, 0] # nueron 1, neuron 2

        dw_output = [0, 0]
        db_output = 0

        for i, (x, target) in enumerate(zip(inputs, targets)):
            z_hidden = [] # row corresponds to input, col corresponds to neuron
            for input in inputs:
                z = []
                z.append(weights_hidden[0][0] * input[0] +
                            weights_hidden[0][1] * input[1] +
                            biases_hidden[0])
                z.append(weights_hidden[1][0] * input[0] +
                            weights_hidden[1][1] * input[1] +
                            biases_hidden[1])
                z_hidden.append(z)
            a_hidden = [[sigmoid(x) for x in z] for z in z_hidden]
            z_output = [weights_output[0] * a_h_row[0] + 
                        weights_output[1] * a_h_row[1] +
                        bias_output
                        for a_h_row in a_hidden]
            a_output = [sigmoid(z) for z in z_output]
        
            prediction = a_output[i]
            d_error = (prediction - target) * prediction * (1 - prediction)
            
            dw_output = [dw + (d_error * h_a) for dw, h_a in zip(dw_output, a_hidden[i])]

            db_output += d_error

            dw_hidden = [[dw + (d_error * weights_output[k] * (a_hidden[i][k] * (1 - a_hidden[i][k])) * x[j])
                         for j, dw in enumerate(weights)] 
                         for k, weights in enumerate(dw_hidden)]

            db_hidden = [db + (d_error * weights_output[j] * (a_hidden[i][j] * (1 - a_hidden[i][j])))
                         for j, db in enumerate(db_hidden)]

        weights_output = [weight_o - (learning_rate * dw_o)
                          for weight_o, dw_o in zip(weights_output, dw_output)]
        bias_output -= learning_rate * db_output
        weights_hidden = [[weight_h - (learning_rate * dw_h)
                           for weight_h, dw_h in zip(neuron, dw_h_row)] 
                           for neuron, dw_h_row in zip(weights_hidden, dw_hidden)]
        biases_hidden = [bias_h - (learning_rate * db_h)
                         for bias_h, db_h in zip(biases_hidden, db_hidden)]
        

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def predict() -> list[int]:
    predictions = []
    for input in inputs:
        hidden_neuron1_z = input[0] * weights_hidden[0][0] + \
                           input[1] * weights_hidden[0][1] + \
                           biases_hidden[0]

        hidden_neuron2_z = input[0] * weights_hidden[1][0] + \
                           input[1] * weights_hidden[1][1] + \
                           biases_hidden[1]

        hidden_neuron1_a = sigmoid(hidden_neuron1_z)
        hidden_neuron2_a = sigmoid(hidden_neuron2_z)

        output_neuron_z = hidden_neuron1_a * weights_output[0] + \
                          hidden_neuron2_a * weights_output[1] + \
                          bias_output

        output_neuron_a = sigmoid(output_neuron_z)
        prediction = 1 if output_neuron_a >= 0.5 else 0
        predictions.append(prediction)
                         
    return predictions

def main() -> None:
    print("Before training: \n" \
         f"Hidden Weights: {weights_hidden} \n" \
         f"Output Weights: {weights_output} \n" \
          "--------------------------------------------------------")
    train(epochs=5000, learning_rate=.5)
    print("After training: \n" \
         f"Hidden Weights: {weights_hidden} \n" \
         f"Hidden Biases: {biases_hidden} \n\n" \
         f"Output Weights: {weights_output} \n" \
         f"Output Bias: {bias_output}       \n" \
         "--------------------------------------------------------")
    print(predict())

if __name__ == "__main__":
    main()