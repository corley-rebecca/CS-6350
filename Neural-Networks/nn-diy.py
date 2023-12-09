import random
import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, no_hiddenlayers=2, hl_units=(2, 2)):
        self.hiddenlayers = no_hiddenlayers
        self.hlunits = hl_units
        self.neuralnet = []
        self.final_output = None
        self.loss = []

    def init_nn(self, no_inputs):
        for i in range(self.hiddenlayers):
            hl = [{'weights': np.array([random.gauss(mu=0.0, sigma=1.0) for i in range(no_inputs + 1)])} for j in
                  range(self.hlunits[i])]
            no_inputs = self.hlunits[i]
            self.neuralnet.append(hl)
        output_layer = [{'weights': np.array(
            [random.gauss(mu=0.0, sigma=1.0) for i in range(self.hlunits[self.hiddenlayers - 1] + 1)])}]
        self.neuralnet.append(output_layer)

    def init_nn_zero(self, no_inputs):
        for i in range(self.hiddenlayers):
            hl = [{'weights': np.array([0.0 for i in range(no_inputs + 1)])} for j in range(self.hlunits[i])]
            no_inputs = self.hlunits[i]
            self.neuralnet.append(hl)
        output_layer = [{'weights': np.array([0.0 for i in range(self.hlunits[self.hiddenlayers - 1] + 1)])}]
        self.neuralnet.append(output_layer)

    def neuronoutput(self, weights, input, layer):
        sum = weights[-1]
        for i in range(len(weights) - 1):
            sum += weights[i] * input[i]
        if layer == len(self.neuralnet) - 1:
            return sum
        return 1 / (1 + np.exp(-sum))

    def forward_propogate(self, input_val):
        for layer in range(len(self.neuralnet)):
            neuron_output = []
            for neuron in self.neuralnet[layer]:
                neuron['output'] = self.neuronoutput(neuron["weights"], input_val, layer)
                neuron['input'] = np.array(input_val)
                neuron_output.append(neuron['output'])
            input_val = neuron_output
        self.final_output = input_val
        return input_val

    def sigmoid_derivative(self, value):
        return value * (1 - value)

    def back_propogate(self, actual_value, lr):
        for i in reversed(range(len(self.neuralnet))):
            layer = self.neuralnet[i]
            if i == len(self.neuralnet) - 1: # Output Layer
                for j in range(len(layer)):
                    neuron = layer[j]
                    error = neuron['output'] - actual_value
                    neuron['dl_dz'] = error
                    neuron['input'] = np.append(neuron['input'], [1])
                    neuron['weights'] -= lr * neuron['dl_dz'] * neuron['input']
            else:
                for j in range(len(layer)):
                    dl_dz = np.zeros(shape=(1,), dtype=float)
                    for neuron in self.neuralnet[i + 1]:
                        dl_dz[0] += neuron['weights'][j] * neuron['dl_dz']
                    layer[j]['dl_dz'] = dl_dz[0]
                    layer[j]['input'] = np.append(layer[j]['input'], [1])
                    layer[j]['weights'] -= lr * layer[j]['dl_dz'] * self.sigmoid_derivative(layer[j]['output']) * \
                                           layer[j]['input']

    def sgd(self, X, y, lr=None, epochs=20):
        for i in range(epochs):
            indexes = random.sample(range(len(X)), len(X))
            loss = 0
            for j in indexes:
                row = X[j]
                self.forward_propogate(row)
                expected = y.iloc[j]
                self.back_propogate(expected, lr(i))

    def predict(self, X_test):
        X = np.array(X_test)
        output = lambda data: self.forward_propogate(data)
        predicted = np.array([output(data)[0] for data in X])
        modified_output = predicted.copy()
        modified_output[predicted > 0.5] = 1
        modified_output[predicted < 0.5] = 0
        return modified_output

    def calc_error(self, actual, predicted):
        return 1 - (np.sum(actual == predicted) / len(actual))


if __name__ == "__main__":
    X_train = pd.read_csv('bank-note/train.csv', header=None)
    X_test = pd.read_csv('bank-note/test.csv', header=None)
    y = X_train.iloc[:, 4]
    X_train = X_train.iloc[:, :4]
    y_test = X_test.iloc[:, 4]
    X_test = X_test.iloc[:, :4]
    widths = [5, 10, 25, 50, 100]
    y0 = 0.01
    for width in widths:
        print("Width: ", width)
        learning_rate = lambda i: y0 / (1 + (y0 * i) / width)
        nn = NeuralNetwork(no_hiddenlayers=2, hl_units=(width, width))
        nn.init_nn(X_train.shape[1])
        nn.sgd(np.array(X_train), y, learning_rate)
        y_pred_train = nn.predict(X_train)
        y_pred_test = nn.predict(X_test)
        print("Training Error: ", nn.calc_error(y, y_pred_train))
        print("Test Error: ", nn.calc_error(y_test, y_pred_test))

    print("Initializing all weights to zero")
    for width in widths:
        print("Width: ", width)
        learning_rate = lambda i: y0 / (1 + (y0 * i) / width)
        nn = NeuralNetwork(no_hiddenlayers=2, hl_units=(width, width))
        nn.init_nn_zero(X_train.shape[1])
        nn.sgd(np.array(X_train), y, learning_rate)
        y_pred_train = nn.predict(X_train)
        y_pred_test = nn.predict(X_test)
        print("Training Error: ", nn.calc_error(y, y_pred_train))
        print("Test Error: ", nn.calc_error(y_test, y_pred_test))