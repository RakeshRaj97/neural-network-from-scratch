import numpy as np


class OurNeuralNetwork:
    def __init__(self):
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()

        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv_sigmoid(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def feedforward(self, x):
        # x is a numpy array with 3 elements
        h1 = self.sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        h2 = self.sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
        h3 = self.sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
        o1 = self.sigmoid(self.w10 * x[0] + self.w11 * x[1] + self.w12 * x[2] + self.b4)
        return o1

    def train(self, data, all_y_true):
        learn_rate = 0.1
        epochs = 10000

        for i in range(epochs):
            for x, y_true in zip(data, all_y_true):
                # feedforward
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
                h1 = self.sigmoid(sum_h1)

                sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
                h2 = self.sigmoid(sum_h2)

                sum_h3 = self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b2
                h3 = self.sigmoid(sum_h3)

                sum_o1 = self.w10 * x[0] + self.w11 * x[1] + self.w12 * x[2] + self.b2
                o1 = self.sigmoid(sum_o1)

                y_pred = o1

                # partial_derivatives
                dl_dypred = -2 * (y_true - y_pred)

                # Neuron o1
                dypred_dw10 = h1 * self.deriv_sigmoid(sum_o1)
                dypred_dw11 = h2 * self.deriv_sigmoid(sum_o1)
                dypred_dw12 = h3 * self.deriv_sigmoid(sum_o1)
                dypred_db4 = self.deriv_sigmoid(sum_o1)

                dypred_dh1 = self.w10 * self.deriv_sigmoid(sum_o1)
                dypred_dh2 = self.w11 * self.deriv_sigmoid(sum_o1)
                dypred_dh3 = self.w12 * self.deriv_sigmoid(sum_o1)

                # Neuron h1
                dh1_dw1 = x[0] * self.deriv_sigmoid(sum_h1)
                dh1_dw2 = x[1] * self.deriv_sigmoid(sum_h1)
                dh1_dw3 = x[2] * self.deriv_sigmoid(sum_h1)
                dh1_b1 = self.deriv_sigmoid(sum_h1)

                # Neuron h2
                dh2_dw4 = x[0] * self.deriv_sigmoid(sum_h2)
                dh2_dw5 = x[1] * self.deriv_sigmoid(sum_h2)
                dh2_dw6 = x[2] * self.deriv_sigmoid(sum_h2)
                dh2_b2 = self.deriv_sigmoid(sum_h2)

                # Neuron h3
                dh1_dw7 = x[0] * self.deriv_sigmoid(sum_h3)
                dh1_dw8 = x[1] * self.deriv_sigmoid(sum_h3)
                dh1_dw9 = x[2] * self.deriv_sigmoid(sum_h3)
                dh3_b3 = self.deriv_sigmoid(sum_h3)

                # gradient_descent Updating weights and biases
                # Neuron1
                self.w1 -= learn_rate * dl_dypred * dypred_dh1 * dh1_dw1
                self.w2 -= learn_rate * dl_dypred * dypred_dh1 * dh1_dw2
                self.w3 -= learn_rate * dl_dypred * dypred_dh1 * dh1_dw3
                self.b1 -= learn_rate * dl_dypred * dypred_dh1 * dh1_b1

                # Neuron2
                self.w4 -= learn_rate * dl_dypred * dypred_dh2 * dh2_dw4
                self.w5 -= learn_rate * dl_dypred * dypred_dh2 * dh2_dw5
                self.w6 -= learn_rate * dl_dypred * dypred_dh2 * dh2_dw6
                self.b2 -= learn_rate * dl_dypred * dypred_dh2 * dh2_b2

                # Neuron3
                self.w7 -= learn_rate * dl_dypred * dypred_dh3 * dh1_dw7
                self.w8 -= learn_rate * dl_dypred * dypred_dh3 * dh1_dw8
                self.w9 -= learn_rate * dl_dypred * dypred_dh3 * dh1_dw9
                self.b3 -= learn_rate * dl_dypred * dypred_dh3 * dh3_b3

                # Neurono1
                self.w10 -= learn_rate * dl_dypred * dypred_dw10
                self.w11 -= learn_rate * dl_dypred * dypred_dw11
                self.w12 -= learn_rate * dl_dypred * dypred_dw12
                self.b4 -= learn_rate * dl_dypred * dypred_db4

                if i % 1000 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = self.mse_loss(all_y_true, y_preds)
                    print("Epoch %d loss: %.3f" % (i, loss))


# define_dataset
data = np.array([[1, 0, 1],
                 [1, 1, 1],
                 [1, 0, 1],
                 [0, 0, 1],
                 [0, 0, 0],
                 [0, 1, 0]])
all_y_trues = np.array([1, 1, 1, 0, 0, 0])

network = OurNeuralNetwork()
network.train(data, all_y_trues)

# predictions
f = np.array([1, 0, 0])
print("Prediction for f: %.3f" % network.feedforward(f))
input1 = int(input("Input 1:"))
input2 = int(input("Input 2:"))
input3 = int(input("Input 3:"))
e = np.array([input1, input2, input3])
print("Input values:", input1, input2, input3)

print(network.feedforward(e))
