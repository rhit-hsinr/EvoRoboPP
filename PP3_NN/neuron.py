import numpy as np
import matplotlib.pyplot as plt

def step(x):
    if x <= 0:
        return 0
    else: 
        return 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
    
class Perceptron:

    def __init__(self, inputs, lr, range):
        self.W = np.random.uniform(-range, range, inputs)
        self.bias = np.random.uniform(-range, range)
        self.lr = lr
    
    def forward(self, I):
        return tanh(np.dot(self.W,I) + self.bias)
    
    def train(self, I, target):
        output = self.forward(I)
        error = target - output
        derivative = tanh_derivative(output)
        self.W += self.lr * error * derivative * I
        self.bias += self.lr * error * derivative
        return abs(error)
    
    def set_weights(self, W, bias):
        self.W = W
        self.bias = bias

    def get_weights(self):
        return self.W, self.bias

    def viz(self, granular, title):
        X = np.linspace(-1.05,1.05,granular)
        Y = np.linspace(-1.05,1.05,granular)
        output = np.zeros((granular,granular))
        i = 0
        for x in X:
            j = 0
            for y in Y:
                output[i,j] = self.forward([x,y])
                j += 1
            i += 1

        plt.pcolormesh(X,Y,output)
        plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.show()


class NeuralNet():

    def __init__(self,hiddenUnits,r):
        self.nh = hiddenUnits
        self.H = []
        for i in range(self.nh):
            self.H.append(Perceptron(2,0.1,r))
        self.nO = Perceptron(hiddenUnits,0.1,r)

    def forward(self,Input):
        hiddenOutput = []
        for unit in self.H:
            hiddenOutput.append(unit.forward(Input))
        return self.nO.forward(hiddenOutput)

    def set_weights(self, hidden_weights, output_weights):
        for i in range(self.nh):
            self.H[i].set_weights(hidden_weights[i][0], hidden_weights[i][1])
        self.nO.set_weights(output_weights[0], output_weights[1])

    def get_weights(self):
        hidden_weights = [unit.get_weights() for unit in self.H]
        output_weights = self.nO.get_weights()
        return hidden_weights, output_weights
    
    def train(self,Input,target):
        # 3. Propagate forward
        hiddenOutput = []
        for unit in self.H:
            hiddenOutput.append(unit.forward(Input))
        output = self.nO.forward(hiddenOutput)

        # # 4. Calculate error for the output neuron
        # derivative = output * (1.0 - output)
        # errorOutput = (target - output)*derivative

        # # 5. Calculate error for the hidden units
        # error = []
        # for i in range(self.nh):
        #     derivative = hiddenOutput[i] * (1.0 - hiddenOutput[i])
        #     error.append(self.nO.W[i] * errorOutput * derivative)

        # # Update the hidden-output weights
        # for i in range(self.nh):
        #     self.nO.W[i] += hiddenOutput[i] * errorOutput * 0.1
        # self.nO.bias += 1 * errorOutput * 0.1

        # # Update the input-hidden weights
        # for i in range(self.nh):
        #     self.H[i].W[0] += Input[0] * error[i] * 0.1
        #     self.H[i].W[1] += Input[1] * error[i] * 0.1
        #     self.H[i].bias += 1 * error[i] * 0.1

        # return abs(errorOutput)
        errorOutput = (target - output) * tanh_derivative(output)
        errorHidden = [errorOutput * self.nO.W[i] * tanh_derivative(hiddenOutput[i]) for i in range(self.nh)]

        # Update output weights
        for i in range(self.nh):
            self.nO.W[i] += 0.1 * hiddenOutput[i] * errorOutput
        self.nO.bias += 0.1 * errorOutput

        # Update input-hidden weights
        for i in range(self.nh):
            self.H[i].W += 0.1 * errorHidden[i] * np.array(Input)
            self.H[i].bias += 0.1 * errorHidden[i]

        return abs(errorOutput)
    
    def viz2(self, granular, title, dataset, label):
        X = np.linspace(-1.05,1.05,granular)
        Y = np.linspace(-1.05,1.05,granular)
        output = np.zeros((granular,granular))
        i = 0
        for x in X:
            j = 0
            for y in Y:
                output[i,j] = self.forward([x,y])
                j += 1
            i += 1

        plt.pcolormesh(X,Y,output)
        plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")

        for i in range(len(dataset)):
            if label[i] == 1:
                plt.plot(dataset[i][0],dataset[i][1],'wo')
            else:
                plt.plot(dataset[i][0],dataset[i][1],'wx')

        plt.title(title)
        plt.show()

    