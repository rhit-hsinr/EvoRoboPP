import numpy as np

class PredatorNN:
    def __init__(self, units_per_layer, activations=None):
        self.units_per_layer = units_per_layer
        self.num_layers = len(units_per_layer)

        # self.activation = lambda x: 1 / (1 + np.exp(-x))
        self.activation = lambda x: np.tanh(x)

        # Initialize weight and bias range scaling
        self.weightrange = 0.5
        self.biasrange = 0.5

        # Initialize weights and biases to None, they will be set using setParams()
        self.weights = []
        self.biases = []

    def setParams(self, params):
        """ Set the weights and biases of the neural network """
        self.weights = []
        start = 0
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l]*self.units_per_layer[l+1]
            self.weights.append((params[start:end]*self.weightrange).reshape(self.units_per_layer[l],self.units_per_layer[l+1]))
            start = end
        self.biases = []
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l+1]
            self.biases.append((params[start:end]*self.biasrange).reshape(1,self.units_per_layer[l+1]))
            start = end

    def get_num_params(self):
        """ Calculate the total number of parameters (weights + biases) in the network """
        num_params = 0
        for l in range(self.num_layers - 1):
            num_params += self.units_per_layer[l] * self.units_per_layer[l + 1]  # weights
            num_params += self.units_per_layer[l + 1]  # biases
        return num_params

    def forward(self, inputs):
        """ Forward propagate the given inputs through the network """
        states = inputs
        for l in np.arange(self.num_layers - 1):
            if states.ndim == 1:
                states = [states]
            states = self.activation(np.matmul(states, self.weights[l]) + self.biases[l])
        return states