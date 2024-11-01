# hodgkin_huxley.py

import numpy as np

class HodgkinHuxleyNeuron:
    def __init__(self, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L):
        self.V = -65.0                     # Membrane potential (mV)
        self.n = 0.317                      # Potassium activation gating variable
        self.m = 0.05                       # Sodium activation gating variable
        self.h = 0.6                        # Sodium inactivation gating variable
        self.C_m = C_m                      # Membrane capacitance (µF/cm²)
        self.g_Na = g_Na                    # Sodium maximum conductance (mS/cm²)
        self.g_K = g_K                      # Potassium maximum conductance (mS/cm²)
        self.g_L = g_L                      # Leak maximum conductance (mS/cm²)
        self.E_Na = E_Na                    # Sodium reversal potential (mV)
        self.E_K = E_K                      # Potassium reversal potential (mV)
        self.E_L = E_L                      # Leak reversal potential (mV)
        self.input_current = 0              # Input current
       

    def alpha_n(self, V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    
    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65) / 80)
    
    def alpha_m(self, V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    
    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65) / 18)
    
    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65) / 20)
    
    def beta_h(self, V):
        return 1 / (1 + np.exp(-(V + 35) / 10))
    
    def step(self, dt):
        # Calculate currents
        I_Na = self.g_Na * (self.m**3) * self.h * (self.V - self.E_Na)
        I_K = self.g_K * (self.n**4) * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)

        # Membrane voltage differential equation
        dVdt = (self.input_current - I_Na - I_K - I_L) / self.C_m
        self.V += dt * dVdt

        # Gating variables differential equations
        dn_dt = self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n
        dm_dt = self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m
        dh_dt = self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h
        self.n += dt * dn_dt
        self.m += dt * dm_dt
        self.h += dt * dh_dt

class HodgkinHuxleyNetwork:
    def __init__(self, size):
        self.size = size                     # Number of neurons
        self.neurons = []                    # List to hold neurons
        self.weights = np.random.uniform(-0.5, 0.5, (size, size))  # Synaptic weight matrix
        self.input_currents = np.zeros(size)  # Input current vector
        self.spike_times = [[] for _ in range(size)]

    def add_neuron(self, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L):
        # Add a new neuron with specified parameters
        C_m = np.random.uniform(0.5, 1.5)     # Membrane capacitance (µF/cm²)
        g_Na = np.random.uniform(80.0, 120.0) # Sodium maximum conductance (mS/cm²)
        g_K = np.random.uniform(20.0, 36.0)   # Potassium maximum conductance (mS/cm²)
        g_L = np.random.uniform(0.1, 0.3)      # Leak maximum conductance (mS/cm²)
        E_Na = np.random.uniform(40.0, 60.0)   # Sodium reversal potential (mV)
        E_K = np.random.uniform(-80.0, -70.0)  # Potassium reversal potential (mV)
        E_L = np.random.uniform(-60.0, -50.0)  # Leak reversal potential (mV)
        neuron = HodgkinHuxleyNeuron(C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)
        self.neurons.append(neuron)

    def update_input_currents(self):
        # Update each neuron's input current with synaptic input
        for i in range(self.size):
            synaptic_input = np.dot(self.weights[i], [n.V >= 30 for n in self.neurons])  # Spiking causes voltage reset
            self.neurons[i].input_current = synaptic_input

    def step(self, dt):
        # Update input currents for all neurons based on network dynamics
        self.update_input_currents()
        
        # Step each neuron
        for i, neuron in enumerate(self.neurons):
            neuron.step(dt)

            # Check for spikes and record spike times
            if neuron.V >= 30:
                neuron.V = -65.0  # Reset voltage after spike
                self.spike_times[i].append(len(self.spike_times[i]))  # Append spike time index

    def get_voltages(self):
        # Get membrane potentials for all neurons
        return np.array([neuron.V for neuron in self.neurons])
    
    def calculate_firing_rate(self, dt, window_size=50):
        firing_rate = np.zeros((len(self.spike_times), self.size))

        for i in range(self.size):
            # Calculate firing rate for neuron i
            spikes = np.array(self.spike_times[i])
            for t in range(len(spikes)):
                window_start = max(0, spikes[t] - window_size // 2)
                window_end = min(len(spikes), spikes[t] + window_size // 2)
                firing_rate[window_start:window_end, i] += 1
        
        # Convert counts to Hz
        # firing_rate /= (window_size * dt / 1000)  # Hz
        return firing_rate
