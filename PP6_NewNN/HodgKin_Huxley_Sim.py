# network_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from Hodgkin_Huxley import HodgkinHuxleyNetwork

# Simulation parameters
network_size = 10        # Number of neurons
time = 100               # Simulation time in ms
dt = 0.01                # Time step in ms
external_input = 10      # External current in µA/cm²
num_runs = 5             # Number of simulation runs

# Hodgkin-Huxley parameters
C_m = 1.0                # Membrane capacitance (µF/cm²)
g_Na = 120.0             # Sodium maximum conductance (mS/cm²)
g_K = 36.0               # Potassium maximum conductance (mS/cm²)
g_L = 0.3                # Leak maximum conductance (mS/cm²)
E_Na = 50.0              # Sodium reversal potential (mV)
E_K = -77.0              # Potassium reversal potential (mV)
E_L = -54.387            # Leak reversal potential (mV)

# Initialize storage for all runs
all_voltages = []

# Run multiple simulations
for run in range(num_runs):
    # Initialize network
    network = HodgkinHuxleyNetwork(size=network_size)

    # Add neurons with random parameters
    for _ in range(network_size):
        network.add_neuron(C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)

    # Record voltages for this run
    voltages = []

    # Run simulation
    for _ in range(int(time / dt)):
        network.step(dt)
        voltages.append(network.get_voltages())

    # Store the voltages of this run
    all_voltages.append(voltages)

# Convert all_voltages to a numpy array for easier plotting
all_voltages = np.array(all_voltages)

# Plot the results for all runs

# plt.figure(figsize=(12, 8))
# for run in range(num_runs):
#     for i in range(network_size):
#         plt.plot(np.arange(0, time, dt), all_voltages[run][:, i], label=f"Run {run + 1}, Neuron {i + 1}", alpha=0.5)

# Plot all runs 1 neuron

# plt.figure(figsize=(12, 8))
# for run in range(num_runs):
#     # Plot only the first neuron of each run
#     plt.plot(np.arange(0, time, dt), all_voltages[run][:, 0], label=f"Run {run + 1}, Neuron 1", alpha=0.5)

# Plot one run all neurons

# plt.figure(figsize=(12, 8))
# for i in range(network_size):
#     plt.plot(np.arange(0, time, dt), all_voltages[0][:, i], label=f"Neuron {i + 1}", alpha=0.5)

# plt.xlabel("Time (ms)")
# plt.ylabel("Membrane Potential (mV)")
# plt.title("Hodgkin-Huxley Network Simulation (Single Runs)")
# plt.legend(loc='upper right')
# plt.show()

# Plot firing rate
firing_rates = network.calculate_firing_rate(dt)

# Plot firing rates

# plt.figure(figsize=(12, 8))
# for i in range(network_size):
#     plt.plot(np.arange(0, len(firing_rates), 1), firing_rates[:, i], label=f"Neuron {i + 1}", alpha=0.5)

# plt.xlabel("Time (time steps)")
# plt.ylabel("Firing Rate")
# plt.title("Firing Rates of Hodgkin-Huxley Neurons")
# plt.legend(loc='upper right')
# plt.show()
