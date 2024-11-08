# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:29:34 2024

@author: satvi
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
n = 50  # Number of nodes
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate
Tmax = 100  # Maximum simulation time

# Initialize the network
G = nx.erdos_renyi_graph(n, 0.1)
states = {node: 'S' for node in G.nodes()}  # All nodes start as susceptible
initial_infected = np.random.choice(G.nodes())
states[initial_infected] = 'I'  # Random initial infected node

# Initialize the event lists
event_times = []
num_infected = []

# Function to calculate infection and recovery rates
def calculate_rates(G, states, beta, gamma):
    infection_rates = {}
    recovery_rates = {}
    for node in G.nodes():
        if states[node] == 'S':
            # Infection rate: beta * number of infected neighbors
            infected_neighbors = sum(1 for neighbor in G.neighbors(
                node) if states[neighbor] == 'I')
            infection_rates[node] = beta * infected_neighbors
        else:
            # Recovery rate: gamma
            recovery_rates[node] = gamma
    return infection_rates, recovery_rates


# Initial rates
infection_rates, recovery_rates = calculate_rates(G, states, beta, gamma)

# Simulation loop
t = 0
while t < Tmax and 'I' in states.values():
    # Calculate total rates
    total_infection_rate = sum(infection_rates.values())
    total_recovery_rate = sum(recovery_rates.values())
    total_rate = total_infection_rate + total_recovery_rate

    # Generate time to next event
    if total_rate == 0:
        break
    dt = np.random.exponential(1 / total_rate)
    t += dt

    # Determine which event occurs
    # np.randoom.rand generates a number between 0-1
    if np.random.rand() < total_infection_rate / total_rate:
        
        # Infection event
        # A susceptible node is chosen to become infected from the keys of the infection rates dictionary
        # with probabilities proportional to their infection rates
        infected_node = np.random.choice(
            list(infection_rates.keys()),
            p=np.array(list(infection_rates.values())) / total_infection_rate
        )
        # The state is updated to infected
        states[infected_node] = 'I'
        # The node is removed from the infection_rates dictionary since it's no longer susceptible
        del infection_rates[infected_node]
        # The node is added to recovery_rates dictionary with a recovery rate of gamma
        recovery_rates[infected_node] = gamma
    else:
        # Recovery event
        recovered_node = np.random.choice(list(recovery_rates.keys()), p=np.array(
            list(recovery_rates.values())) / total_recovery_rate)
        states[recovered_node] = 'S'
        del recovery_rates[recovered_node]
        # the infection rate is recalculated, it is proportional to the number of its infected neighbors
        infection_rates[recovered_node] = beta * \
            sum(1 for neighbor in G.neighbors(
                recovered_node) if states[neighbor] == 'I')

    # Update rates for neighbors of the affected node
    for neighbor in G.neighbors(infected_node if 'infected_node' in locals() else recovered_node):
        if states[neighbor] == 'S':
            infection_rates[neighbor] = beta * \
                sum(1 for n in G.neighbors(neighbor) if states[n] == 'I')

    # Record the number of infected nodes and current time
    num_infected.append(sum(1 for state in states.values() if state == 'I'))
    event_times.append(t)

# Plot the number of infected nodes over time
plt.figure(figsize=(12, 6))
plt.plot(event_times, num_infected, 'orange', label='Infected')
plt.xlabel('Time')
plt.ylabel('Number of Infected Nodes')
plt.title('Networked SIS Process Simulation')
plt.legend()
plt.grid(True)
plt.show()
