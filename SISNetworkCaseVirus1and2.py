# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:32:16 2024

@author: satvi
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
n = 50  # Number of nodes
beta1 = 0.3  # Infection rate for Virus 1
beta2 = 0.2  # Infection rate for Virus 2
gamma1 = 0.1  # Recovery rate for Virus 1
gamma2 = 0.05 # Recovery rate for Virus 2
Tmax = 100  # Maximum simulation time

# Initialize the network
G = nx.erdos_renyi_graph(n, 0.1)
states = {node: 'S' for node in G.nodes()}  # All nodes start as susceptible
initial_infected1, initial_infected2 = np.random.choice(G.nodes(), 2, replace=False)
states[initial_infected1] = 'I1'  # Random initial infected node with Virus 1
initial_infected2 = np.random.choice(G.nodes())
states[initial_infected2] = 'I2'  # Random initial infected node with Virus 2

# Initialize the event lists
event_times = []
num_infected1 = [] # Number of infected nodes with Virus 1
num_infected2 = [] # Number of infected nodes with Virus 2

# Function to calculate infection and recovery rates
def calculate_rates(G, states, beta1, beta2, gamma1, gamma2):
    infection_rates1 = {}
    infection_rates2 = {}
    recovery_rates1 = {}
    recovery_rates2 = {}
    for node in G.nodes():
        if states[node] == 'S':
            # Infection rate: beta * number of infected neighbors
            #Virus 1
            infected_neighbors1 = sum(1 for neighbor in G.neighbors(
                node) if states[neighbor] == 'I1')
            infection_rates1[node] = beta1 * infected_neighbors1
            
            #Virus 2
            infected_neighbors2 = sum(1 for neighbor in G.neighbors(
                node) if states[neighbor] == 'I2')
            infection_rates2[node] = beta2 * infected_neighbors2
        
        elif states[node] == 'I1':
            # Recovery rate: gamma
            # Virus 1
            recovery_rates1[node] = gamma1
            
        elif states[node] == 'I2':
            # Virus 2
            recovery_rates2[node] = gamma2

            
            
    return infection_rates1, recovery_rates1, infection_rates2, recovery_rates2


# Initial rates
infection_rates1, recovery_rates1, infection_rates2, recovery_rates2 = calculate_rates(G, states, beta1, beta2, gamma1, gamma2)

# Simulation loop
t = 0
while t < Tmax and ('I1' in states.values() or 'I2' in states.values()):
    # Calculate total rates
    total_infection_rate1 = sum(infection_rates1.values())
    total_infection_rate2 = sum(infection_rates2.values())
    total_recovery_rate1 = sum(recovery_rates1.values())
    total_recovery_rate2 = sum(recovery_rates2.values())
    total_rate = total_infection_rate1 + total_recovery_rate1 + total_infection_rate2 + total_recovery_rate2

    # Generate time to next event
    if total_rate == 0:
        break
    dt = np.random.exponential(1 / total_rate)
    t += dt

    # Determine which event occurs
    # np.randoom.rand generates a number between 0-1
    if np.random.rand() < total_infection_rate1 / total_rate:
        #Infection event for Virus 1
        # A susceptible node is chosen to become infected from the keys of the infection rates dictionary
        # with probabilities proportional to their infection rates
        if total_infection_rate1 > 0: # If it is greater than 0, then infection is possible
            infected_node1 = np.random.choice(
                list(infection_rates1.keys()),
                p=np.array(list(infection_rates1.values())) / total_infection_rate1)
        # The state is updated to infected
        states[infected_node1] = 'I1'
        # The node is removed from the infection_rates dictionary since it's no longer susceptible
        if infected_node1 in infection_rates1:
            del infection_rates1[infected_node1]
        # The node is added to recovery_rates dictionary with a recovery rate of gamma
        recovery_rates1[infected_node1] = gamma1
        
    elif np.random.rand() < (total_infection_rate1 + total_infection_rate2) / total_rate:
        # Infection event for Virus 2
        if total_infection_rate2 > 0:
            infected_node2 = np.random.choice(
                list(infection_rates2.keys()),
                p=np.array(list(infection_rates2.values())) / total_infection_rate2)
        # The state is updated to infected
        states[infected_node2] = 'I2'
        # The node is removed from the infection_rates dictionary since it's no longer susceptible
        if infected_node2 in infection_rates2:
            del infection_rates2[infected_node2]
        # The node is added to recovery_rates dictionary with a recovery rate of gamma
        recovery_rates2[infected_node2] = gamma2
    
    elif np.random.rand() < (total_infection_rate1 + total_infection_rate2 + total_recovery_rate1) / total_rate:
        #Recovery event for Virus 1
        if total_recovery_rate1 > 0:
            recovered_node1 = np.random.choice(list(recovery_rates1.keys()), p=np.array(
                list(recovery_rates1.values())) / total_recovery_rate1)
        states[recovered_node1] = 'S'
        if recovered_node1 in recovery_rates1:
            del recovery_rates1[recovered_node1]
        # the infection rate is recalculated, it is proportional to the number of its infected neighbors
        infection_rates1[recovered_node1] = beta1 * \
            sum(1 for neighbor in G.neighbors(
                recovered_node1) if states[neighbor] == 'I1')
        
    else:
        # Recovery event for Virus 2
        if total_recovery_rate2 > 0:
            recovered_node2 = np.random.choice(list(recovery_rates2.keys()), p=np.array(
                list(recovery_rates2.values())) / total_recovery_rate2)
        states[recovered_node2] = 'S'
        if recovered_node2 in recovery_rates2:
            del recovery_rates2[recovered_node2]
        # the infection rate is recalculated, it is proportional to the number of its infected neighbors
        infection_rates2[recovered_node2] = beta2 * \
            sum(1 for neighbor in G.neighbors(
                recovered_node2) if states[neighbor] == 'I2')

    # Update rates for neighbors of the affected node for Virus 1
    if 'infected_node1' in locals():
        for neighbor in G.neighbors(infected_node1):
            if states[neighbor] == 'S':
                infection_rates1[neighbor] = beta1 * sum(1 for n in G.neighbors(neighbor) if states[n] == 'I1')
                
    # Update rates for neighbors of the affected node for Virus 2
    if 'infected_node2' in locals():
        for neighbor in G.neighbors(infected_node2):
            if states[neighbor] == 'S':
                infection_rates1[neighbor] = beta2 * sum(1 for n in G.neighbors(neighbor) if states[n] == 'I2')
    
    # Record the number of infected nodes and current time for both Viruses
    num_infected1.append(sum(1 for state in states.values() if state == 'I1'))
    num_infected2.append(sum(1 for state in states.values() if state == 'I2'))
    event_times.append(t)

# Plot the number of infected nodes over time
plt.figure(figsize=(12, 6))
plt.plot(event_times, num_infected1, label='Infected with Virus 1')
plt.plot(event_times, num_infected2, label='Infected with Virus 2')
plt.xlabel('Time')
plt.ylabel('Number of Infected Nodes')
plt.title('Networked SIS Process Simulation for Two Viruses')
plt.legend()
plt.grid(True)
plt.show()
