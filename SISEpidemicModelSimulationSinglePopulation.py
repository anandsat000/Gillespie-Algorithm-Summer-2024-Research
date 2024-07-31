# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:04:22 2024

@author: satvi
"""

# 1. **Initialization**: 
# Define the initial state of the population, the infection rate, and the recovery rate. 
# The population is divided into two compartments: Susceptible (S) and Infected (I).
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000          # Total population size
beta = 0.3        # Infection rate
gamma = 0.1       # Recovery rate
I0 = 1            # Initial number of infected individuals
S0 = N - I0       # Initial number of susceptible individuals
Tmax = 100        # Maximum simulation time

# Initial state
S = S0
I = I0
t = 0

# Lists to store the time series
time_series = [t]
S_series = [S]
I_series = [I]

# 2. **Event Selection and Time Advancement**: 
# Calculate the propensity functions for the infection and recovery events. 
# Use these propensities to determine the time until the next event and which event occurs.

# Gillespie algorithm for SIS model
while t < Tmax and I > 0:
    # Calculate propensities
    infection_rate = beta * S * I / N
    recovery_rate = gamma * I
    total_rate = infection_rate + recovery_rate
    # Generate the time to the next event
    if total_rate == 0:
        break
    dt = np.random.exponential(1 / total_rate)
    t += dt

    # 3. **Event Execution and State Update**: 
    # Update the number of susceptible and infected individuals based on the selected event.
    # Determine which event occurs

    if np.random.rand() < infection_rate / total_rate:
        # Infection event
        S -= 1
        I += 1
    else:
        # Recovery event
        S += 1
        I -= 1

    # Store the results
    time_series.append(t)
    S_series.append(S)
    I_series.append(I)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_series, S_series, label='Susceptible')
plt.plot(time_series, I_series, label='Infected')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIS Epidemic Model Simulation')
plt.legend()
plt.show()