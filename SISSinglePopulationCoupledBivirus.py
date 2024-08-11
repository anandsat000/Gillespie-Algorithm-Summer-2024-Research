# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:17:34 2024

@author: satvi
"""

# 1. **Initialization**: 
# Define the initial state of the population, the infection rate, and the recovery rate. 
# The population is divided into two compartments: Susceptible (S) and Infected (I).
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000                  # Total population size
beta1 = 0.3               # Infection rate for virus 1
beta2 = 0.2               # Infection rate for virus 2 
gamma1 = 0.1              # Recovery rate for virus 1
gamma2 = 0.05             # Recovery rate for virus 2
coupling = 0.05           # Coupling effect due to simultaneous bivirus infection
I0 = 1                    # Initial number of infected individuals with virus 1
I02 = 1                   # Initial number of infected individuals with virus 2  
I012 = 1                  # Initial number of infected individuals with both viruses
S0 = N - I0 - I02 - I012  # Initial number of susceptible individuals
Tmax = 250                # Maximum simulation time

# Initial state
S = S0
I = I0
I2 = I02
I12 = 0
t = 0

# Lists to store the time series
time_series = [t]
S_series = [S]
I_series = [I]
I2_series = [I2]
I12_series = [I12]


# 2. **Event Selection and Time Advancement**: 
# Calculate the propensity functions for the infection and recovery events. 
# Use these propensities to determine the time until the next event and which event occurs.

# Gillespie algorithm for SIS model
while t < Tmax and I > 0:
    # Calculate propensities
    infection_rate1 = beta1 * ((I * S) / N + (coupling * I2 * S) / N)
    recovery_rate1 = gamma1 * (I + I12)
    infection_rate2 = beta2 * ((I2 * S) / N + (coupling * I * S) / N)
    recovery_rate2 = gamma2 * (I2 + I12)
    # I to I12
    coupled_infection_rate1 = beta1 * (I * I2 / N)
    # I2 to I12
    coupled_infection_rate2 = beta2 * (I * I2 / N)  
    total_rate = infection_rate1 + recovery_rate1 + infection_rate2 + recovery_rate2 + coupled_infection_rate1 + coupled_infection_rate2
    # Generate the time to the next event
    if total_rate == 0:
        break
    dt = np.random.exponential(1 / total_rate)
    t += dt

    # 3. **Event Execution and State Update**: 
    # Update the number of susceptible and infected individuals based on the selected event.
    # Determine which event occurs
      

    if np.random.rand() < infection_rate1 / total_rate:
        # Infection event for virus 1
        if S > 0:
            S -= 1
            I += 1
    elif np.random.rand() < (infection_rate1 + infection_rate2) / total_rate:
        # Infection event for virus 2
        if S > 0:
            S -= 1
            I2 += 1
    elif np.random.rand() < (infection_rate1 + infection_rate2 + coupled_infection_rate1) / total_rate:
        # Infection event for both viruses (I1 to I12)
        if I > 0:
            I -= 1
            I12 += 1
    elif np.random.rand() < (infection_rate1 + infection_rate2 + coupled_infection_rate1 + coupled_infection_rate2) / total_rate:
        # Infection event for both viruses (I2 to I12)
        if I2 > 0:
            I2 -= 1
            I12 += 1
    elif np.random.rand() < (infection_rate1 + infection_rate2 + coupled_infection_rate1 + coupled_infection_rate2 + recovery_rate2) / total_rate:
        # Recovery event for virus 2
        if I2 > 0:
            S += 1
            I2 -= 1
        elif I12 > 0:
            S += 1
            I12 -= 1
            # only infected by virus 1 transition
            I += 1
    else:
        # Recovery event for virus 1
        if I > 0:
            S += 1
            I -= 1
        elif I12 > 0:
            S += 1
            I12 -= 1
            # only infected by virus 2 transition
            I2 += 1

    # Store the results
    time_series.append(t)
    S_series.append(S)
    I_series.append(I)
    I2_series.append(I2)
    I12_series.append(I12)
        

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_series, S_series, label='Susceptible')
plt.plot(time_series, I_series, label='Infected with Virus 1')
plt.plot(time_series, I2_series, label='Infected with Virus 2')
plt.plot(time_series, I12_series, label='Infected with Both Viruses')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIS Epidemic Model Simulation with Two Viruses')
plt.legend()
plt.show()
