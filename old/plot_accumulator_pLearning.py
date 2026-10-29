# plot_accumulator_ProbabilityLearning.py
# eemeric2@jhu.edu
# erik.emeric@gmail.com

import numpy as np
import matplotlib.pyplot as plt

def simulate_stochastic_accumulator(steps=100, drift=0.1, noise_scale=0.5):
    """
    Simulates a stochastic accumulator with positive drift.
    """
    # Initialize accumulator
    accumulator = 0
    trajectory = [accumulator]
    
    for _ in range(steps):
        # 1. Stochastic component: random noise (Gaussian)
        noise = np.random.normal(0, noise_scale)
        
        # 2. Accumulation: current value + drift + noise
        accumulator += drift + noise
        trajectory.append(accumulator)
        
    return trajectory

# Parameters
n_steps = 60
x = list(range(61))
print(len(x))

n_trials = 1
plt.figure(figsize=(10, 6))

# Run multiple trials to show stochasticity
for i in range(1):
    path = simulate_stochastic_accumulator(steps=n_steps, drift=1.5, noise_scale=2)

print([len(x), len(path)])
plt.plot(x, path)

# # Parameters
# n_steps = 60*30
# x = list(range(60*30+1))
# print(len(x))
# 
# n_trials = 1
# # Run multiple trials to show stochasticity
# for i in range(1):
#     path = simulate_stochastic_accumulator(steps=n_steps, drift=0.05, noise_scale=0.2)
# 
# print([len(x), len(path)])
# plt.plot(x, path)


plt.title('Stochastic Accumulator: Evidence Increasing Over Time')
plt.xlabel('Time Steps (frames)')
plt.ylabel('Accumulated Evidence')
plt.axhline(0, color='black', linestyle='--')
plt.grid(True)
plt.savefig('/home/eeemeric/Documents/Ushuer_project/accumulator_Percept.png')
# plt.show()
