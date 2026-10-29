# plot_decks_pBART.py
# eemeric2@jhu.edu
# erik.emeric@gmail.com


import numpy as np
import matplotlib.pyplot as plt

points_accumulated = list(range(0,21))
print(f"points accumulated: {points_accumulated}")

# Create 3 rows and 1 column
fig, axs = plt.subplots(3, 1, figsize=(6, 8))

# Card values
values = list(range(1, 10 + 1))
print(f"Card values: {values}")

token_weights = [1]*9 + [4]
# Multiply element-wise
result = [v * w for v, w in zip(values, token_weights)]
print(f"High risk token weights: {token_weights}")
print(f"Mean: {sum(result) / len(result)}")
# Access each subplot using indexing (0 to 2)
# Create bar chart
axs[0].bar(values, token_weights, color='skyblue', edgecolor='navy')
axs[0].axvline(x=sum(result) / len(result), color='r', linestyle='--')
axs[0].set_title(f"High Risk Deck: {sum(result) / len(result)}")

token_weights = [1]*4 + [4] + [1]*5
# Multiply element-wise
result = [v * w for v, w in zip(values, token_weights)]
print(f"Med  risk token weights: {token_weights}")
print(f"Mean: {sum(result) / len(result)}")
# Access each subplot using indexing (0 to 2)
# Create bar chart
axs[1].bar(values, token_weights, color='skyblue', edgecolor='navy')
axs[1].axvline(x=sum(result) / len(result), color='r', linestyle='--')
axs[1].set_title(f"Medium Risk Deck: {sum(result) / len(result)}")
axs[1].set_ylabel('Frequency')

token_weights = [4] + [1]*9
# Multiply element-wise
result = [v * w for v, w in zip(values, token_weights)]
print(f"Low risk token weights : {token_weights}")
print(f"Mean: {sum(result) / len(result)}")
# Access each subplot using indexing (0 to 2))
# Create bar chart
axs[2].bar(values, token_weights, color='skyblue', edgecolor='navy')
axs[2].axvline(x=sum(result) / len(result), color='r', linestyle='--')
axs[2].set_title(f"Medium Risk Deck: {sum(result) / len(result)}")
axs[2].set_xlabel('Card Value')

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()
# Save the plot (format inferred from file extension)
plt.savefig('plot_decks_pBART.png')
# Show the plot (must come AFTER savefig)
# plt.show()
plt.close()

# get points_accumulated versus p(bust)
# make array of cards
all_crds = np.array([])
for i in range(0,len(token_weights)):    
    j = 0
    crd = np.tile(values[i],[1,token_weights[i]])
    all_crds = np.append(all_crds,crd)
    
print(f"cards in deck: {all_crds}")
    
# for i in range(0,21):
#     print(points_accumulated[i])
    