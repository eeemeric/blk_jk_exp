# deck_value_risk_pBART.py
# eemeric2@jhu.edu
# erik.emeric@gmail.com
import numpy as np
import matplotlib.pyplot as plt
import itertools

points_accumulated = list(range(0,21))
# print(f"points accumulated: {points_accumulated}")

# Card values
values = list(range(1, 11))
# print(f"Card values: {values}")

deck_lables = ["High value", "moderate value", "low value"]
card_weights = [[1]*9 + [4],[1]*4 + [4] + [1]*5, [4] + [1]*9 ]
# print(f"card weights: {card_weights}")

# get mean of weighted decks
mean_weighted_decks = [None]*len(card_weights)
weighted_card_values = [None]*len(card_weights)
weighted_decks = [0]*len(card_weights)
pBUST_fx = [None]*len(weighted_decks)

# Multiply element-wise
for i in range(len(card_weights)):
    weighted_card_values[i] = [v * w for v, w in zip(values, card_weights[i])]
    mean_weighted_decks[i] = sum(weighted_card_values[i])/len(weighted_card_values[i])
    for j in range(len(values)):
        if j == 0:
            weighted_decks[i] = [values[j]]*card_weights[i][j]
        else:
            # weigh the decks
            weighted_decks[i].extend([values[j]]*card_weights[i][j])


# for each deck and points_accumulated, what is the probability of a BUST trial?
for i in range(len(weighted_decks)):
    for j in range(len(points_accumulated)):
        # use a list comprehension to find what proportion of cards result in bust for each deck
        # add single element from points_accumulated to every element of decks[i]
        pts_after_hit = ([x + points_accumulated[j] for x in weighted_decks[i]])
        # how many busts (pts_after_hit > 0)?
        bust = [x > 20 for x in pts_after_hit]
        pBST = [sum(bust) / len(bust)]
        if j == 0:
            pBUST_fx[i] = pBST
        else:
            pBUST_fx[i].extend(pBST)

# Create 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(7, 12))

# Left column: bar plots of card distributions
for i in range(len(card_weights)):
    ax = axs[i, 0]
    ax.bar(values, card_weights[i], color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(mean_weighted_decks[i], color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_weighted_decks[i]:.2f}')
    ax.set_xlabel('Card Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{deck_lables[i]} Deck')
    ax.set_xticks(values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Right column: P(BUST) as function of accumulated points
for i in range(len(pBUST_fx)):
    ax = axs[i, 1]
    ax.plot(points_accumulated, pBUST_fx[i], marker='o', linewidth=2, 
            markersize=6, color='darkred')
    ax.fill_between(points_accumulated, pBUST_fx[i], alpha=0.3, color='red')
    # Draw a black, dashed horizontal line across the entire plot at y = 20
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Points Accumulated')
    ax.set_ylabel('P(BUST|HIT)')
    ax.set_title(f'{deck_lables[i]} Deck - P(BUST) by Accumulated Points')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 21, 2))

plt.tight_layout()
plt.savefig('deck_distributions_and_bust_probabilities.png', dpi=300, bbox_inches='tight')
plt.show()