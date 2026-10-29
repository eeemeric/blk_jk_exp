#     tokens_before_choice     p_hit  n_trials
# 0                      0  0.992095       506
# 1                      1  1.000000        30
# 2                      2  0.975000        40
# 3                      3  1.000000        44
# 4                      4  1.000000        41
# 5                      5  1.000000        44
# 6                      6  1.000000        61
# 7                      7  1.000000        63
# 8                      8  1.000000        53
# 9                      9  1.000000        67
# 10                    10  0.995146       206
# 11                    11  0.898305        59
# 12                    12  0.758065        62
# 13                    13  0.682540        63
# 14                    14  0.423729        59
# 15                    15  0.135593        59
# 16                    16  0.057971        69
# 17                    17  0.016129        62
# 18                    18  0.020833        48
# 19                    19  0.000000        50
# 20                    20  0.063291        79

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# 1: Generate real data
x = np.linspace(0, 20, 21)
p_hit = np.array([0.992095, 1.000000, 1.000000, 0.975000, 1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
                  1.000000,  0.995146, 0.898305, 0.758065, 0.682540, 0.423729, 0.135593, 0.057971, 0.016129, 0.020833, 0.000000,])
n_trials = np.array([506, 30, 40, 44, 41, 44, 61, 63, 53, 67, 206,
                     59, 62, 63, 59, 59, 69, 62, 48, 50, 79 ])

# 2. Fit Binomial GLM (Logistic Regression)
X = sm.add_constant(x)                  # Add intercept
model = sm.GLM(p_hit, X, family=sm.families.Binomial())
result = model.fit()

# 3. Predict and Plot
# Generate smooth x-values for the fitted curve
x_smooth = np.linspace(0, 20, 100)
X_smooth = sm.add_constant(x_smooth)
y_pred = result.predict(X_smooth)

plt.plot(x, p_hit,'o')
plt.plot(x_smooth, y_pred, color='red')
plt.xlabel('tokens')
plt.ylabel('p(hit)')
plt.show()
