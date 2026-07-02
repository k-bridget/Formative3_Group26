import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#==================
# Load dataset
#==================

df = pd.read_csv('data/GaltonFamilies.csv')
print(df.head())


#==================
# Mix dataset (For having a new dataset with no structure relation with the original one)
#==================

father = df['father'].to_numpy()
children = df['childHeight'].to_numpy()

heights = np.concatenate((father, children))

np.random.seed(42)
np.random.shuffle(heights)

print(heights[:10])


#==================
# Visualize data
#==================

plt.hist(heights, bins=30)
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.title('Mixed heights')
plt.show()


#==========================
# Initialize parameters (Guess initial parameters for the EM algorithm / Mean / Variance / Mixing coefficients)
#==========================

mu1 = np.percentile(heights, 25)
mu2 = np.percentile(heights, 75)

sigma1 = np.std(heights)
sigma2 = np.std(heights)

pi1 = 0.5
pi2 = 0.5

print(mu1)
print(mu2)
print(sigma1)
print(sigma2)
print(pi1)
print(pi2)


#==========================
# Build Gaussian function (Probability Density Function Equation)
#==========================

def gaussian(data, mu, sigma):
    coefficient = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = np.exp(-((data - mu) ** 2) / (2 * sigma ** 2))
    return coefficient * exponent


#==========================
# Responsabilities (E-step) Example for a single data point (height = 70)
#==========================

height = 70  # Example height value

child = pi1 * gaussian(height, mu1, sigma1)
father = pi2 * gaussian(height, mu2, sigma2)

total = child + father

r_child = child / total
r_father = father / total

print(r_child)
print(r_father)
print(r_child + r_father)  # Should be 1.0


#==========================
# Implement E-step (For finding the responsibilities for in the dataset which means probabilities of each data point belonging to each Gaussian distribution)
#==========================

def expectation_step(data, mu1, mu2, sigma1, sigma2, pi1, pi2):

    # Probability under Gaussian 1 (Children)
    prob1 = pi1 * gaussian(data, mu1, sigma1)

    # Probability under Gaussian 2 (Fathers)
    prob2 = pi2 * gaussian(data, mu2, sigma2)

    # Total probability
    total = prob1 + prob2

    # Responsibilities
    r1 = prob1 / total
    r2 = prob2 / total

    return r1, r2


# M-step (updates the parameters of the Gaussian distributions based on responsibilities calculated in the E-step)


def maximization_step(data, r1, r2):

    mu1 = np.sum(r1 * data) / np.sum(r1)
    mu2 = np.sum(r2 * data) / np.sum(r2)

    sigma1 = np.sqrt(
        np.sum(r1 * (data - mu1) ** 2) / np.sum(r1)
    )

    sigma2 = np.sqrt(
        np.sum(r2 * (data - mu2) ** 2) / np.sum(r2)
    )

    pi1 = np.mean(r1)
    pi2 = np.mean(r2)

    return mu1, mu2, sigma1, sigma2, pi1, pi2

#  log-likelihood 

def log_likelihood(data, mu1, mu2, sigma1, sigma2, pi1, pi2):
    
    likelihood = (
        pi1 * gaussian(data, mu1, sigma1)
        +
        pi2 * gaussian(data, mu2, sigma2)
    )
    
    likelihood = np.clip(likelihood, 1e-12, None)

    return np.sum(np.log(likelihood))
