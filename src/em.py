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

# EM loop 

max_iterations = 100
tolerance = 1e-6

previous_ll = -np.inf

print(
    f"{'Iter':<5}"
    f"{'μ1':>10}"
    f"{'μ2':>10}"
    f"{'σ1²':>10}"
    f"{'σ2²':>10}"
    f"{'π1':>10}"
    f"{'π2':>10}"
    f"{'LogLik':>15}"
)

for iteration in range(max_iterations):

    # E-Step
    r1, r2 = expectation_step(
        heights,
        mu1,
        mu2,
        sigma1,
        sigma2,
        pi1,
        pi2
    )

    # M-Step
    mu1, mu2, sigma1, sigma2, pi1, pi2 = maximization_step(
        heights,
        r1,
        r2
    )

    # Log Likelihood
    ll = log_likelihood(
        heights,
        mu1,
        mu2,
        sigma1,
        sigma2,
        pi1,
        pi2
    )

    # Check for convergence
    if abs(ll - previous_ll) < tolerance:
        print(f"Converged after {iteration} iterations.")
        break

    previous_ll = ll

    print(
        f"{iteration:<5}"
        f"{mu1:>10.2f}"
        f"{mu2:>10.2f}"
        f"{sigma1**2:>10.2f}"
        f"{sigma2**2:>10.2f}"
        f"{pi1:>10.3f}"
        f"{pi2:>10.3f}"
        f"{ll:>15.2f}"
    )

# Random heights (For handling random height values during presentation)


height = float(input("Enter a height: "))

def classify_height(height, mu1, mu2, sigma1, sigma2, pi1, pi2):

    p1 = pi1 * gaussian(height, mu1, sigma1)
    p2 = pi2 * gaussian(height, mu2, sigma2)

    total = p1 + p2

    child_probability = p1 / total
    father_probability = p2 / total

    print(f"Height: {height:.2f}")
    print(f"Probability Child : {child_probability:.4f}")
    print(f"Probability Father: {father_probability:.4f}")

    if child_probability > father_probability:
        print("Prediction: Child")
    else:
        print("Prediction: Father")

classify_height(
    height,
    mu1,
    mu2,
    sigma1,
    sigma2,
    pi1,
    pi2
)


# Plot final Gaussian distributions after EM convergence


x = np.linspace(min(heights), max(heights), 500)

plt.hist(heights, bins=30, density=True, alpha=0.6)

plt.plot(x, pi1 * gaussian(x, mu1, sigma1), label="Children")
plt.plot(x, pi2 * gaussian(x, mu2, sigma2), label="Fathers")

plt.legend()
plt.show()
