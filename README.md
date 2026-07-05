# Probability, Bayesian Inference, and Gradient Descent

## Group Members

| Name   Contribution |

| Armstrong Hirwa    Part 1 |
| Nazira umucyo      Part 2 |
| Bridget Karungi    Part 3 |
| Kudakwashe Chikovo Part 4 |

---

# Project Overview

This project implements fundamental Machine Learning concepts from scratch using Python.

The project is divided into four parts:

1. Expectation-Maximization (EM) Algorithm
2. Bayesian Probability using IMDb Movie Reviews
3. Manual Gradient Descent Calculations
4. Gradient Descent Implementation in Python

The objective is to understand the mathematics behind these algorithms instead of relying on machine learning libraries.

---

# Project Structure

```
project/
│
├── data/
│   ├── GaltonFamilies.csv
│   └── IMDB Dataset.csv
│
├── notebooks/
│   ├── Merged notebook.ipynb
│   └── Part 3 Merged PDF.pdf
│
├── src/
│   ├── bayes.py
│   ├── em.py
│   └── gradient.py
│
├── _BSE Group Assignments_Task Sheet_Mathematics for Machine Learning_Formative3_Cohort 2_Team26.xlsx
|
├── README.md
│
└── requirements.txt
```

---

# Part 1 – Expectation Maximization (EM)

## Objective

Separate a mixed dataset of Father and Child heights into two Gaussian distributions without using labels.

## Implementation

The following components were implemented from scratch:

- Gaussian Probability Density Function
- Expectation Step
- Maximization Step
- Log-Likelihood Calculation
- EM Optimization Loop
- Posterior Probability Prediction

No machine learning libraries were used.

## Outputs

- Histogram of mixed heights
- Estimated Gaussian distributions
- Parameter updates
- Log-Likelihood
- Classification of a random height

---

# Part 2 – Bayesian Probability

## Objective

Use Bayes' Theorem to calculate

P(Positive | Keyword)

using the IMDb Movie Reviews dataset.

## Keywords

Positive keywords:

- excellent
- great
- amazing
- love

## Probabilities Computed

For every keyword we calculate:

- Prior
- Likelihood
- Marginal
- Posterior

Bayes' theorem is implemented using only basic Python operations.

---

# Part 3 – Manual Gradient Descent

## Objective

Manually compute Gradient Descent updates for a multiple linear regression model.

## Manual Calculations

The following were computed by hand:

- Predictions
- Errors
- Mean Squared Error
- Gradient of MSE
- Weight Updates
- Bias Updates

All calculations use matrix multiplication.

---

# Part 4 – Gradient Descent in Python

## Objective

Convert the manual calculations into Python.

## Implementation

The program performs

- Matrix multiplication
- Prediction
- Cost computation
- Gradient computation
- Parameter updates
- Error tracking

SciPy is used to demonstrate automatic derivative computation.

Matplotlib is used to visualize:

- Weight updates
- Bias updates
- Error reduction

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy

---

# Installation

Clone the repository

```bash
git clone <repository_link>
```

Install dependencies

```bash
pip install -r requirements.txt
```
or
```bash
python -m pip install -r requirements.txt
```

---

# Running the Project

Part 1

```bash
python em.py
```

Part 2

```bash
python bayes.py
```

Part 4

```bash
python gradient.py
```

---

# Results

The EM algorithm successfully estimates two Gaussian distributions corresponding to Fathers and Children.

Bayesian inference computes the probability of a review being positive given selected keywords.

Gradient Descent demonstrates how parameters converge toward values that minimize the Mean Squared Error.

---

# Learning Outcomes

Through this project we learned

- Gaussian Distributions
- Mixture Models
- Expectation-Maximization
- Bayesian Probability
- Bayes' Theorem
- Matrix Multiplication
- Mean Squared Error
- Gradient Descent
- Numerical Optimization

---

# References

- Galton Families Dataset
- IMDb Movie Reviews Dataset
- NumPy Documentation
- Pandas Documentation
- SciPy Documentation
- Matplotlib Documentation
