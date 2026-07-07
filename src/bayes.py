import pandas as pd


#==================
# Load dataset
#==================
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'IMDB Dataset.csv'))

print(df.head())


#========================
# Count total reviews
#========================

total_reviews = len(df)

print(total_reviews)


#==========================
# Count positive reviews
#==========================

positive_reviews = len(
    df[df["sentiment"] == "positive"]
)

print(positive_reviews)


#===================
# Count the prior (P(positive))
#===================

prior = positive_reviews / total_reviews

print(prior)


#===================
# Choose a keyword
#===================

keyword = "excellent"


#===============================
# Count positive with keyword (P(keyword | positive))
#===============================

positive_keyword = df[
    (df["sentiment"] == "positive")
    &
    (
        df["review"]
        .str.lower()
        .str.contains(keyword)
    )
]
positive_keyword_count = len(positive_keyword)

print(f"Count of positive reviews containing '{keyword}': {positive_keyword_count}")


#===============================
# Count the likelihood (P(keyword | positive)) Positive reviews with keyword / Total positive reviews
#===============================

likelihood = positive_keyword_count / positive_reviews

print(f"Likelihood of '{keyword}' given positive sentiment: {likelihood}")


#===============================
# Count all reviews with keyword (P(keyword))
#===============================

keyword_reviews = df[
    df["review"]
    .str.lower()
    .str.contains(keyword)
]

keyword_count = len(keyword_reviews)

print(f"Count of reviews containing '{keyword}': {keyword_count}")

#===============================
# Count the marginal (P(keyword)) All reviews with keyword / Total reviews
#===============================

marginal = keyword_count / total_reviews

print(f"Marginal probability of '{keyword}': {marginal}")


#===============================
# Count the Bayes theorem (P(positive | keyword)) Likelihood * Prior / Evidence
#===============================

posterior = (
    likelihood
    * prior
) / marginal

print(f"Posterior probability of positive sentiment given '{keyword}': {posterior:.2f}")


#=========================================
# Reusable function for Bayes theorem
#=========================================
print("\n================\n BAYES RESULTS\n================\n")
print(
    f"{'---------------------------------------------------------------------------------------\n'}"
    f"| {'Keywords':<15} |"
    f"{'Prior':>15} |"
    f"{'Likelihood':>15} |"
    f"{'Marginal':>15} |"
    f"{'Posterior':>15} |"
    f"{'\n---------------------------------------------------------------------------------------'}"
)
def bayes_keyword(df, keyword_input):

    total_reviews = len(df)

    positive_reviews = len(df[df["sentiment"] == "positive"])

    prior = positive_reviews / total_reviews

    positive_keyword = len(
        df[
            (df["sentiment"] == "positive")
            &
            (
                df["review"]
                .str.lower()
                .str.contains(keyword_input)
            )
        ]
    )

    likelihood = positive_keyword / positive_reviews

    keyword_count = len(
        df[
            df["review"]
            .str.lower()
            .str.contains(keyword_input)
        ]
    )

    marginal = keyword_count / total_reviews

    posterior = (likelihood * prior) / marginal

    print(
        f"| {keyword_input:<15} |"
        f"{prior:>15.4f} |"
        f"{likelihood:>15.4f} |"
        f"{marginal:>15.4f} |"
        f"{posterior:>15.4f} |"
    )


#===========================
# Calling bayes function
#===========================

keywords = ["excellent", "great", "amazing", "love"]

for keyword_input in keywords:
    bayes_keyword(df, keyword_input)