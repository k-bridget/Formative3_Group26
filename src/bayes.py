import pandas as pd


#==================
# Load dataset
#==================

df = pd.read_csv("data/IMDB Dataset.csv")

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

:positive_keyword = df[
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
