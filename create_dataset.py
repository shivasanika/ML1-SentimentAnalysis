import pandas as pd

# Creating a dictionary with movie reviews and corresponding sentiments
data = {
    'review': [
        "this movie is so good! i love it!",  # Positive review
        "worst film , i hate it the most!",  # Negative review
        "pakka 1000 cr, super hit! amazing",  # Positive review
        "loved it will watch again",  # Positive review
        "not bad , just okay!",  # Negative review
        "average, one time watch",  # Positive review
        "one man show",  # Positive review
        "great movie",  # Positive review
        "i hate that movie worst of all time",  # Negative review
        "good movie , can be done better",  # Positive review
        "had an amazing time there",  # Positive review
        "i personally didnt like it!",  # Negative review
        "dont go with family,",  # Positive review
        "can watch 1 time!"  # Negative review
    ],
    'sentiment': [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]
}

# Converting the dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# Saving the DataFrame to a CSV file named 'movie_reviews.csv' without the index
df.to_csv('movie_reviews.csv', index=False)

# Printing a confirmation message
print("dataset created and saved as 'movie_reviews.csv'")
