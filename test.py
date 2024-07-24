import numpy as numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Read the CSV file into a DataFrame
data = pd.read_csv('movie_reviews.csv')

# Extract features and labels
x = data['review']
y = data['sentiment']

# Split the data into training and testing sets - 20% for testing and 80% for training 
# to ensure data is split in reproducible way The number 42 is arbitrary but using the same seed value ensures that you get the same split every time you run the code.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the CountVectorizer
# ML gorithms generally cannot work with raw text data directly. Instead, they require numerical input. CountVectorizer converts text data into a numerical format (bag-of-words) that can be used by machine learning models
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_vectorized = vectorizer.fit_transform(x_train)

# Transform the test data
X_test_vectorized = vectorizer.transform(x_test)

# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
# Trains the Naive Bayes model on the training data
model.fit(X_train_vectorized, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict the sentiment of a given review
def predict_sentiment(review):
    # Transform the review text into a vectorized format using the trained CountVectorizer
    vectorized_review = vectorizer.transform([review])
    # Predict the sentiment using the trained model
    prediction = model.predict(vectorized_review)
    # Return "Positive" if the prediction is 1, otherwise return "Negative"
    return "Positive" if prediction[0] == 1 else "Negative"

# List of new reviews to be analyzed
new_review = [
    "this movie is next level! i loved it alot!",  # Example of a positive review
    "terrible film , assal bale! waste of time and money.",  # Example of a negative review
    "it was okay, not that good, and nopt that bad eaither."  # Example of a neutral review
]

# Loop through each review in the new_review list
for review in new_review:
    # Print the review text
    print(f"Review: {review}")
    # Predict and print the sentiment of the review
    print(f"Predicted sentiment: {predict_sentiment(review)}\n")

# Print a message indicating that the user can enter their own reviews
print("Now can enter your own reviews!")

# Infinite loop to continuously ask the user for input
while True:
    # Prompt the user to enter a movie review or type 'quit' to stop
    user_review = input("enter a movie review (or say 'quit' to stop it!):")
    # Check if the user wants to quit
    if user_review.lower() == "quit":
        break  # Exit the loop if the user types 'quit'
    # Predict and print the sentiment of the user's review
    print(f"Predicted sentiment: {predict_sentiment(user_review)}\n")

# Print a closing message
print("thanks guys! jagratta! ede manam cheyyalsindi!")
