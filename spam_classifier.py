from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# 1. Training data: emails and their correct labels
emails = [
    "Win a free prize now",
    "Claim your free cash reward",
    "Congratulations you won a lottery",
    "Get free money today",
    "Click here to claim your prize",
    "You have won a free gift",
    "Let's schedule a meeting tomorrow",
    "Please send me the project report",
    "Your appointment is confirmed",
    "Can we discuss the project today",
    "Please review the attached document",
    "Your meeting starts at 10 AM",
    "I want to thank you for your help with the project",
    "Now time deploy the code to production",
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam",
    "Not Spam", "Not Spam", "Not Spam", "Not Spam" , "Not Spam"
]

# 2. Create a pipeline with CountVectorizer and MultinomialNB
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# 3. Train the model
model.fit(emails, labels)

print("Model trained successfully!")


while True:
    input_email = input("Enter an email to classify (or type 'exit' to quit): ")
    if input_email.lower() == 'exit':
        break

    # 4. Predict the label for the input email
    prediction = model.predict([input_email])
    print(f"The email is classified as: {prediction[0]}")