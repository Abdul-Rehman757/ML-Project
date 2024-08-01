import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create a synthetic dataset
np.random.seed(42)

# Generate spam emails
spam_words = ['offer', 'money', 'discount', 'limited', 'click', 'now', 'free', 'cash', 'prize', 'winner']
spam_emails = []
for _ in range(1000):
    email = ' '.join(np.random.choice(spam_words, size=np.random.randint(5, 10)))
    spam_emails.append((email, 1))  # 1 indicates spam

# Generate non-spam emails
non_spam_words = ['meeting', 'project', 'report', 'schedule', 'update', 'team', 'deadline', 'client', 'presentation', 'budget']
non_spam_emails = []
for _ in range(1000):
    email = ' '.join(np.random.choice(non_spam_words, size=np.random.randint(5, 10)))
    non_spam_emails.append((email, 0))  # 0 indicates not spam

# Combine and shuffle the dataset
all_emails = spam_emails + non_spam_emails
np.random.shuffle(all_emails)

# Create a DataFrame
df = pd.DataFrame(all_emails, columns=['text', 'label'])

# Save the dataset
df.to_csv('emails.csv', index=False)
print("Dataset created and saved as 'emails.csv'")



# Load the dataset
data = pd.read_csv('emails.csv')

# Split the data into features (X) and target (y)
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text into a matrix of token counts
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_counts)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Function to classify new emails
def classify_email(email_text):
    email_counts = vectorizer.transform([email_text])
    prediction = clf.predict(email_counts)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test the classifier with some example emails
test_emails = [
    "Free cash prize for you! Click now!",
    "Team meeting scheduled for tomorrow at 10 AM",
    "Limited time discount offer on all products",
    "Project report due by end of week",
]

for email in test_emails:
    print(f"Email: {email}")
    print(f"Classification: {classify_email(email)}\n")