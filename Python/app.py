
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB





df = pd.read_csv(r"C:\Users\7rbe2\OneDrive\سطح المكتب\Main projects\Grad project\Phishward\PhishWarden\app\Python\combined_data.csv")

df["text"] = df["text"].str.lower()


def remove_special_characters(word):
    return word.translate(str.maketrans("", "", string.punctuation))


def remove_stop_words(words):
    return [word for word in words if word not in ENGLISH_STOP_WORDS]


def remove_hyperlink(word):
    return re.sub(r"http\S+", "", word)


df["text"] = df["text"].apply(remove_special_characters)
ENGLISH_STOP_WORDS = set(stopwords.words("english"))
df["text"] = df["text"].apply(word_tokenize)
df["text"] = df["text"].apply(remove_stop_words)
df["text"] = df["text"].apply(" ".join)
df["text"] = df["text"].apply(remove_hyperlink)


cv = CountVectorizer()
feature = cv.fit_transform(df.text)


# Replace 'new_email_text' with the text of the email you want to test



def detection(new_email_text):
# Clean the email text (convert to lowercase, remove special characters, hyperlinks, etc.)
    new_email_text = new_email_text.lower()
    new_email_text = remove_special_characters(new_email_text)
    new_email_text = remove_hyperlink(new_email_text)

    # Tokenize the text
    new_email_tokens = word_tokenize(new_email_text)

    # Remove stop words
    new_email_tokens = remove_stop_words(new_email_tokens)

    # Rejoin the tokens into a single string
    new_email_cleaned = ' '.join(new_email_tokens)

    # Convert the cleaned email text into features using the same vectorizer used for training

    #new_email_features = cv.transform([new_email_cleaned])
    new_email_features = cv.transform([new_email_cleaned])
    # Apply the trained model to predict whether the email is phishing or not


    # Assuming 'X_test' contains your test data


    nb_loaded = joblib.load(r'C:\Users\7rbe2\OneDrive\سطح المكتب\Main projects\Grad project\Phishward\PhishWarden\app\Python\nb_model.joblib')


    new_email_features = cv.transform([new_email_cleaned])
    # Apply the trained model to predict whether the email is phishing or not
    prediction = nb_loaded.predict(new_email_features)

    # Apply the trained model to predict the probability of the email being phishing
    probability = nb_loaded.predict_proba(new_email_features)
    ph =-1
    prob = -1
    # Print the prediction and probability score
    if prediction[0] == 1:
        ph = 1
        prob = format(probability[0][1] * 100, '.2f')
        print("The email is classified as phishing with a probability of {:.2f}%.".format(probability[0][1] * 100))
    else:
        print("The email is classified as legitimate with a probability of {:.2f}%.".format(probability[0][0] * 100))
        ph = 0
        prob = format(probability[0][0] * 100, '.2f')
    return ph, prob

from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route('/ml', methods=['POST'])
def ml_endpoint():
    content = request.json 
    input_text = content['input_text']  
    result, prob =detection(input_text)
    print(result)
    return jsonify({'det': result, 'prob': prob})
__name__ == '__main__'
app.run(debug=True, port=5001)