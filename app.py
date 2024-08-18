from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
import re
import numpy as np 
from bs4 import BeautifulSoup
from keras.models import load_model
import joblib
from nltk.stem import PorterStemmer



model = load_model('nn_disease_detection.h5')
label_encoder = joblib.load('label_encoder.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')


app = Flask(__name__)

def removeApostrophe(x):
    x = re.sub("won\`t", "will not", x)
    x = re.sub("can\`t", "can not", x)
    x = re.sub(r"couldn\`t", "could not", x)
    x = re.sub("wouldn\`t", "would not", x)
    x = re.sub("n\`t", " not", x)
    x = re.sub("\`re", " are", x)
    x = re.sub(r"\`s", " is", x)
    x = re.sub("\`d", " would", x)
    x = re.sub(r"\`ll", " will", x)
    x = re.sub(r"\`t", " not", x)
    x = re.sub(r"\`ve", " have", x)
    x = re.sub(r"\`m", " am", x)
    return x



def removeHTMLTags(x):
    soup = BeautifulSoup(x, 'lxml')
    return soup.get_text()


def removeSpecialChars(x):
    return re.sub('[^a-zA-Z]', ' ', x)

def removeurl(x):
    x = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', x, flags=re.MULTILINE)
    return(x)


def removeAlphaNumericWords(x):
    return re.sub("\S*\d\S*", "", x).strip()

def dostemming(x):

    #Removing Stopwords and Lemmatization
    porter = PorterStemmer()
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
      
    return x

def preprocess_text(x):
    x = removeurl(x)
    x = removeHTMLTags(x)
    x = removeApostrophe(x)
    x = removeAlphaNumericWords(x)
    x = removeSpecialChars(x)
    x = x.strip().lower()
    x = dostemming(x)
    return x


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    # data2 = request.form['b']
    # data3 = request.form['c']
    # data4 = request.form['d']
    # arr = np.array([[data1, data2, data3, data4]])

    processed_text = preprocess_text(data1)


    # Transform the processed text using the TfidfVectorizer
    text_count_3 = tfidf.transform([processed_text])

    dense_input = text_count_3.toarray()

    # Predict using the model
    model_output = model.predict(dense_input)


    # Get the predicted class
    predicted_class_index = np.argmax(model_output, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_index])



    # pred = model.predict(arr)
    return render_template('after.html', data=predicted_label)


if __name__ == "__main__":
    app.run(debug=True)















