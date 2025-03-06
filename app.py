import  pickle
import streamlit as st
import  pandas as pd
import  nltk
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
from nltk.tokenize import  word_tokenize
from nltk.stem import  PorterStemmer

ps=PorterStemmer()

with open('vectorizer.pkl','rb') as file:
    vectorizer=pickle.load(file)

with open('model.pkl','rb') as file:
    model=pickle.load(file)


def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = word_tokenize(text)  # Tokenize

    y = []
    for i in text:
        if i.isalnum():
            if i not in stopwords.words('english'):
                y.append(ps.stem(i))

    return ' '.join(y)


st.title('Email Spam Detection')
user_input = st.text_area("Enter your email:")
if st.button('Analyse'):
    text=transform_text(user_input)
    vector=vectorizer.transform([text]).toarray()
    pred=model.predict(vector)
    if pred==1:
        st.write('Spam')
    else:
        st.write('Ham')