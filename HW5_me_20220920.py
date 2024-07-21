import streamlit as st
import numpy as np
import pandas as pd
import spacy
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
from scipy.spatial.distance import cosine

st.title("Simple Bot")

question = st.text_input("請輸入您的問題：")


def get_stop_words():
    with open('./cn_stopwords.txt', encoding='utf8') as f:
        stopwords = f.read()
        return stopwords
        

# @st.cache
def get_model():
    nlp = spacy.load("zh_core_web_md")
    doc1= nlp(question)
    df = pd.read_json("./CMID_Traditional.json")
    df['originalText']
    df2 = pd.DataFrame()
    df2['X'] = df.originalText
    df2['y'] = df.label_36class.map(lambda x: x[0].replace("'", ""))
    answers = df2.X.values
    #return answers
    #score = cosine_similarity(tfidf_vectorizer.transform([question]), tfidf_matrix)
    #return score
    #print(doc1.similarity(doc2), '  ', answers)
    doc1 = nlp(question).vector
    #doc2 = nlp(answers).vector

    for text in answers:
        doc2 = nlp(text).vector
        dist = cosine(doc1, doc2)

        if dist >=0.5:
            return dist, text
        else:
            return "None"
    
    with open('./cn_stopwords.txt', encoding='utf8') as f:
        stopwords = f.read()
        

    

if len(question) > 0:         
    answer = f'最相似的問題：{get_model()}'
    st.text_area("Bot:", value=answer, height=200, max_chars=None, key=None)
    
else:
    st.text_area("Bot:", height=200, max_chars=None, key=None)


