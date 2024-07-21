import streamlit as st
import numpy as np
import pandas as pd
import spacy
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string

    
def get_stop_words():
    with open('./cn_stopwords.txt', encoding='utf8') as f:
        stopwords = f.read()
        return stopwords
    

# @st.cache
def get_model():
    nlp = spacy.load("zh_core_web_md")
    # question = input()
    # doc1= nlp(question)
    df = pd.read_json("./CMID_Traditional.json")
    df['originalText']
    answers = df['originalText']
    return answers
    # for text in answers:
        # doc2 = nlp(text)
        # print(doc1.similarity(doc2), '  ', text)
        



st.title("Simple Bot")

question = st.text_input("請輸入您的問題：")


if len(question) > 0:     
    #score = cosine_similarity(doc1.vector.reshape(1,-1), doc2)
    #index = np.argmax(np.array(score))
    #answer = f'索引：{index}\n類別：{df.y.iloc[index]}\n相似度：{score[0, index]}\n最相似的問題：{df.X.iloc[index]}'
    answer = f'最相似的問題：{get_model()}'
    st.text_area("Bot:", value=answer, height=200, max_chars=None, key=None)
    
else:
    st.text_area("Bot:", height=200, max_chars=None, key=None)


