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
    #question = input()
    doc1= nlp(question)
    df = pd.read_json("./CMID_Traditional.json")
    df['originalText']
    answers = df['originalText']
    #df2 = pd.DataFrame()
    # df2['X'] = df.originalText
    # df2['y'] = df.label_36class.map(lambda x: x[0].replace("'", ""))
    # answers = df2.X.values
    
    #answer = df['originalText'].values
    #return answers
    #score = cosine_similarity(tfidf_vectorizer.transform([question]), tfidf_matrix)
    #return score
    #print(doc1.similarity(doc2), '  ', answers)
    doc1 = nlp(question).vector
    #doc2 = nlp(answers).vector

    #先註解掉,之後再想如何去除停用詞
    # def remove_stop_words(text):
        # doc1 = nlp(text)
        # new_text = ''
        # for token in doc1:
            # if token.is_oov or token.is_stop: continue
            # new_text += token.text
            # doc2 = nlp(new_text)  
            # print(text)
        # for token in doc2:
            # if token.is_oov or token.is_stop or len(token.text)<=1: continue
            # print(token.text)
        

    for text in answers:     
        answers = nlp(text).vector
        dist = cosine(doc1, answers)
        #return dist, text
    
        if dist >=0.5:
            return dist, text
        else:
            return "None"
  
        
    # if max(score[0]) >= 0.75 : 
        # key_no = np.argmax(score[0])
        # bot_message = random.choice(responses[question_list[key_no]])
    # else: 
        # bot_message = "No answer"
    # return bot_message  
    
    
    with open('./cn_stopwords.txt', encoding='utf8') as f:
        stopwords = f.read()
        

    

if len(question) > 0:     
    #score = cosine_similarity(doc1.vector.reshape(1,-1), doc2)
    #index = np.argmax(np.array(score))
    #answer = f'索引：{index}\n類別：{df.y.iloc[index]}\n相似度：{score[0, index]}\n最相似的問題：{df.X.iloc[index]}'
    
    answer = f'最相似的問題：{get_model()}'
    st.text_area("Bot:", value=answer, height=200, max_chars=None, key=None)
    
else:
    st.text_area("Bot:", height=200, max_chars=None, key=None)


