import streamlit as st
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string

name = "Chinese medicine chatbot 101" 
situation_1 = "吃壞肚子, 還有什麼症狀?" 
situation_2 = "去看醫生!"

responses = { 
"你叫什麼名字?": [ 
    "他們叫我 {0}".format(name), 
    "我是 {0}".format(name), 
    "親愛的,我是 {0}".format(name) ],
"我肚子痛": [ 
    "我認為應該是 {0}".format(situation_1), 
    "可能是 {0} today".format(situation_1), 
    "嗯...讓我想想,應該是 {0} today".format(situation_1) ],
"我長水痘": [ 
    "趕快去看醫生!", 
    "不管怎麼樣,請趕快去看醫生", 
    "你有看醫生嗎? 快去吧", ],
"不想吃飯噁心": [ 
    "哇! 怎麼會這樣,你還好吧?! {0}".format(situation_2), 
    "{0}! 不要拖延!".format(situation_2), 
    "天啊! 你還好吧?! {0}! 多久了?".format(situation_2), ],
    
"你好嗎?": [ 
    "我很好喔! ", 
    "非常好!", 
    "好的不得了, 你呢?", ],
    
"請問你是男生還是女生?": [ 
    "我是女生! ", 
    "女的啊", 
    "問這幹嘛?!", ],
    

"default": [
    "I'm a Chinese medicine chatbot, how can I help you?"] }
    
def get_stop_words():
    stop_words = stopwords.words('chinese')
    stop_words += string.punctuation
    return stop_words
    
@st.cache()    
def get_cleaned_question_list(responses):
    question_list = list(responses.keys())[:-1] 
    stop_words = get_stop_words()
    cleaned_question_list = []
    for q in question_list:
        q = q.lower()
        list1 = [w for w in q.split(' ') if not w in stop_words]
        cleaned_question_list.append(' '.join(list1))
        
    return question_list, cleaned_question_list
    
    

def get_response(question, question_list):
    score = cosine_similarity(tfidf_vectorizer.transform([question]), tfidf_matrix)
    # print(score)
    # if question in responses: 
    if max(score[0]) >= 0.75 : 
        key_no = np.argmax(score[0])
        bot_message = random.choice(responses[question_list[key_no]])
    else: 
        bot_message = random.choice(responses["default"])
    return bot_message    


st.title("Chinese medicine chatbot")

question = st.text_input("請輸入您的問題：")
tfidf_vectorizer = TfidfVectorizer()
if len(question) > 0:
    question_list, cleaned_question_list = get_cleaned_question_list(responses)
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_question_list)
    st.text_area("Bot:", value=get_response(question, question_list), height=200, max_chars=None, key=None)
else:
    st.text_area("Bot:", height=200, max_chars=None, key=None)



