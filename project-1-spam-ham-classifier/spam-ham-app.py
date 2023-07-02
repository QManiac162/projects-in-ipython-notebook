import streamlit as sl
import joblib
model = joblib.load('spam-ham')
sl.title('SPAM-HAM CLASSIFIER')
ip = sl.text_input('ENTER THE MESSAGE')
op = model.predict([ip])
if sl.button('PREDICT'):
    sl.title(op[0])
