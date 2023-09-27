import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# set app config
st.set_page_config(page_title="TMDB", page_icon="🍿", layout="wide")    
st.markdown(f"""
            <style>
            .stApp {{background-image: url(""); 
                     background-attachment: fixed;
                     base: light;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# Load models and MovieDB
df = joblib.load('models/movie_db.df')
tfidf_matrix = joblib.load('models/tfidf_mat.tf')
tfidf = joblib.load('models/vectorizer.tf')
cos_mat = joblib.load('models/cos_mat.mt')

# define functions
def get_keywords_rec(keywords, n):
    keywords = keywords.split()
    keywords = " ".join(keywords)
    # transform the string to vector representation
    key_tfidf = tfidf.transform([keywords])
    # compute the cosine-similarity
    result = cosine_similarity(key_tfidf, tfidf_matrix)
    # sort the top n similar movies
    sim_key_movies = sorted(list(enumerate(result[0])), reverse=True, key=lambda x: x[1])
    # extract names from the dataframe and return the movie name
    rec = []
    for i in sim_key_movies[1: n+1]:
        rec.append(df.iloc[i[0]].title)
    return rec

def get_rec(movie):
    # get index from dataframe
    index = df[df['title'] == movie].index[0]
    # sort top n similar movies
    sim_movies = sorted(list(enumerate(cos_mat[index])), reverse=True, key=lambda x: x[1])
    # extracting names form dataframe and return the movie names
    rec = []
    for i in sim_movies[1: 6]:
        rec.append(df.iloc[i[0]].title)
    return rec

def fetch_poster(movies):
    ids = []
    posters = []
    for i in movies:
        ids.append(df[df.title==i]['id'].values[0])
        
    for i in ids:    
        url = f"https://api.themoviedb.org/3/movie/{i}?api_key=7430f7e0e63ce13129c70423d90a6114"
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        posters.append(full_path)
    return posters

# App Layout
st.image("images/applogo1.png")
st.title("Movie Finder")
posters = 0
movies = 0

with st.sidebar:
    st.image("images/app1.png", use_column_width=True)
    st.header("Get Recommendations by")
    search_type = st.radio("", ('Movie Title', 'Keywords'))
    st.header("Source Code")
    st.markdown("[GitHub Repository](https://github.com/QManiac162/projects-in-ipython-notebook/tree/main/project-5-movie-recommendor)")

# call functions based on selectbox
if search_type == 'Movie Title': 
    st.subheader("Select Movie")   
    movie_name = st.selectbox('', df.title)
    if st.button('Recommend'):
        with st.spinner('Wait for it...'):
            movies = get_rec(movie_name)
            posters = fetch_poster(movies)        
else:
    st.subheader('Enter Cast / Crew / Tags / Genre ')
    keyword = st.text_input('', 'Christopher Nolan')
    if st.button('Recommend'):
        with st.spinner('Wait for it...'):
            movies = get_keywords_rec(keyword, 6)
            posters = fetch_poster(movies)
              
# display posters       
if posters:
    col1, col2, col3, col4, col5 = st.columns(5, gap='medium')
    with col1:
        st.text(movies[0])
        st.image(posters[0])
    with col2:
        st.text(movies[1])
        st.image(posters[1])
    with col3:
        st.text(movies[2])
        st.image(posters[2])
    with col4:
        st.text(movies[3])
        st.image(posters[3])
    with col5:
        st.text(movies[4])
        st.image(posters[4])