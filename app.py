import streamlit as st
#import pickle
from data import new_df,similarity
import pandas as pd
import requests
st.set_page_config(page_title='Movie Recommender App',page_icon='assets/logo.png',layout='wide')
st.title('Movie Recommender App')
def fetch_poster(movie_id):
    data = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=925f5abb87a3e487dda2cdd5babea3b8&language=en-US').json()
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:16]
    recommended_movies = []
    recommended_movies_poster = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_poster(movie_id=movie_id))
    return recommended_movies,recommended_movies_poster
movies = pd.DataFrame(new_df.to_dict())
option = st.selectbox('Select a movie',movies['title'].values)
if st.button('Recommend'):
    names,posters = recommend(option)
    col1,col2,col3,col4,col5,col6,col7 = st.columns(7)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
    with col6:
        st.text(names[5])
        st.image(posters[5])
    with col7:
        st.text(names[6])
        st.image(posters[6])
    col8,col9,col10,col11,col12,col13,col14 = st.columns(7)
    with col8:
        st.text(names[7])
        st.image(posters[7])
    with col9:
        st.text(names[8])
        st.image(posters[8])
    with col10:
        st.text(names[9])
        st.image(posters[9])
    with col11:
        st.text(names[10])
        st.image(posters[10])
    with col12:
        st.text(names[11])
        st.image(posters[11])
    with col13:
        st.text(names[12])
        st.image(posters[12])
    with col14:
        st.text(names[13])
        st.image(posters[13])