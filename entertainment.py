# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:15:26 2022

@author: debna
"""
import pandas as pd
data = pd. read_csv("E:/Python Code  Datasets_Recommendation Engine/Day18-Recommendation Engine/Entertainment.csv")
data.Category
from sklearn. feature_extraction. text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words = "english")

data["Category"]. isnull(). sum()
tfidf_matrix = tfidf. fit_transform(data.Category)
tfidf_matrix. shape

from sklearn. metrics. pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
data_index = pd. Series(data. index, index = data['Titles']). drop_duplicates()
movie_id = data_index["Jumanji (1995)"]
movie_id

def get_recommendations(Name,topN):
    topN = 10
    movie_id = data_index[Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[movie_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    data_idx = [i[0] for i in cosine_scores_N]
    data_scores = [[i] for i in cosine_scores_N]
    data_similar_show = pd.DataFrame(columns=["name", "Score"])
    data_similar_show["name"] = data.loc[data_idx, "Titles"]
    data_similar_show["Score"] = data_scores
    data_similar_show. reset_index(inplace = True)
    print(data_similar_show)

data_index["Jumanji (1995)"]    
get_recommendations('Jumanji (1995)', topN=10)










