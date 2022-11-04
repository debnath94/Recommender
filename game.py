# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:11:25 2022

@author: debna
"""
import pandas as pd
data = pd. read_csv("E:/Python Code  Datasets_Recommendation Engine/Day18-Recommendation Engine/game.csv")
data. shape
data.game
from sklearn. feature_extraction. text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words = "english")

data["game"]. isnull().sum()
data["userId"]. isnull().sum()
data["rating"]. isnull().sum()
tfidf_matrix = tfidf. fit_transform(data.game)
tfidf_matrix.shape

from sklearn. metrics. pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
data_index = pd. Series(data. index, index= data['game']). drop_duplicates()
data.groupby('game') ['rating']. mean().sort_values(ascending= False).head(10)
data_id = data_index["Marvel Pinball"]

def get_recommendations(Name, topN):
    topN = 10
    
    data_id = data_index[Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[data_id]))
    cosine_scores = sorted(cosine_scores, key =lambda x:x[1], reverse = True)
    cosine_scores_n = cosine_scores[0: topN+1]
    data_idx = [i[0] for i in cosine_scores_n]
    data_scores = [i[1] for i in cosine_scores_n]
    data_similar_show = pd.DataFrame(columns= ["name", "score"])
    data_similar_show["name"] = data.loc[data_idx, "game"]
    data_similar_show["score"] = data_scores
    data_similar_show. reset_index(inplace = True)
    print(data_similar_show)














