# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 00:17:14 2022

@author: debna
"""
import pandas as pd
game_data = pd.read_csv("E:/Python Code  Datasets_Recommendation Engine/Day18-Recommendation Engine/game.csv")
game_data.shape
game_data.columns
game_data.head()
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf = TfidfVectorizer(stop_words="english")
game_data["rating"].isnull().sum()
tfidf_matrix = Tfidf.fit_transform(game_data.game)
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cos_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
game_data_index = pd.Series(game_data.index, index= game_data["userId"]).drop_duplicates()
game_data_index.head(10)
game_data_id = game_data_index[269]
game_data_id

def get_recommendations(UserId, topN):
    # topN = 10
    
    #getting the game index sing its userid
    game_data_id = game_data_index[UserId]
    
    # Getting the pair wise similarity score for all the anime's with that
    cosine_scores = list(enumerate(cos_sim_matrix[game_data_id]))
    
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse=True)
    
    cosine_scores_N = cosine_scores[0:topN+1]
    
    
    # Getting the game index 
    game_data_idx = [i[0] for i in cosine_scores_N]
    
    game_data_scores = [i[1] for i in cosine_scores_N]
    
    
    games_similar = pd.DataFrame(columns=["game", "rating"])
    
    games_similar["game"] = game_data.loc[game_data_idx, "game"]    
    
    games_similar["rating"] = game_data_scores
    
    games_similar.reset_index(inplace = True) 
    
    #games_similar.drop(["game"], axis=1, inplace=True)
    print(games_similar)


get_recommendations(255, topN=10)

game_data_index[285]
























