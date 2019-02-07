
# coding: utf-8

# In[ ]:


import sys
import gensim, logging
import pandas as pd
import os
import numpy as np
import gensim.downloader as api

model = api.load("word2vec-ruscorpora-300")

DATA_PATH = './data//added_data'

File_names_by_cat={'sport':['double-sport.csv', 'pro-bike.csv'],'food':['abrikos.csv', 'auchan.csv', 'foodband.csv'],'tablets':['iherb_food.csv', 'gold-standart.csv', 'iherb_sport.csv']}

def Search_column_content(data,Search_content,file_name,noun_bonus=1,adj_bonus=1):
    """
    Content - pd.Series с контентом.
    Search_content - List с контентом(!) поиска [горный_NOUN,горный_ADJ,....]  
    
    Выход - по датафрейм (Similarity | id(просто чтобы не просрать) | магазин.csv)
    
    """
    new_data = data.copy()
    Similarity=[]
    for row in new_data.words:
        #row строка вида велосипед_NOUN,велосипед_ADJ, и тд
        content_words=row.split(',')
        Final_rating=[]

        for word in Search_content:
            coef=1
            if word.split('_')[1]=='NOUN':
                coef=noun_bonus
            else:
                coef=adj_bonus
            ratings=[]
            for cont in content_words:
                #print(word,cont)
                
                ratings.append(model.similarity(word, cont)*coef)
            Final_rating.append(np.mean(ratings))
        Similarity.append(np.mean(Final_rating))
    
    new_data['Similarity'] = np.array(Similarity)
    new_data['Name'] = np.full(new_data.shape[0], file_name)
    return new_data

def Total_search(Category,Search_content): 
    """
    File_names = File_names_by_cat[Category]
    Search_content - Обработанная строка поиска в виде List
    
    Выход - отсортированный по Similarity descending полный Dataframe вида (Similarity | id(в оригинальном файле) | магазин.csv)
    
    """
    CUR_PATH = os.getcwd()
    os.chdir(DATA_PATH)
    File_names = File_names_by_cat[Category]
    All_similarities=[]
    for file in File_names:
        current_file = pd.read_csv(file)
        All_similarities.append(Search_column_content(current_file,Search_content,file,2,0.75))
    pre_data = pd.concat(All_similarities, axis=0).sort_values(by=['Similarity'], ascending=False)
    os.chdir(CUR_PATH)
    return pd.DataFrame(pre_data.values, index=range(pre_data.shape[0]), columns=pre_data.columns)

def category(quest):
    cat = np.array([0., 0., 0.])
    cats = {0: 'food', 1: 'sport', 2: 'tablets'}
    for word in quest:
        cat[0] += model.similarity(word, 'еда_NOUN')
        cat[1] += model.similarity(word, 'спорт_NOUN')
        cat[2] += model.similarity(word, 'таблетка_NOUN')
    return cats[np.argmax(cat)]