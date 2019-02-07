
# coding: utf-8

# In[22]:


import json
import pandas as pd


# In[100]:


def Make_json(data):
    """
    data - pd.Dataframe['Название','Описание','Цена','NAME','WEB','CASH_BACK','TRANCHE_STMT_COUNT','OFFER_TYPE','ADVERT_TEXT']
    output=json[5]
    """
    Output=[]
    for i in range(5):
        
        single_answer={}
        
        offer={}
        offer['offer']=data['NAME'].iloc[i]
        offer['web']=data['WEB'][i]
        offer['cashback']=data['CASH_BACK'].iloc[i]
        offer['period']=data['TRANCHE_STMT_COUNT'].iloc[i]
        offer['offer_type']=data['OFFER_TYPE'].iloc[i]
        offer['advert_text']=data['ADVERT_TEXT'].iloc[i]
       
        product={}
        product=[{'Item':data['Название'].iloc[i],'Attributes':data['Описание'].iloc[i],'Price':data['Цена'].iloc[i]}]  
        
        single_answer['offer']=offer
        single_answer['product']=product
        #print(single_answer)
        Output.append(single_answer)
        
        
    print(Output[4])
        
    with open('result/data.json', 'w+', encoding='utf-16') as outfile:
        json.dump(Output, outfile, ensure_ascii=False, indent=4)
        

