import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn

def age_map(x: int) -> int:
	if x < 10: return 0
	elif 10 <= x < 20: return 1
	elif 20 <= x < 30: return 2
	elif 30 <= x < 40: return 3
	elif 40 <= x < 50: return 4
	elif 50 <= x < 60: return 5
	elif 60 <= x < 70: return 6
	elif 70 <= x < 100: return 7
	else: return 8
        
def change_count(x):
    if x < 2: return 0 
    elif 2 <= x < 3: return 1
    elif 3 <= x < 5: return 2
    elif 5 <= x < 7: return 3 
    elif 7 <= x < 10: return 4
    elif 10 <= x < 30: return 5 
    elif 30 <= x < 100: return 6
    else: return 7

def cat_data_load(args):
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')   
     
    users['age'] = users['age'].apply(age_map)
    
    ids = pd.concat([train['user_id'], submission['user_id']]).unique()
    isbns = pd.concat([train['isbn'], submission['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    users_ = users.copy()
    books_ = books.copy()
    books_ = books_.drop(['img_url','img_path'],axis=1)
    
    train = pd.merge(train, users_, on='user_id', how='left')
    submission = pd.merge(submission, users_, on='user_id', how='left')
    test = pd.merge(test, users_, on='user_id', how='left')

    train = pd.merge(train, books_, on='isbn', how='left')
    submission = pd.merge(submission, books_, on='isbn', how='left')
    test = pd.merge(test, books_, on='isbn', how='left')

    train['user_id'] = train['user_id'].map(user2idx)
    submission['user_id'] = submission['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    submission['isbn'] = submission['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    
    train['year_of_publication'] = train['year_of_publication'].astype(int)
    submission['year_of_publication'] = submission['year_of_publication'].astype(int)
    test['year_of_publication'] = test['year_of_publication'].astype(int)
    
    train = train.fillna('-1')
    submission = submission.fillna('-1')
    test = test.fillna('-1')

    sub_rating = submission['rating']
    submission = submission.drop(columns='rating')
    submission['rating'] = sub_rating
    
    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'users':users,
            'books':books,
            'sub':submission,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }
    
    return data

def cat_data_split(args,data):   
    X_train, X_valid, y_train, y_valid =  train_test_split(data['train'].drop(['rating'],axis=1), 
                                                           data['train']['rating'], 
                                                           test_size=args.test_size, 
                                                           random_state=args.seed,
                                                           shuffle=True
                                                           )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data    

