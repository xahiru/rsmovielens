import pandas as pd
import math
from math import sqrt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


#https://github.com/pranzell/Recommender-Systems/blob/master/Recommendation%20Problem%20-%20Movie%20Data%20Set.ipynb


#    -- Scikit Learn --
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cross_validation import train_test_split
##  Tool for calculating MSE
from sklearn.metrics import mean_squared_error
 

batch_size = 943

user_data = pd.read_table('u.data', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
user_user = pd.read_table('u.user', sep='|',names=['user id', 'age', 'gender', 'occupation', 'zip code'])

n_users = user_data['user id'].unique().shape[0]
n_items = user_data['item id'].unique().shape[0]
user_data2 = user_data.copy()

print("Number of users = " + str(n_users) + " | Number of movies = " + str(n_items))

train_data, test_data = train_test_split(user_data, random_state=4, train_size=.80, test_size=.20)



# (Training) User x Item Matrix --
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
    # print(line)

# (Testing) User x Item Matrix --
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

# print(np.count_nonzero(train_data_matrix==0))
    # user - user similarity Matrix (943x943) :
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

# item -item similarity Matrix (1682x1682) :
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

# user - user similarity Matrix (943x943) :
cosine_user_similarity = 1 - user_similarity

# item -item similarity Matrix (1682x1682) :
cosine_item_similarity = 1 - item_similarity


def predict(ratings, similarity, type='user'):
    # print(ratings)
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        # print(ratings.shape)
        # print(similarity.shape)
        # print(np.array([np.abs(similarity).sum(axis=1)]).shape)
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)]) 
    elif type == 'single':
        mean = ratings.mean()
        ratings_diff = ratings - mean
        pred = pred = mean + ratings_diff * similarity #/ np.array([np.abs(similarity).sum(axis=1)]).T    
    return pred


def trust_predict(ratings, trust_weights, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        # print(trust_weights.shape)
        # print(ratings_diff.shape)
        pred = mean_user_rating[:, np.newaxis] + trust_weights.dot(ratings_diff) / np.array([np.abs(trust_weights).sum(axis=1)]).T
    elif type == 'item':
        # print(ratings.shape)
        # print(trust_weights.shape)
        # print(np.array([np.abs(trust_weights).sum(axis=1)]).T.shape)
        pred = ratings.dot(trust_weights) / np.array([np.abs(trust_weights).sum(axis=1)])
        # pred = trust_weights.dot(ratings) / np.array([np.abs(trust_weights).sum(axis=1)]).T
    return pred

# user - user CF:
user_prediction = predict(train_data_matrix, cosine_user_similarity, type='user')
# item - item CF:
item_prediction = predict(train_data_matrix, cosine_item_similarity, type='item')

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

def gen_trust_matrix_leave_one_out(ratings,similarity,batch_size,prediction, ptype):
    trust_matrix = np.zeros((batch_size, batch_size))
    dim = 1
    profiles_sep_option = 'no'
    # percentage = 80 #80 20    
    for x in range(batch_size):
        ratings_new = ratings.copy()
        similarity_new = similarity.copy()
        if ptype == 'item':
            ratings_new.T
            dim = 0
        ratings_new[x] = 0
        similarity_new[x] = 0

        if ptype == 'item':
            ratings_new.T

        xhat_predict = predict(ratings_new, similarity_new,ptype)
        
        xhat_predict[np.isnan(xhat_predict)] = 0
        # print(np.any(np.isnan(xhat_predict)))

        predic_diff = abs(prediction - xhat_predict)

        # np.where(np.isnan(predic_diff), predic_diff, 0)
        
        
        # if ((x/batch_size)*100) > 80:
        #     print('append zeros to consumers')
        

        trust_row = ((predic_diff <10).sum(dim))/predic_diff.shape[dim]
        
        # np.where(np.isinf(trust_row), trust_row, 0)
        # np.where(np.isnan(trust_row), trust_row, 0)
        
        trust_matrix[x] = trust_row
    
    return trust_matrix

def get_harmonic_mean(trust_matrix, cos_similarity):
    return (2*(trust_matrix*cos_similarity))/(trust_matrix + cos_similarity)

def get_trust_prediction(train_data_matrix,cos_similarity,batch_size, prediction, ptype):
    trust_matrix = gen_trust_matrix_leave_one_out(train_data_matrix, cos_similarity,batch_size, prediction, ptype)
    trust_weights = get_harmonic_mean(trust_matrix,cos_similarity)
    
    return predict(train_data_matrix,trust_weights,ptype)


tw_user_predictions = get_trust_prediction(train_data_matrix[:batch_size,:], cosine_user_similarity[:batch_size,:batch_size],batch_size,user_prediction[:batch_size,:],ptype='user')
tw_item_predictions = get_trust_prediction(train_data_matrix[:,:batch_size], cosine_item_similarity[:batch_size,:batch_size],batch_size,item_prediction[:,:batch_size], ptype='item')


print('Tw-based user CF RMSE: ' + str(rmse(tw_user_predictions, test_data_matrix[:batch_size,:])))
print('Tw-Items-based user CF RMSE: ' + str(rmse(tw_item_predictions, test_data_matrix[:,:batch_size])))


# ptype = 'item'

# if ptype == 'item':
#     batch_count = int(n_items/batch_size)
#     ini = 0
#     for x in range(batch_count):
#         t = train_data_matrix[:,ini:ini+batch_size]
#         # print(t)
#         sim = cosine_item_similarity[ini:ini+batch_size,ini:ini+batch_size]
#         p = item_prediction[:,ini:ini+batch_size]
#         test = test_data_matrix[:,ini:ini+batch_size]

#         tp = get_trust_prediction(t,sim,batch_size,p,ptype)
#         # print(np.any(np.isnan(tp)))
#         # print(str(rmse(tp, test)))
#         # print(tp)
#         # test_data_matrix[:batch_size,:]

#         # print('similarity between '+ str(ini) +', '+ str(ini+batch_size))
#         # print(sim)
#         ini += batch_size
#         # print(ini)
# else:
#     batch_count = int(n_users/batch_size)
#     ini = 0
#     for x in range(batch_count):
        
#         t = train_data_matrix[:ini,ini+batch_size]

#         sim = cosine_user_similarity[ini:ini+batch_size,ini:ini+batch_size]
#         print(t)
#         ini += batch_size


