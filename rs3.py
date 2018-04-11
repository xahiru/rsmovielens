import pandas as pd
import math
from math import sqrt
import numpy as np
import datetime
import statsmodels.formula.api as sm
import statsmodels.stats.diagnostic as sms

#https://github.com/pranzell/Recommender-Systems/blob/master/Recommendation%20Problem%20-%20Movie%20Data%20Set.ipynb


#    -- Scikit Learn --
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
##  Importing a package for Using Ridge and Lasso Analysis
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import scale
##  Splitting the Data Set into two Folds: Training and Testing using train_test_split()
from sklearn.cross_validation import train_test_split
##  Tool for calculating MSE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale


#    -- Packages for Visualization --
import seaborn as sns
##  Plotly
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.offline as plot
# plot.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
##  Scatter Plot
from pandas.tools.plotting import scatter_matrix
##  Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#   -- Packages for t-SNE Algorithm --
# from skdata.mnist.view import OfficialImageClassification
# from matplotlib import pyplot as plt
# from tsne import bh_sne

minibatch_size = 100


user_data = pd.read_table('u.data', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
print(user_data.shape)
user_user = pd.read_table('u.user', sep='|',names=['user id', 'age', 'gender', 'occupation', 'zip code'])

n_users = user_data['user id'].unique().shape[0]
n_items = user_data['item id'].unique().shape[0]

print("Number of users = " + str(n_users) + " | Number of movies = " + str(n_items))

train_data, test_data = train_test_split(user_data, random_state=4, train_size=.80, test_size=.20)

# (Training) User x Item Matrix --
train_data_matrix = np.zeros((n_users, n_items))
# print(train_data.shape)
# print('>>>>>>>>>>>>>>')

for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
    # print(line)

# print(train_data_matrix[30:33,:])
# (Testing) User x Item Matrix --
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
    # user - user similarity Matrix (943x943) :
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

# item -item similarity Matrix (1682x1682) :
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
print(user_similarity[30:33,:10])

# user - user similarity Matrix (943x943) :
cosine_user_similarity = 1 - user_similarity

# item -item similarity Matrix (1682x1682) :
cosine_item_similarity = 1 - item_similarity


# We can use 'cosine_similarity' method directly, but for missing values in the dataset it puts NAN!! 
# cosine_user_similarity = cosine_similarity(train_data_matrix[:,:], train_data_matrix)
# cosine_item_similarity = cosine_similarity(train_data_matrix.T[:,:], train_data_matrix.T)
# print(cosine_user_similarity)

def predict(ratings, similarity, type='user'):
    # print(ratings)
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
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
        print(trust_weights.shape)
        print(ratings_diff.shape)
        pred = mean_user_rating[:, np.newaxis] + trust_weights.dot(ratings_diff) / np.array([np.abs(trust_weights).sum(axis=1)]).T
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

import time
# start_time = time.time()


def gen_trust_matrix(ratings,similarity,n_users, predict='yes'):

    trust_matrix = np.zeros((n_users, n_users))
    pair_matrix = np.zeros((2, n_items))

    train_matrix = ratings.copy()
    ratings_matrix = train_matrix[:n_users,:]
    sim_matrix = similarity[:n_users,:n_users]

    mean_user_rating = ratings_matrix.mean(axis=1)
    ratings_diff = (ratings_matrix - mean_user_rating[:, np.newaxis])

    if predict == 'yes':
        prediction_matrix = predict(train_matrix.copy(),sim_matrix.copy(),type='user')
    else:
        prediction_matrix = predict

    for x in range(n_users):
        for y in range(n_users):
            if x != y:
                # print(train_data_mini_batch_1[x,:])
                pair_matrix[0] = ratings_matrix[x,:]
                pair_matrix[1] = ratings_matrix[y,:]
                sim = sim_matrix[x,y]
                pair_prediction = mean_user_rating[y] + ratings_diff[y] * sim
                # pair_prediction = predict(pair_matrix[1], sim, type='user')
                accurracy = abs(prediction_matrix[y] - pair_prediction)

                true_accuracy = sum(accurracy < 0.2)
                
                trust_matrix[x,y] = true_accuracy/accurracy.size
                # print(true_accuracy/accurracy.size)

    return trust_matrix
    
batch_size = n_users

trust_weights = gen_trust_matrix(train_data_matrix,cosine_user_similarity,batch_size, user_prediction)
        
trustbased_predictions = trust_predict(train_data_matrix[:batch_size,:], trust_weights, type='user')
print('Trust-based user CF RMSE: ' + str(rmse(trustbased_predictions, test_data_matrix[:batch_size,:])))


# print(trust_matrix_mini_batch_1)

# gx = np.random.randn(500)
# datax = [go.Histogram(x=gx)]

# py.plot(datax, filename='basic histogram')
# data = trust_matrix_mini_batch_1.flatten()


# data = [go.Histogram(x=data)]
# plotly.offline.plot({
#     "data": [Histogram(x=data, y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])],
#     "layout": Layout(title="p trust")
# })
# py.plot(data, filename='basic histogram')

# plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
# plt.show()


# np.savetxt('maximums.txt', trust_matrix_mini_batch_1)
# for idx, user in enumerate(train_data_matrix):
# #     # print(user)
# #     # p = user
#     for jdx, consumer in enumerate(train_data_matrix):

#         if idx != jdx:

#             if mat_method == 1:
#                 print("none")
#             #     temp_data = train_data_matrix.copy()
#             #     temp_data[jdx] = 0
#             #     # print(temp_data)
#             #     user_temp_similarity = pairwise_distances(temp_data, metric='cosine')
#             #     print('user_temp_similarity')
#             #     user_temp_similarity = 1 - user_temp_similarity
#             #     print(user_temp_similarity)
#             #     user_pp = predict(temp_data, user_temp_similarity)
#             #     print(pp)
#             #     print(user_pp)
#             #     print(abs(user_pp - pp)<0.2)
#             #     print(idx,jdx)
#             #     turst_matrix[idx,jdx] = 1
#             #     print(turst_matrix)
#             else:
#                 print('single vector method')
#             #     # mean = consumer.mean()
#                 # sim = pairwise_distances(user.reshape(1, -1), consumer.reshape(1, -1), metric='cosine')
#                 # print(sim)
#             #     # print(mean)
#             #     # print(sim.dot(mean))
#             #     print('consumer')
#             #     print(consumer)
#             #     # print('provider')
#             #     print('predict')
#             #     print(predict(consumer, sim, type='single'))
# # end_time = time.time()
# # print(start_time - end_time)


