import pandas as pd
import math
from math import sqrt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
# warnings.filterwarnings("ignore",category =RuntimeWarning)
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

batch_size = 20

user_data = pd.read_table('u.data', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
# print(user_data.shape)
user_user = pd.read_table('u.user', sep='|',names=['user id', 'age', 'gender', 'occupation', 'zip code'])

n_users = user_data['user id'].unique().shape[0]
n_items = user_data['item id'].unique().shape[0]
user_data2 = user_data.copy()

print("Number of users = " + str(n_users) + " | Number of movies = " + str(n_items))

train_data, test_data = train_test_split(user_data, random_state=4, train_size=.80, test_size=.20)

# print('test_data')
# print(test_data.shape)
# (Training) User x Item Matrix --
train_data_matrix = np.zeros((n_users, n_items))
# print(train_data.shape)
# print('>>>>>>>>>>>>>>')

main_data_matrix = np.zeros((n_users, n_items))
np.savetxt('main_data_matrix.csv', main_data_matrix,delimiter=",")


for line in user_data2.itertuples():
    main_data_matrix[line[1]-1, line[2]-1] = line[3]


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
print('user_similarity')
# print(user_similarity[30:33,:10])

# user - user similarity Matrix (943x943) :
cosine_user_similarity = 1 - user_similarity

# item -item similarity Matrix (1682x1682) :
cosine_item_similarity = 1 - item_similarity
# print(user_similarity[:2,:])

cosine_item_similarity2 = cosine_item_similarity.copy()
train_data_matrix2 = train_data_matrix.copy()

# np.savetxt('train_data_matrix.csv', train_data_matrix,delimiter=",")
# np.savetxt('cosine_item_similarity.csv', cosine_item_similarity,delimiter=",")
# np.savetxt('cosine_user_similarity.csv', cosine_user_similarity,delimiter=",")
# We can use 'cosine_similarity' method directly, but for missing values in the dataset it puts NAN!! 
# cosine_user_similarity = cosine_similarity(train_data_matrix[:,:], train_data_matrix)
# cosine_item_similarity = cosine_similarity(train_data_matrix.T[:,:], train_data_matrix.T)
# print('cosine_user_similarity')
# print(cosine_user_similarity)

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

# import time
# start_time = time.time()


def gen_trust_matrix_leave_one_out(ratings,similarity,batch_size,prediction, ptype):
    trust_matrix = np.zeros((batch_size, batch_size))
    dim = 1
    profiles_sep_option = 'no'
    percentage = 80 #80 20    
    for x in range(batch_size):
        ratings_new = ratings.copy()
        similarity_new = similarity.copy()
        if ptype == 'item':
            ratings_new.T
            dim = 0
        ratings_new[x] = 0
        similarity_new[x] = 0
        # print(ratings_new[x])
        # print(similarity.shape)
        if ptype == 'item':
            ratings_new.T
        xhat_predict = predict(ratings_new, similarity_new,ptype)
        # print('xhat_predict')
        # print(xhat_predict.shape)
        # print(prediction.shape)
        # print((xhat_predict).sum(1))
        # if ptype =='user':
        #     predic_diff = abs(prediction - xhat_predict)
        # else:
        #     
        predic_diff = abs(prediction - xhat_predict)

        np.where(np.isnan(predic_diff), predic_diff, 0)
        # predic_diff = predic_diff[~np.isnan(predic_diff)]
        
        if ((x/batch_size)*100) > 80:
            print('append zeros to consumers')
        # trust_matrix.append(((predic_diff <0.001).sum(dim))/predic_diff.shape[dim])
        # np.allclose(x * x, y)
        # print(predic_diff[:,:5] <0.2)


        trust_row = ((predic_diff <0.2).sum(dim))/predic_diff.shape[dim]
        # trust_row = trust_row[~np.isnan(trust_row)]
        # trust_row = trust_row[~np.isfinite(trust_row)]
        np.where(np.isinf(trust_row), trust_row, 0)
        # if np.any(np.isfinite(trust_row <0.001)):
        #     print('inf')
        # np.any(np.isnan(trust_row))
        trust_matrix[x] = trust_row
        # print((predic_diff <0.001).sum(dim))

    return trust_matrix

def gen_trust_matrix(ratings,similarity,batch_size, prediction_matrix):

    trust_matrix = np.zeros((batch_size, batch_size))
    pair_matrix = np.zeros((2, ratings.shape[1]))

    train_matrix = ratings.copy()
    ratings_matrix = train_matrix[:batch_size,:]
    sim_matrix = similarity[:batch_size,:batch_size]

    mean_user_rating = ratings_matrix.mean(axis=1)
    ratings_diff = (ratings_matrix - mean_user_rating[:, np.newaxis])


    for x in range(batch_size):
        for y in range(batch_size):
            if x != y:
                # print(train_data_mini_batch_1[x,:])
                pair_matrix[0] = ratings_matrix[x,:]
                pair_matrix[1] = ratings_matrix[y,:]
                sim = sim_matrix[x,y]
                pair_prediction = mean_user_rating[y] + ratings_diff[y] * sim
                # pair_prediction = predict(pair_matrix[1], sim, type='user')
                # print("Pair prediction")
                # print(pair_prediction)
                # print("original prediction")
                # print(prediction_matrix[y])
                accurracy = abs(prediction_matrix[y] - pair_prediction)

                # true_accuracy = sum(accurracy < 0.8) worked well for batch size 10
                true_accuracy = sum(accurracy < 0.8)
                
                trust_matrix[x,y] = true_accuracy/accurracy.size
                # print(true_accuracy/accurracy.size)

    return trust_matrix

trust_weights = gen_trust_matrix(train_data_matrix,cosine_user_similarity,batch_size, user_prediction)
trust_item_weights = gen_trust_matrix(train_data_matrix.T,cosine_item_similarity,batch_size, item_prediction.T)

# print('trust_weights')
# print(trust_weights)

trust_weights_user_harmonic_mean = (2*(trust_weights*cosine_user_similarity[:batch_size,:batch_size]))/(trust_weights + cosine_user_similarity[:batch_size,:batch_size])
trust_weights_item_harmonic_mean = (2*(trust_weights*cosine_item_similarity[:batch_size,:batch_size]))/(trust_weights + cosine_item_similarity[:batch_size,:batch_size])


trustbased_user_predictions = predict(train_data_matrix[:batch_size,:], trust_weights_user_harmonic_mean, type='user')
trustbased_item_predictions = trust_predict(train_data_matrix[:batch_size,:].T, trust_weights_item_harmonic_mean, type='item')


print('Trust-based user CF RMSE: ' + str(rmse(trustbased_user_predictions, test_data_matrix[:batch_size,:])))
print('Trust-based item CF RMSE: ' + str(rmse(trustbased_item_predictions.T, test_data_matrix[:batch_size,:])))

cos_i2 = cosine_item_similarity2[:batch_size,:batch_size]
train2 = train_data_matrix2[:,:batch_size]
print(cos_i2.shape,train2.shape)
print('testing cos2 and train2')
predict(train2,cos_i2,type='item')
# batch_size = 15
tw = gen_trust_matrix_leave_one_out(train_data_matrix[:batch_size,:], cosine_user_similarity[:batch_size,:batch_size],batch_size, user_prediction[:batch_size,:],ptype='user')
tw_item = gen_trust_matrix_leave_one_out(train_data_matrix[:,:batch_size], cosine_item_similarity[:batch_size,:batch_size],batch_size, item_prediction[:,:batch_size],ptype='item')

# tw_item = tw_item.as_matrix().astype(np.float)
# X = X.as_matrix().astype(np.float)
# print(tw.shape)
# print(trust_matrix_mini_batch_1)
# np.savetxt('tw.csv', tw,delimiter=",")
def get_harmonic_mean(trust_matrix, cos_similarity,batch_size):
    return (2*(trust_matrix*cos_similarity[:batch_size,:batch_size]))/(trust_matrix + cos_similarity[:batch_size,:batch_size])

tw_user_harmonic_mean = (2*(tw*cosine_user_similarity[:batch_size,:batch_size]))/(tw + cosine_user_similarity[:batch_size,:batch_size])
# print(tw_item)
# np.array(tw_item).reshape(train2.shape)


tw_item_harmonic_mean = (2*(tw_item*cosine_item_similarity[:batch_size,:batch_size]))/(tw_item + cosine_item_similarity[:batch_size,:batch_size])

tw_user_predictions = predict(train_data_matrix[:batch_size,:], tw_user_harmonic_mean, type='user')
tw_item_predictions = predict(train_data_matrix[:,:batch_size], tw_item_harmonic_mean, type='item')

# print(np.any(np.isnan(tw_item_predictions)))
print('Tw-based user CF RMSE: ' + str(rmse(tw_user_predictions, test_data_matrix[:batch_size,:])))
print('Tw-Items-based user CF RMSE: ' + str(rmse(tw_item_predictions, test_data_matrix[:,:batch_size])))

# test_batch_sizes = [50,100,200,300,400,500,600,700,800,900,n_users]



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


