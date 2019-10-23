import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#getting data from lightfm datasets
data = fetch_movielens(min_rating=4.0)

#print data movielens divide data into training and testing
print(repr(data['train']))
print(repr(data['test']))

#create model
#using the loss function warp = weighted approximate rank pairwise
model = LightFM(loss='warp')
#train model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(mdoel,data,user_ids):

    #number of users and movies in training data
    _, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies user already liked
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies they will like
        scores = model.predict(user_id,np.arange(n_items))
        #rank the movies in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #printing the result
        print("User %s" %  user_id)
        print("      Known positives:")

        for x in known_positives[:3]:
            print("              %s" %x)

        print("      Recommended:")

        for x in top_items[:3]:
            print("              %s" %x)

#calling the model and giving random user id
sample_recommendation(model , data , [3,25,420])
