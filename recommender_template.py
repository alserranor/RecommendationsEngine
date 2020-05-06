import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments
import recommender_functions as rf

class Recommender():
    '''
    This Recommender predicts ratings of movies based on FunkSVD. It uses either FunkSVD or Knowledge Based Recommendations (the highest-ranked) to provide ratings to users depending on their novelty. Also, if a movie is input, other movies will be recommended using Content Based Recommendations.
    '''
    def __init__(self, ):
        '''
        No required attributes are needed to instantiate the Recommender. They will be provided as arguments of the model in the fit function.
        '''



    def fit(self, reviews_path, movies_path, latent_features = 12, learning_rate = 0.0001, iters = 100):
        '''
        Fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        INPUT
        - reviews_path: Path where the reviews data is located
        - movies_path: Path where the movies data is located
        - latent_features: (int) Number of latent features to be included in the model
        - learning_rate: (float) Learning rate of the model
        - iters: (int) Iterations to be performed in the fitting
        
        OUTPUT
        None. Stores in the class the following attributes:
        - n_users: Number of users
        - n_movies: Number of movies
        - num_ratings: Number of ratings
        - user_item_mat: (np array) A user by item numpy array with ratings as values 
        - reviews: Dataframe with columns 'user_id', 'movie_id', 'rating', 'timestamp'
        - movies: Dataframe with movies information
        - latent_features, learning_rate and iters described in INPUT section
        
        - user_mat: (np array) A user by latent_features matrix part of SVD model
        - movie_mat: (np array) A latent_features by movies matrix part of SVD model
        - ranked_movies: Dataframe with ranked movies used for knowledge-based model
        '''
        # Read data and store as dataframes
        self.reviews = pd.read_csv(reviews_path)
        self.movies = pd.read_csv(movies_path)
        
        # Create user-movie matrix
        user_item = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.user_item_mat = np.array(self.user_item_df)
        
        # Store more inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters
        
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))
        
        # Initialize the user and movie matrices with random values
        user_mat = np.random.rand(n_users, latent_features)
        movie_mat = np.random.rand(latent_features, n_movies)
    
        # Initialize sse at 0 for first iteration
        sse_accum = 0

        # Header for running results
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")

        # For each iteration
        for iteration in range(self.iters):

            # Update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # If the rating exists
                    if self.user_item_mat[i][j] > 0:

                        # Compute the error
                        diff = self.user_item_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_mat[i, k] += self.learning_rate*(2*diff*movie_mat[k, j])
                            movie_mat[k, j] += self.learning_rate*(2*diff*user_mat[i, k])
                            
        # Print results for iteration
        print("%d \t\t %f" % (iteration + 1, sse_accum/num_ratings))
        
        # SVD-based fit
        # Save user_mat and movie_mat for safe keeping
        self.user_mat = user_mat
        self.movie_mat = movie_mat
        
        # Knowledge-based fit
        self.ranked_movies = rf.create_ranked_df(movies)
        

    def predict_rating(self, ):
        '''
        makes predictions of a rating for a user on a movie-user combo
        '''

    def make_recs(self,):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        '''


if __name__ == '__main__':
    # test different parts to make sure it works
