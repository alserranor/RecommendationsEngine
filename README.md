# Recommendations Engine

A recommendations engine to propose new movies to users, based on MovieTweetings data.

## Contents

 - recommender.py: Recommender class that uses FunkSVD algorithm to make predictions of exact movie ratings. It uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.
 - recommender_functions.py: General set of functions to use while creating a recommendations engine.
 - recommender_template.py: Class template for a recommendations engine.
 - movies_clean.csv: A movie ratings dataset based on [MovieTweetings](http://crowdrec2013.noahlab.com.hk/papers/crowdrec2013_Dooms.pdf) data. See this [link](https://github.com/sidooms/MovieTweetings) for more information.
 - train_data.csv: Training dataset for the Recommender.
 - LICENSE: MIT license for the code of the repository
 - README.md: This file.

## License
The code used in this repository is licensed under a MIT license included in LICENSE.txt

## Acknowledgements

Must give credit to Udacity for providing the dataset.
