import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")

def load_ratings():
    return pd.read_csv("ratings.csv")

def load_course_sims():
    return pd.read_csv("sim.csv")

def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_bow():
    return pd.read_csv("courses_bows.csv")

def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id

# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict

def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

# Model training
def train(model_name, params):
    if model_name == models[0]:
        # Course Similarity model doesn't require training
        pass
    elif model_name == models[1]:
        # User Profile model
        # This could involve creating user embeddings based on their course history
        pass
    elif model_name == models[2]:
        # Clustering model
        ratings_df = load_ratings()
        kmeans = KMeans(n_clusters=params['cluster_no'])
        kmeans.fit(ratings_df[['user', 'item', 'rating']])
        # Save the model for later use
    elif model_name == models[3]:
        # Clustering with PCA
        ratings_df = load_ratings()
        pca = PCA(n_components=params['n_components'])
        reduced_data = pca.fit_transform(ratings_df[['user', 'item', 'rating']])
        kmeans = KMeans(n_clusters=params['cluster_no'])
        kmeans.fit(reduced_data)
        # Save both PCA and KMeans models
    elif model_name == models[4]:
        # KNN model
        ratings_df = load_ratings()
        knn = NearestNeighbors(n_neighbors=params['n_neighbors'])
        knn.fit(ratings_df[['user', 'item', 'rating']])
        # Save the KNN model
    elif model_name == models[5]:
        # NMF model
        ratings_df = load_ratings()
        nmf = NMF(n_components=params['n_components'])
        nmf.fit(ratings_df.pivot(index='user', columns='item', values='rating').fillna(0))
        # Save the NMF model
    elif model_name == models[6]:
        # Neural Network model
        ratings_df = load_ratings()
        model = Sequential()
        model.add(Embedding(input_dim=ratings_df['user'].max() + 1, output_dim=params['neurons_per_layer']))
        model.add(Flatten())
        for _ in range(params['hidden_layers']):
            model.add(Dense(params['neurons_per_layer'], activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(ratings_df['user'], ratings_df['rating'], epochs=10)
        # Save the neural network model
    elif model_name == models[7]:
        # Regression with Embedding Features
        ratings_df = load_ratings()
        embedding = Embedding(input_dim=ratings_df['user'].max() + 1, output_dim=params['embedding_size'])
        user_embeddings = embedding(ratings_df['user']).numpy()
        regression = LinearRegression()
        regression.fit(user_embeddings, ratings_df['rating'])
        # Save both embedding and regression models
    elif model_name == models[8]:
        # Classification with Embedding Features
        ratings_df = load_ratings()
        embedding = Embedding(input_dim=ratings_df['user'].max() + 1, output_dim=params['embedding_size'])
        user_embeddings = embedding(ratings_df['user']).numpy()
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(user_embeddings, pd.cut(ratings_df['rating'], bins=params['n_classes'], labels=range(params['n_classes'])))
        # Save both embedding and classifier models

# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        if model_name == models[0]:
            # Course Similarity model (existing code)
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        elif model_name in models[1:]:
            # For other models, we would load the trained model and make predictions
            # This is a placeholder and should be implemented based on how models are saved
            pass

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df