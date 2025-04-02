# This file handles loading and training TF-IDF and KNN Model.

# Importing libraries
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# This is our ModelLoader class consist of load_or_train_models method.
# Here we loads saved model or trains new one if they don't exist.

class ModelLoader:
    def __init__(self, data_path='datasets/medicines_cleaned.csv'):
        self.data_path = data_path
        self.dataset = pd.read_csv(self.data_path)
        self.dataset['drug_content'] = self.dataset['drug_content'].fillna('').astype(str)

        # Path for saved models
        self.TFIDF_PATH = 'utils/tfidf_matrix.npz'
        self.VECTORIZER_PATH = 'utils/tfidf_vectorizer.pkl'
        self.KNN_PATH = 'utils/knn_model.pkl'

    def load_or_train_models(self):
        if os.path.exists(self.TFIDF_PATH):
            self.tfidf_matrix = load_npz(self.TFIDF_PATH)
            with open(self.VECTORIZER_PATH, 'rb') as file:
                self.tfidf_vectorizer = pickle.load(file)
        else:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=6000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.dataset['drug_content'])
            save_npz(self.TFIDF_PATH, self.tfidf_matrix)
            with open(self.VECTORIZER_PATH, 'wb') as file:
                pickle.dump(self.tfidf_vectorizer, file)

        if os.path.exists(self.KNN_PATH):
            with open(self.KNN_PATH, 'rb') as file:
                self.knn = pickle.load(file)
        else:
            self.knn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(self.tfidf_matrix)
            with open(self.KNN_PATH, 'wb') as file:
                pickle.dump(self.knn, file)

        return self.dataset, self.tfidf_matrix, self.knn