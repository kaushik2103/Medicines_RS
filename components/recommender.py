# This file finds similar medicines using KNN.
# We will pass disease name on that basis it will recommend the medicines.

# Importing libraries
import pandas as pd
import random

# This is our recommender class that consist of recommend_medicines method who recommend medicines using KNN based on drug content similarity.
class Recommender:
    def __init__(self, dataset, tfidf_matrix, knn):
        self.dataset = dataset
        self.tfidf_matrix = tfidf_matrix
        self.knn = knn

    def recommend_medicines(self, disease_name, num_meds = 5):
        disease_meds = self.dataset[self.dataset['disease_name'] == disease_name]
        if disease_meds.empty:
            return 'No medicine found'

        ref_index = random.choice(disease_meds.index)
        distances, indices = self.knn.kneighbors(self.tfidf_matrix[ref_index], n_neighbors=num_meds + 1)
        similar_meds = indices.flatten()[1:num_meds + 1]

        return self.dataset.iloc[similar_meds][['med_name', 'final_price', 'drug_manufacturer', 'prescription_required',
            'drug_manufacturer_origin', 'drug_content', 'img_urls']]
