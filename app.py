# This file connects everything togather into a UI.
# Here we build a pipline that connect our two components.

# Importing libraries
import streamlit as st
import pandas as pd
from components.model_loader import ModelLoader
from components.recommender import Recommender

st.title("Medicine Recommender System")
st.write("Enter a disease name to get medicine recommendations.")

# Load Model
model_loader = ModelLoader()
dataset, tfidf_matrix, knn = model_loader.load_or_train_models()

# Create an instance of the recommender
recommender = Recommender(dataset, tfidf_matrix, knn)

# User input for disease name
disease_name = st.text_input("Enter Disease Name:", "").capitalize()

if disease_name:
    recommendations = recommender.recommend_medicines(disease_name)
    if isinstance(recommendations, str):
        st.write(recommendations)

    else:
        st.success(f"Recommended Medicines for {disease_name}")
        for _, row in recommendations.iterrows():
            with st.expander(f"{row['med_name']} - {row['final_price']}"):
                st.write(f"**Manufacturer:** {row['drug_manufacturer']} ({row['drug_manufacturer_origin']})")
                st.write(f"**Prescription Required:** {'Yes' if row['prescription_required'] else 'No'}")
                st.write(f"**Details:** {row['drug_content']}")

                if pd.notna(row['img_urls']):
                    st.image(row['img_urls'].split(',')[0], width=150, caption=row['med_name'])