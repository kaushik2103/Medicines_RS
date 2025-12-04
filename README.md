# Medicines_RS
## This repo consist of only a best performing model that is KNN with 98.3 % F1-score.
Built a medicine recommendation engine covering 141 disease categories, benchmarking multiple ML models — Decision Tree (88.6% F1-score), Random Forest (93.4% F1-score), and SVM (96.1% F1-score) — and confirmed that K-Nearest Neighbors outperformed all models with a 98.33% F1-score for drug-composition-based similarity matching.

Implemented the final system using TF-IDF vectorization + KNN, generating the top 5 medicine recommendations based on drug composition, disease type, and symptom inputs.

Created a dynamic Streamlit UI for real-time recommendations, presenting key medicine attributes such as price, manufacturer, drug class, and prescription status for improved decision-making.

Achieved sub-4000ms query times by applying model caching, persistent storage for TF-IDF vectors and KNN indices, and precomputing similarity matrices.
