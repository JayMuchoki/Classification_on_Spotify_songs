# 🎵 **Spotify Song Clustering using Unsupervised Classification**

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-green)](https://spotifysongclassification.streamlit.app/)

<img width="1440" height="984" alt="screencapture-localhost-8501-2025-11-26-12_55_15" src="https://github.com/user-attachments/assets/78c029b2-e2d8-4bbf-aeae-f08423802aa1" />


---

## Overview
This project focuses on **unsupervised classification of Spotify songs**. The goal is to **cluster songs based on their audio features** to automate playlist generation and enhance music recommendations. By leveraging machine learning, we can identify distinct groups of songs that share similar musical characteristics.

---

## Dataset
The dataset consists of thousands of tracks fetched from **Spotify’s Web API**, including features such as:

- `energy`  
- `danceability`  
- `tempo`  
- `valence`  
- `popularity`, `duration_ms`, `explicit`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `time_signature`

---

## Challenges
- **No predefined labels:** Songs do not have predefined clusters, so we must identify meaningful groups using unsupervised learning.  
- **High feature variation:** Danceability, energy, valence, and loudness vary greatly, providing opportunities to cluster by mood or style.  
- **Skewed distributions:** Features like instrumentalness, acousticness, and speechiness are heavily skewed toward 0.  
- **Duration differences:** Most songs are under ~17 minutes (`1,000,000 ms`), with a few outliers.  
- **Explicit vs non-explicit content:** Majority of songs are non-explicit.  
- **Mode skew:** Most songs are in a major key (`mode=1`).

---

## Preprocessing
1. **Feature selection:** Chose numerical features:  
   `['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']`

2. **Standardization:** Applied `StandardScaler` to bring features to the same scale, ensuring equal importance in clustering.  

3. **Dimensionality reduction:** Used **PCA (Principal Component Analysis)** to reduce features to 2 components while retaining most of the dataset’s variation.  

---

## Clustering
Applied the following unsupervised learning algorithms:

- ✅ **K-Means Clustering**  
- ✅ **Hierarchical Clustering**  

Both methods produced **3 consistent clusters**.

**Visualization:**  
- Cluster 0 → Red  
- Cluster 1 → Blue  
- Cluster 2 → Green  

<img width="1215" height="725" alt="image" src="https://github.com/user-attachments/assets/61adc957-e52c-40ff-94cc-268b20e14efa" />

---

## Cluster Interpretation
After clustering, meaningful labels were assigned based on dominant musical features:

| Cluster | Label | Characteristics |
|---------|-------|----------------|
| 0 | Chill Pop / Acoustic Groove | Lower energy, higher acousticness, relaxed tempo |
| 1 | Mainstream Dance / EDM-Pop | High energy, danceable, electronic elements |
| 2 | Stripped Down / Acoustic Ballads | Minimal instrumentation, emotional tone, acoustic sounds |



---

## Insights & Recommendations
- PCA helped reduce dimensionality while preserving critical patterns, making clustering more effective.  
- KMeans grouped songs by shared musical features, enabling automated playlist generation.  
- Future work: Incorporate **user interaction data** (likes, skips, play counts) for more personalized clustering and recommendations.

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn (StandardScaler, PCA, KMeans, Hierarchical Clustering)  
- Matplotlib / Seaborn for visualizations  
- Streamlit for interactive deployment

---

## Try it Out
Access the interactive Streamlit app here:  
[Spotify Song Classification](https://spotifysongclassification.streamlit.app/)
