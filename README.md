# Music Genre Classification & Wine Quality Clustering

A two-part machine learning project in Python using pandas and scikit-learn.

| Part | Task | Method | Data |
|---|---|---|---|
| 1 | Multi-class classification (10 music genres) | SVM (RBF kernel) vs Random Forest | Audio-feature dataset |
| 2 | Unsupervised clustering | K-Means + PCA visualisation | UCI Red Wine Quality |

## Highlights
- Stratified 80/20 split; `StandardScaler` fitted on training data only (`fit` on train, `transform` elsewhere) to avoid data leakage
- SVM (~58% accuracy) slightly outperformed Random Forest (~57%) on a 10-class problem
- Error analysis: Rap/Hip-hop and Alternative/Rock/Country are the most confused genres because they overlap in audio features
- K-Means (k = 3, chosen with the elbow method) split wines into three interpretable groups driven by acidity/pH and sulphur dioxide levels

## Data
- **Music genre data:** provided in a university course and **not included** in this repository. Columns: `instance_id`, `artist_name`, `track_name`, `popularity`, audio features (`acousticness`, `danceability`, `duration_ms`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`) and the target `music_genre`. To run Part 1, place a CSV with this layout in the project folder as `FIT1043-MusicGenre-Dataset.csv`.
- **Wine data:** [UCI Red Wine Quality (Kaggle)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). Download `winequality-red.csv` into the project folder.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook music-genre-classification-and-wine-clustering.ipynb
```

## Possible next steps
Hyperparameter search with cross-validation, feature engineering, gradient boosting, and silhouette scores for cluster stability.
