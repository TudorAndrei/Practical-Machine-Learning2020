import random
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coclust.evaluation.external import accuracy
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                             classification_report, completeness_score,
                             homogeneity_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.text import TSNEVisualizer

warnings.simplefilter(action='ignore', category=FutureWarning)
rs = 42  # random state
np.random.seed(rs)


def random_prediction(X, y):
    """ Generate a random prediction and evaluate it """
    labels = list(set(y))
    predictions = [np.random.choice(labels) for _ in X]
    return accuracy_score(y, predictions)


def get_random_prediction(X_test, y_test, trials=1000):
    """Predict for N ammount of trials"""
    np.random.seed()
    sum = 0
    trials
    for i in tqdm(range(trials)):
        sum += random_prediction(X_test, y_test)
    np.random.seed(rs)
    return sum/trials


colors = ['blue', 'green', 'chocolate', 'gold',
          'yellow', 'lime', 'pink', 'fuchsia']
centroid_color = ['red']


def plot_clusters(X, y_true, y_pred, mode=None, centroids=None):
    """Plotting function for the Kmeans algorithm"""
    # this was inspired from the plot2d from the lab exercises
    transformer = None
    X_rescaled = X

    if mode is not None:
        transformer = mode(n_components=2, random_state=42)
        if mode == TSNE:
            if centroids is not None:
                X_centroids = np.append(X, centroids, axis=0)
                X_centroids = transformer.fit_transform(X_centroids)
                X_rescaled = X_centroids[:X.shape[0]]

        X_rescaled = transformer.fit_transform(X)

    for x, yp in zip(X_rescaled, y_pred):
        plt.plot(x[0], x[1],
                 c=colors[yp],
                 marker='*',
                 label=str(yp)
                 )

    if centroids is not None:
        # TNSE does not plot correctly the centroids
        C_rescaled = centroids
        if transformer is not None:
            if mode == TSNE:
                C_rescaled = X_centroids[-1 * centroids.shape[0]:]
            elif mode == PCA:
                C_rescaled = transformer.fit_transform(centroids)

        for c in C_rescaled:
            plt.plot(c[0], c[1],
                     marker='X',
                     markersize=10,
                     c='red')
#     plt.legend()
    plt.show()


def plot_dbscan(model, X, mode=None):
    """Plotting function for the DBSCAN algorithm"""
    # this was inspired from the sklearn example of using DBSCAN
    # Get the indexes of the labels, ignores the noise (-1)
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True

    unique_labels = set(model.labels_)
    n_clusters = len(unique_labels) - 1
    colors = plt.cm.get_cmap('rainbow', len(unique_labels))

    X_rescaled = X
    if mode is not None:
        transformer = mode(n_components=2, random_state=42)
        X_rescaled = transformer.fit_transform(X)

    for index, label in enumerate(unique_labels):
        if label == -1:
            # set the color for noise points as black
            c = (0, 0, 0, 1)
        else:
            c = colors(index)
        # get a mask for a label
        class_member_mask = (model.labels_ == label)

        # plot the cores of the cluster
        x = X_rescaled[class_member_mask & core_samples_mask]
        plt.plot(x[:, 0], x[:, 1], 'o', color=c, markersize=10)
        # plot the rest of the points from the cluster
        x = X_rescaled[class_member_mask & ~core_samples_mask]
        plt.plot(x[:, 0], x[:, 1], '*', color=c, markersize=2)

    plt.title(f'No of clusters: {n_clusters}')
    plt.show()


def score_results(X, true_labels, predicted_labels):
    """Multiple scoring functions"""
    metrics = {
        'Completeness Score': completeness_score,
        'homogeneity Score': homogeneity_score,
        'Accuracy': accuracy
    }
    for name, func in metrics.items():
        print(f'{name}:\t {round(func(true_labels, predicted_labels), 5) : >5}')

    assert(len(set(predicted_labels)) >
           1), f"Not enough no. of labels ( {len(set(predicted_labels))} < 2 ) "
    print(
        f'Silhouette Score:\t {round(silhouette_score(X, predicted_labels, random_state=rs), 5) : >5}')


def plot_distance(distances, min_=None, max_=None):
    """Plot the distances of the NearestNeighbors"""

    plt.subplot(1, 2, 1)
    dist = distances[:, 1]
    plt.plot(dist)

    if max_ and min_:

        max_ = min(max_, len(dist))
        plt.subplot(1, 2, 2)
        zoom_in = distances[min_: max_, 1]
        x_axis = np.arange(min_, max_)
        plt.plot(x_axis, zoom_in)
        plt.tight_layout()

    plt.show()


# Load the data
path = r'./data/processed_full.csv'
samples = 1000  # the upper limmit of the sampling
data = pd.read_csv(path)
new_data = pd.DataFrame()
cols = ['rock', 'electronic', 'rap', 'pop/r&b',
        'folk/country', 'experimental', 'metal', 'jazz']
for col in cols:
    try:
        # select samples no. of data from the class
        aux = data[data['genre'] == col].sample(
            samples, axis=0, random_state=rs)
    except:
        # In case there not enough samples in the class get all of them
        sub_sample = len(data[data['genre'] == col])
        aux = data[data['genre'] == col].sample(
            sub_sample, axis=0, random_state=rs)
    new_data = pd.concat([aux, new_data], ignore_index=True, axis=0)


# shuffle the data
data = new_data.sample(frac=1).reset_index(drop=True)
X = data.content.apply(lambda x: np.str_(x))
y = data.genre.apply(lambda x: np.str_(x))

# count the no. of samples per class
colect = Counter(data.genre).most_common(10)
n_clusters = len(colect)
# get the no. of cluster
print(f"The number of cluster is {n_clusters} out of {len(data.index)} items:")
for gen, app in colect:
    print(f"{gen} :{app}")


# Parameters for TF-IDF
min_apparitions = 10
max_nr_features = 150
max_apparitions = 100
lower_case = True
params_tfidf = {
    'min_df': min_apparitions,
    'max_features': max_nr_features,
    'lowercase': lower_case,
    'max_df': max_apparitions,
    'stop_words': 'english'}
# Parameters for the dimentionality reduction
params_dr = {'n_components': 2, 'random_state': rs}


tfidf = TfidfVectorizer(**params_tfidf)
pca = PCA(**params_dr)
tsne = TSNE(**params_dr)

X_tfidf = tfidf.fit_transform(X)
le = LabelEncoder()
y_labels = le.fit_transform(y)


print(X_tfidf.shape)


# Generate a random predction
result = float(round(random_prediction(X_tfidf, y_labels), 4)) * 100
print(f"The probability is {result}%")


# Random Forest model

X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, stratify=y, random_state=rs)
vectorizer = TfidfVectorizer(min_df=10)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
model = RandomForestClassifier()
model.fit(X_train, y_train)
result = round(model.score(X_test, y_test), 2) * 100
print(f'Random forest Classification {result}%')


# k Means

kmeans = KMeans(n_clusters=n_clusters, random_state=rs)
kmeans.fit(X_tfidf)
score_results(X_tfidf, y_labels, kmeans.labels_)


plot_clusters(X_tfidf.todense(), y_labels, kmeans.labels_,
              mode=PCA, centroids=kmeans.cluster_centers_)


# Train a Kmeans model on the TSNE reduced data
X_tfidf_tsne = tsne.fit_transform(X_tfidf)
kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=rs)
kmeans_tsne.fit(X_tfidf_tsne)
score_results(X_tfidf_tsne, y_labels, kmeans_tsne.labels_)


plot_clusters(X_tfidf_tsne, y_labels, kmeans_tsne.labels_,
              centroids=kmeans_tsne.cluster_centers_)


# Train a Kmeans model on the PCA reduced data

X_tfidf_pca = pca.fit_transform(X_tfidf.todense())
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=rs)
kmeans_pca.fit(X_tfidf_pca)
score_results(X_tfidf_pca, y_labels, kmeans_pca.labels_)

plot_clusters(X_tfidf_pca, y_labels, kmeans_pca.labels_,
              centroids=kmeans_pca.cluster_centers_)

# DBSCAN on t-SNE
# Find the epsilon value

X_tfidf_tsne = tsne.fit_transform(X_tfidf)
nn_tsne = NearestNeighbors(n_jobs=-1).fit(X_tfidf_tsne)
distances, idx = nn_tsne.kneighbors(X_tfidf_tsne)
distances_tsne = np.sort(distances, axis=0)
plot_distance(distances_tsne, 5750, 6000)
eps_tsne = 4.8
dbscan_tsne = DBSCAN(eps=eps_tsne, min_samples=50, n_jobs=-1)
dbscan_tsne.fit(X_tfidf_tsne)
score_results(X_tfidf_tsne, y_labels, dbscan_tsne.labels_)
plot_dbscan(dbscan_tsne, X_tfidf_tsne)

# DBSCAN on PCA
X_tfidf_pca = pca.fit_transform(X_tfidf.todense())
X_tfidf_pca_norm = Normalizer().fit_transform(X_tfidf_pca)
nn_pca = NearestNeighbors(n_jobs=-1).fit(X_tfidf_pca_norm)
distances_pca, idx = nn_pca.kneighbors(X_tfidf_pca_norm)
distances_pca = np.sort(distances_pca, axis=0)
plot_distance(distances_pca, 5000, 6000)
eps_tsne = 0.005
dbscan_pca = DBSCAN(eps=eps_tsne, min_samples=40, n_jobs=-1)
dbscan_pca.fit(X_tfidf_pca)
score_results(X_tfidf_pca, y_labels, dbscan_pca.labels_)
plot_dbscan(dbscan_pca, X_tfidf_pca)
