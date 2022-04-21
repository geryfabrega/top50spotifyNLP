import pandas as pd
import numpy as np
import time
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

print("starting now")

df = pd.read_csv("Data Science Proj - Sheet1 (1).csv")
print(df.head())

vec = TfidfVectorizer(stop_words="english")
vec.fit(df.lyrics.values)
features = vec.transform(df.lyrics.values)

cls = MiniBatchKMeans(n_clusters=5, random_state=None)
cls.fit(features)

# predict cluster labels for new dataset

cls.predict(features)

# to get cluster labels for the dataset used while
# training the model (used for models that does not
# support prediction on new dataset).
cls.labels_

pca = PCA(n_components=3, random_state=None)
reduced_features = pca.fit_transform(features.toarray())

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)

print(df.song.values)

# plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
# plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

# for i, txt in enumerate(df.song.values):
#     plt.annotate(txt, (reduced_features[i,0], reduced_features[i,1]))

x = reduced_features[:,0]
y = reduced_features[:,1]
z = reduced_features[:,2]

colors = cls.labels_

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    text=df.song.values,
    mode='markers+text',
    marker=dict(
        size=12,
        color=colors,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8,
    ),
    textfont=dict(family='sans serif',
        size=14,
        color='#000000'
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.update_traces(textposition='top center')
fig.update_layout(template="ggplot2")

fig.write_html("Mygraph2.html",full_html=False, include_plotlyjs="cdn")

