# importing necessary libraries
import pandas as pd
import numpy as np
#!pip install kmodes
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
dataframe = pd.read_csv("./data/EVdata.csv")
data = pd.DataFrame(dataframe)
# Elbow curve to find optimal K
cost = []
K = range(1,10)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 10, verbose=1)
    kmode.fit_predict(data)
    cost.append(kmode.cost_)
    
plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()

# Elbow Curve
# Build a model with 4 clusters

# Building the model with 4 clusters
kmode = KModes(n_clusters=4, init = "random", n_init = 5, verbose=1)
clusters = kmode.fit_predict(data)

data.insert(0, "Cluster", clusters, True)
data.to_excel("./data/output.xlsx")
print(data)