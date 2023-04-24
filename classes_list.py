import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

import networkx as nx

from topcorr import *

class TMFG_class:
	def __init__(self, unweighted, edge_type, num=None, dataset_name=None, alpha=None, method=None, correlation_type=None, step=None, centrality_func=nx.degree_centrality):
		self.num = num
		self.dataset_name = dataset_name
		self.alpha = alpha
		self.method = method
		self.correlation_type = correlation_type
		self.rank = None
		self.step = step
		self.centrality_func = centrality_func
		self.unweighted = unweighted
		self.edge_type = edge_type

	def fit(self, x, y=None):
		data = pd.DataFrame(x).fillna(method="ffill").fillna(method="bfill")
		data = data.loc[:, data.std() > 0.0]
		data = data.to_numpy()

		G = tmfg(data, self.method, self.dataset_name, self.correlation_type, self.alpha, self.step, \
	   			unweighted=self.unweighted, edge_type=self.edge_type)
		
		if self.unweighted:
			centrality = self.centrality_func(G)
		else:
			if self.centrality_func == nx.closeness_centrality:
				centrality = self.centrality_func(G, distance='weight')
			elif self.centrality_func == nx.betweenness_centrality:
				centrality = self.centrality_func(G, weight='weight')
			else: #degree
				centrality = self.centrality_func(G)


		sorted_nodes = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1], reverse=True)}
		self.rank = list(sorted_nodes.keys())[:self.num]
		return self

	def transform(self, x, y=None):
		return np.take(x, self.rank, axis=1)

	def fit_transform(self, x, y=None):
		self.fit(x, y)
		transformed_x = self.transform(x, y)
		return transformed_x


