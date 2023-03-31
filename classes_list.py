import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

import networkx as nx

from infinite_features_selection.inf_fs import inf_fs, select_inf_fs
from topcorr import *

class IFS_class:
	def __init__(self, num=None, dataset_name=None, alpha=None, factor=None, step=None):
		self.num = num
		self.dataset_name = dataset_name
		self.alpha = alpha
		self.factor = factor
		self.step = step
		self.rank = None

	def fit(self, x, y=None):
		x_sel, self.rank = select_inf_fs(x, self.num, self.dataset_name, self.alpha, self.factor, self.step)
		return self

	def transform(self, x, y=None):
		return np.take(x, self.rank, axis=1)

	def fit_transform(self, x, y=None):
		self.fit(x, y)
		transformed_x = self.transform(x, y)
		return transformed_x

class PCA_class:
	def __init__(self, num=None, dataset_name=None, scaling=None, step=None):
		self.num = num
		self.dataset_name = dataset_name
		self.scaling = scaling
		self.pca = None
		self.rank = None
		self.explained_variance = None
		self.step = step

	def fit(self, x, y=None):
		if self.scaling == 'StandardScaler':
			scaler = StandardScaler()
		elif self.scaling == 'MinMaxScaler':
			scaler = MinMaxScaler()
		else:
			scaler = None

		if scaler != None:
			pca_X_train = scaler.fit_transform(x)
		else:
			pca_X_train = x

		self.pca = PCA()
		self.pca.fit(pca_X_train)
		
		self.explained_variance = self.pca.explained_variance_ratio_
		
		dict_pca = {}
		for index, element in enumerate(self.explained_variance):
			dict_pca[index] = element

		pca_features = {k: v for k,v in sorted(dict_pca.items(), key=lambda item: item[1], reverse=True)}
		self.rank = list(pca_features.keys())[:self.num]
		return self
		
	def transform(self, x, y=None):
		return np.take(x, self.rank, axis=1)

	def fit_transform(self, x, y=None):
		self.fit(x, y)
		transformed_x = self.transform(x, y)
		return transformed_x

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
		
		####
		if self.unweighted:
			centrality = self.centrality_func(G)
		else:
			if self.centrality_func == nx.closeness_centrality:
				centrality = self.centrality_func(G, distance='weight')
			elif self.centrality_func == nx.betweenness_centrality:
				centrality = self.centrality_func(G, weight='weight')
			elif self.centrality_func == nx.degree_centrality:
				centrality = self.centrality_func(G)
			else: #eigenvector
				centrality = self.centrality_func(G, weight='weight')

		sorted_nodes = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1], reverse=True)}
		self.rank = list(sorted_nodes.keys())[:self.num]
		return self

	def transform(self, x, y=None):
		return np.take(x, self.rank, axis=1)

	def fit_transform(self, x, y=None):
		self.fit(x, y)
		transformed_x = self.transform(x, y)
		return transformed_x


