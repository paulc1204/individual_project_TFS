import glob
import os
from collections import Counter
from joblib import Parallel, delayed
import pickle
import json
import argparse
import copy

import pandas as pd
import numpy as np
import scipy.io
from scipy.io import arff
from scipy.stats import spearmanr
from scipy import stats

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, cross_val_score, cross_validate
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.decomposition import PCA

import networkx as nx
from topcorr import *
from customized_statistical_tests import *
from classes_list import *
from tmfg_core import *

parser = argparse.ArgumentParser(description='TMFG Feature Selection.')
parser.add_argument('--stage', type=str, default='TMFG_FS', help="Computation of correlation matrices.")
parser.add_argument('--dataset', type=str, default='lung_small', help="Dataset to be used for the experiment.")
parser.add_argument('--classification_algo', type=str, default='RF', help="Algorithm to be used during classification.")
parser.add_argument('--cc_type', type=str, default='pearson', help="Type of correlation coefficient to be computed.")
parser.add_argument('--stat_test_pair', type=str, default='tmfg_ifs', help="Pair of classifiers to be statistically compared.")
parser.add_argument('--stat_test_setting', type=str, default='local', help="Local or global model in statistical comparison.")
parser.add_argument('--test_mode', type=str, default='local', help="Test mode: local or global parameters.")
parser.add_argument('--centrality', type=str, default='degree', help="Centrality measure to be used.")
parser.add_argument('--unweighted', type=str, default='false', help="If resulting TMFG will be unweighted.")
parser.add_argument('--edge_type', type=str, default='norm', help="Edge type for calculating centrality: distance, norm, or sq")
parser.add_argument('--corr_type', type=str, default='normal', help="Correlation type for constructing TMFG: normal or square")
args = parser.parse_args()

centrality_dictionary = {
	"degree": nx.degree_centrality,
	"eigenvector": nx.degree_centrality,
	"closeness": nx.closeness_centrality,
	"betweenness": nx.betweenness_centrality
}

def get_mat_file_name(path):

	filename = os.path.splitext(os.path.basename(path))[0]
	return filename

def read_mat_files(path):

	mat = scipy.io.loadmat(path)
	X = mat['X'].astype(float)
	y = mat['Y'][:, 0]

	return X,y

def read_arff_files(path):
	arff_file = arff.loadarff(path)[0]
	df = pd.DataFrame(arff_file)
	X = np.array(df.iloc[:, :-1]).astype(float)
	y = np.array(df.iloc[:, -1]).astype(int)
	
	return X,y

def train_test_split_files(X, y, filename, data_dictionary):

	X, y = shuffle(X, y, random_state=0)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

	local_data_dictionary = {'X_train': X_train,
							'X_test': X_test,
							'y_train': y_train,
							'y_test': y_test}

	data_dictionary[filename] = local_data_dictionary

	return data_dictionary

def get_data_description(data_dictionary, filename, description_dictionary):

	local_description_dictionary = {'#_features': data_dictionary[filename]['X_train'].shape[1],
								'#_samples_training': data_dictionary[filename]['X_train'].shape[0],
								'#_samples_test': data_dictionary[filename]['X_test'].shape[0],
								'counting_labels_training': Counter(data_dictionary[filename]['y_train']),
								'counting_labels_test': Counter(data_dictionary[filename]['y_test'])}

	description_dictionary[filename] = local_description_dictionary

	return description_dictionary

def get_data_files_extension(path):

	extension = os.path.splitext(os.path.basename(path))[1]
	return extension

def read_dexter_dataset(path, dataset_type, read_data_dictionary):

	data_dictionary = read_data_dictionary
	extension = get_data_files_extension(path)

	if extension == '.labels':
		y = np.loadtxt(path)

	vectors = np.zeros((300, 20000))

	if extension == '.data':

		with open(path, mode='r') as fid:
			data = fid.readlines()

		row = 0
		for line in data:
			line = line.strip().split()
			for word in line:
				col, val = word.split(':')
				vectors[row][int(col)-1] = int(val)
			row += 1

		X = vectors

	if dataset_type == 'train' and extension == '.data':

		data_dictionary['X_train'] = X

	if dataset_type == 'train' and extension == '.labels':

		data_dictionary['y_train'] = y

	if dataset_type == 'valid' and extension == '.data':

		data_dictionary['X_test'] = X

	if dataset_type == 'valid' and extension == '.labels':

		data_dictionary['y_test'] = y

	return data_dictionary

def read_non_dexter_dataset(path, dataset_type, read_data_dictionary):

	data_dictionary = read_data_dictionary
	extension = get_data_files_extension(path)

	if extension == '.labels':
		y = np.loadtxt(path)
	else:
		X = np.loadtxt(path)

	if dataset_type == 'train' and extension == '.data':

		data_dictionary['X_train'] = X

	if dataset_type == 'train' and extension == '.labels':

		data_dictionary['y_train'] = y

	if dataset_type == 'valid' and extension == '.data':

		data_dictionary['X_test'] = X

	if dataset_type == 'valid' and extension == '.labels':

		data_dictionary['y_test'] = y

	return data_dictionary

def read_data_files(filename, paths_list, data_dictionary):

	read_data_dictionary = {}
	for path in paths_list:
		
		if ('train' in path) or ('valid' in path):

			dataset_type = 'train' if 'train' in path else 'valid'

			if filename == 'DEXTER':
				read_data_dictionary = read_dexter_dataset(path, dataset_type, read_data_dictionary)
			else:
				read_data_dictionary = read_non_dexter_dataset(path, dataset_type, read_data_dictionary)

	data_dictionary[filename] = read_data_dictionary

	return data_dictionary

def produce_correlation_matrix(data_dictionary, dataset_name, method):

	data = pd.DataFrame(data_dictionary['X_train']).fillna(method="ffill").fillna(method="bfill")
	data = data.loc[:, data.std() > 0.0]

	if method == 'spearman':
		data.corr(method='spearman').to_csv(f'spearman_{dataset_name}.csv', index=False)
	elif method == 'pearson':
		data.corr().to_csv(f'pearson_{dataset_name}.csv', index=False)

def hyper_opt_tmfg(classification_algo, X_train, y_train, correlation_value, correlation_type, num, alpha, dataset_name, \
		   centrality, unweighted, edge_type):

	###########
	# if correlation_value == 'energy' and correlation_type == 'square':
	# 	return None
	# else:
	if correlation_value == 'energy' and alpha != None:
		np.random.seed(0)
		if classification_algo == 'LinearSVC':
			clf = LinearSVC(random_state=0, max_iter=50000)
		elif classification_algo == 'KNN':
			clf = KNeighborsClassifier()
		else:
			clf = RandomForestClassifier(random_state=0)

		pipeline = Pipeline([('tmfg_fs', TMFG_class(num=num, dataset_name=dataset_name, alpha=alpha, method=correlation_value, \
						correlation_type=correlation_type, step='cv', centrality_func=centrality_dictionary[centrality], \
						unweighted=unweighted, edge_type=edge_type)), ('scaling', StandardScaler()), ('estimator', clf)])
		
		tmfg_metric = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), scoring='balanced_accuracy').mean()

		tmfg_dictionary = {'correlation_value': correlation_value,
							'correlation_type': correlation_type,
							'num_features': num,
							'alpha': alpha,
							'score': round(tmfg_metric, 2)}

	elif correlation_value != 'energy' and alpha == None:
		np.random.seed(0)
		if classification_algo == 'LinearSVC':
			clf = LinearSVC(random_state=0, max_iter=50000)
		elif classification_algo == 'KNN':
			clf = KNeighborsClassifier()
		else:
			clf = RandomForestClassifier(random_state=0)
		pipeline = Pipeline([('tmfg_fs', TMFG_class(num=num, dataset_name=dataset_name, alpha=alpha, method=correlation_value, correlation_type=correlation_type, step='cv',\
						centrality_func=centrality_dictionary[centrality], unweighted=unweighted, edge_type=edge_type)), ('scaling', StandardScaler()), ('estimator', clf)])
		tmfg_metric = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), scoring='balanced_accuracy').mean()

		tmfg_dictionary = {'correlation_value': correlation_value,
							'correlation_type': correlation_type,
							'num_features': num,
							'alpha': alpha,
							'score': round(tmfg_metric, 2)}
	else:
		tmfg_dictionary = {'correlation_value': correlation_value,
							'correlation_type': correlation_type,
							'num_features': 10000000,
							'alpha': alpha,
							'score': 0}
	return tmfg_dictionary

def tmfg_pipeline(data_dictionary, dataset_name, classification_algo, centrality, unweighted, edge_type, correlation_type):
	unweighted = unweighted=='true'

	data = pd.DataFrame(data_dictionary['X_train']).fillna(method="ffill").fillna(method="bfill")
	data = data.loc[:, data.std() > 0.0]
	data = data.to_numpy()

	correlation_values = ['pearson', 'spearman', 'energy']
	
	if data.shape[1] > 200:
		num_features = [10, 50, 100, 150, 200]
	else:
		num_features = [10, 20, 30, 50, 70]

	alpha_values = list(np.arange(0.2, 1, 0.2))
	alpha_values.append(None)

	output = Parallel(n_jobs=8, verbose=1)(delayed(hyper_opt_tmfg)(classification_algo, data, data_dictionary['y_train'], \
							correlation_value, correlation_type, num_feature, alpha, dataset_name, centrality, unweighted, edge_type)\
							for correlation_value in correlation_values \
							for num_feature in num_features for alpha in alpha_values)
	output = [x for x in output if x is not None]
	output = sorted(output, key=lambda x:x['num_features'])

	list_cv = []
	for e in output:
		if e['num_features'] != 10000000:
			list_cv.append(e)
	del output

	# output_file = open(f'./full_tmfg_cv/{classification_algo}/{dataset_name}_tmfg_full_cv.json', 'w', encoding='utf-8')
	weight = "uw" if unweighted else "w"
	corr_type = '' if correlation_type=='normal' else 'sqcorr_'
	output_file = open(f'./full_tmfg_cv/{classification_algo}/{dataset_name}_{corr_type}{centrality}_{weight}_{edge_type}_tmfg_full_cv.json', 'w', encoding='utf-8')
	for dic in list_cv:
		json.dump(dic, output_file) 
		output_file.write("\n")

	optimization = (max(list_cv, key=lambda x:x['score']))
	print(optimization)
	cv_file = open(f"./optimal_tmfg_cv/{classification_algo}/{dataset_name}_{corr_type}{centrality}_{weight}_{edge_type}_optimal_tmfg_cv.pkl", "wb")
	pickle.dump(optimization, cv_file)
	cv_file.close()

def tmfg_test_pipeline(data_dictionary, dataset_name, test_mode, classification_algo, centrality, unweighted, edge_type, correlation_type):
	unweighted = unweighted=='true'

	X_train = data_dictionary['X_train']
	y_train = data_dictionary['y_train']

	data = pd.DataFrame(X_train).fillna(method="ffill").fillna(method="bfill")
	data = data.loc[:, data.std() > 0.0]
	data = data.to_numpy()
	X_train = data

	X_test = data_dictionary['X_test']
	y_test = data_dictionary['y_test']

	###
	weight = "uw" if unweighted else "w"
	corr_type = '' if correlation_type=='normal' else 'sqcorr_'
	if test_mode == 'local':

		# df = pd.read_json(f'./full_tmfg_cv/{classification_algo}/{dataset_name}_{centrality}_tmfg_full_cv.json',  lines=True)
		df = pd.read_json(f'./full_tmfg_cv/{classification_algo}/{dataset_name}_{corr_type}{centrality}_{weight}_{edge_type}_tmfg_full_cv.json', lines=True)

		if X_test.shape[1] > 200:
			n_features_list = [10, 50, 100, 150, 200]
		else:
			n_features_list = [10, 20, 30, 50, 70]
		matrix_report_dict = {}

		for i in n_features_list:

			local_df = df[df.num_features == i]
			local_df.reset_index(drop=True, inplace=True)
			optimal_values = local_df.iloc[local_df['score'].argmax()]

			np.random.seed(0)
			if classification_algo == 'LinearSVC':
				clf = LinearSVC(random_state=0, max_iter=50000)
			elif classification_algo == 'KNN':
				clf = KNeighborsClassifier()
			else:
				clf = RandomForestClassifier(random_state=0)
			pipeline = Pipeline([('tmfg_fs', TMFG_class(num=i, dataset_name=dataset_name, alpha=optimal_values['alpha'], \
					       method=optimal_values['correlation_value'], correlation_type=optimal_values['correlation_type'], step='test', \
							centrality_func=centrality_dictionary[centrality], unweighted=unweighted, edge_type=edge_type)), ('scaling', StandardScaler()), ('estimator', clf)])
			pipeline.fit(X_train, y_train)
			preds = pipeline.predict(X_test)

			c_matrix = confusion_matrix(y_test, preds)
			classification_report_dict = classification_report(y_test, preds, output_dict=True)

			matrix_report_dict[f'n_features_{i}'] = {'confusion_matrix': c_matrix, 'classification_report': classification_report_dict, 'preds': preds, 'y_true': y_test}

		output_file = open(f'./local_best_configuration_results/{classification_algo}/{dataset_name}_{corr_type}{centrality}_{weight}_{edge_type}_tmfg_test_local.pkl', 'wb')
		pickle.dump(matrix_report_dict, output_file) 
		output_file.close()

def statistical_comparison_pipeline_optima(data_dictionary, dataset_name, stat_test_setting, classifiers, classification_algo):

	X_train = data_dictionary['X_train']
	y_train = pd.DataFrame(data_dictionary['y_train'])

	data = pd.DataFrame(X_train).fillna(method="ffill").fillna(method="bfill")
	data = data.loc[:, data.std() > 0.0]
	data = data.to_numpy()
	X_train = data

	X_test = data_dictionary['X_test']
	y_test = data_dictionary['y_test']

	df_ifs = pd.read_json(f'./full_ifs_cv/{classification_algo}/{dataset_name}_ifs_full_cv.json', lines=True)
	df_tmfg = pd.read_json(f'./full_tmfg_cv/{classification_algo}/{dataset_name}_tmfg_full_cv.json', lines=True)

	n_features_list = [10, 50, 100, 150, 200]
	stat_test_to_be_run = ["paired_ttest_5x2cv"]
	stat_tests_ensamble = {}

	for stat_test in stat_test_to_be_run:

		stat_test_result_list = []

		for i in n_features_list:

			t = None
			p = None

			if 'ifs' in classifiers:

				if stat_test_setting == 'local':
					local_df_ifs = df_ifs[df_ifs.num_features == i]
					local_df_ifs.reset_index(drop=True, inplace=True)
					optimal_values_ifs = local_df_ifs.iloc[local_df_ifs['score'].argmax()]
				
				prep_pipeline_ifs = ('ifs', IFS_class(num=i, dataset_name=dataset_name, alpha=optimal_values_ifs['alpha'], factor=optimal_values_ifs['factor'], step='cv'))

			if 'tmfg' in classifiers:
				
				if stat_test_setting == 'local':
					local_df_tmfg = df_tmfg[df_tmfg.num_features == i]
					local_df_tmfg.reset_index(drop=True, inplace=True)
					optimal_values_tmfg = local_df_tmfg.iloc[local_df_tmfg['score'].argmax()]

				prep_pipeline_tmfg = ('tmfg_fs', TMFG_class(num=i, dataset_name=dataset_name, alpha=optimal_values_tmfg['alpha'], method=optimal_values_tmfg['correlation_value'], correlation_type=optimal_values_tmfg['correlation_type'], step='cv'))

			prep_pipeline_1 = prep_pipeline_tmfg
			prep_pipeline_2 = prep_pipeline_ifs

			if classification_algo == 'LinearSVC':
				clf_1 = LinearSVC(random_state=0, max_iter=50000)
			elif classification_algo == 'KNN':
				clf_1 = KNeighborsClassifier()
			else:
				clf_1 = DecisionTreeClassifier(random_state=0)
			pipeline_1 = Pipeline([prep_pipeline_1, ('scaling', StandardScaler()), ('estimator', clf_1)])

			if classification_algo == 'LinearSVC':
				clf_2 = LinearSVC(random_state=0, max_iter=50000)
			elif classification_algo == 'KNN':
				clf_2 = KNeighborsClassifier()
			else:
				clf_2 = DecisionTreeClassifier(random_state=0)
			pipeline_2 = Pipeline([prep_pipeline_2, ('scaling', StandardScaler()), ('estimator', clf_2)])

			pipeline_1 = pipeline_1
			pipeline_2 = pipeline_2
			X_estimator_1 = X_train
			X_estimator_2 = X_train

			t, p, additional_metrics_estimator_1, additional_metrics_estimator_2 = paired_ttest_5x2cv(estimator1=pipeline_1, 
				estimator2=pipeline_2, 
				X_estimator_1=X_estimator_1, 
				X_estimator_2=X_estimator_2,  
				y_estimator_1=y_train, 
				y_estimator_2=y_train, 
				scoring='balanced_accuracy')
			
			stat_test_result_list.append((i, t, p, additional_metrics_estimator_1, additional_metrics_estimator_2))

		stat_tests_ensamble[stat_test] = stat_test_result_list

	output_file = None

	if ('tmfg' in classifiers) and ('ifs' in classifiers):
		output_file = open(f'./stat_tests_{stat_test_setting}/{classification_algo}/{dataset_name}_stat_tests_tmfg_ifs_{stat_test_setting}.pkl', 'wb')
	
	pickle.dump(stat_tests_ensamble, output_file)
	output_file.close()

data_dictionary = {}
description_dictionary = {}

# entire_datasets = glob.glob('./data/entire_datasets/*')
entire_datasets = glob.glob('./data/*')
splitted_datasets = {'DEXTER': glob.glob('./data/splitted_datasets/DEXTER/*'), 
						'GISETTE': glob.glob('./data/splitted_datasets/GISETTE/*'),
						'MADELON': glob.glob('./data/splitted_datasets/MADELON/*'),
						'ARCENE': glob.glob('./data/splitted_datasets/ARCENE/*')}

for file in entire_datasets:

	if os.path.splitext(file)[1]=='.arff':
		filename = os.path.splitext(os.path.basename(file))[0]
		X, y = read_arff_files(file)
		data_dictionary = train_test_split_files(X, y, filename, data_dictionary)
		continue

	filename = get_mat_file_name(file)
	X, y = read_mat_files(file)
	data_dictionary = train_test_split_files(X, y, filename, data_dictionary)


for file in entire_datasets:

	filename = get_mat_file_name(file)
	description_dictionary = get_data_description(data_dictionary, filename, description_dictionary)

if args.stage == 'CM_Computation':
	produce_correlation_matrix(data_dictionary[args.dataset], args.dataset, args.cc_type)

elif args.stage == 'TMFG_FS':
	tmfg_pipeline(data_dictionary[args.dataset], args.dataset, args.classification_algo, args.centrality, \
	       			args.unweighted, args.edge_type, args.corr_type)

elif args.stage == 'TMFG_FS_TEST':
	tmfg_test_pipeline(data_dictionary[args.dataset], args.dataset, args.test_mode, args.classification_algo, \
		    		args.centrality, args.unweighted, args.edge_type, args.corr_type)

elif args.stage == 'INSAMPLE_STAT_COMPARISON_OPTIMA':
	stat_test_pair = args.stat_test_pair.split('_')
	statistical_comparison_pipeline_optima(data_dictionary[args.dataset], args.dataset, args.stat_test_setting, stat_test_pair, args.classification_algo)

elif args.stage == 'TMFG':
	tmfg_pipeline(data_dictionary[args.dataset], args.dataset, args.classification_algo, args.centrality, \
	       		args.unweighted, args.edge_type, args.corr_type)
	tmfg_test_pipeline(data_dictionary[args.dataset], args.dataset, args.test_mode, args.classification_algo, \
		    	args.centrality, args.unweighted, args.edge_type, args.corr_type)
elif args.stage == 'TMFG_Batch':
	unweighted = 'true'
	datasets = ['mfeat_fourier', 'har'] #madelon, mfeat_fourier, har
	centralities = ['betweenness', 'closeness', 'degree']
	edge_types = ['norm']
	corr_type = ['normal'] #square needed
	
	for dataset in datasets:
		for c in centralities:
			for etype in edge_types:
				for corr_t in corr_type:
					tmfg_pipeline(data_dictionary[dataset], dataset, args.classification_algo, c, unweighted, etype, corr_t)
					tmfg_test_pipeline(data_dictionary[dataset], dataset, args.test_mode, args.classification_algo,\
					c, unweighted, etype, corr_t)

else:
	print('The chosen stage is not valid.')