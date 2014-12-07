import json
import numpy as np
import re
from neo4jrestclient.client import GraphDatabase
from sklearn import cross_validation
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.grid_search import GridSearchCV

GRAPH_SOURCES = [u'http://pr.cs.cornell.edu/anticipation/', 
		u'http://wordnet.princeton.edu/', 
		u'http://sw.opencyc.org', 
		u'http://pr.cs.cornell.edu/hallucinatinghumans/']

# reads the graph_feedback.json file, and returns two lists of dictionary objects
# - one for nodes and another for links.
# make sure that filepath to json file is correct for your local setting.
def ingest_graph_feedback():
	nodes = []
	links = []
	with open('../project_data/graph_feedback.json') as feedback:
		for line in feedback:
			raw_object = json.loads(line)
			graph_object = {'id_node': raw_object['id_node'], 
				'feedback_type': raw_object['feedback_type'], 
				'node_handle': raw_object['node_handle'], 
				'action_type': raw_object['action_type']}
			# 'id_node' for links takes the form 'linkAtoB'
			if graph_object['id_node'][1:5] == 'link':
				links.append(graph_object)
			else:
				nodes.append(graph_object)
	return nodes, links


def loocv_clf(X, Y, clf=svm.SVC(kernel='linear', C=1)):
	misclassified = 0
	true_good = 0
	true_bad = 0
	false_good = 0
	false_bad = 0
	loo = cross_validation.LeaveOneOut(len(X))
	for train, test in loo:
		X_train, X_test = [X[i] for i in train], X[test]
		y_train, y_test = [Y[i] for i in train], Y[test]
		clf.fit(X_train, y_train)
		y_hat = clf.predict(X_test)[0]
		if y_hat != y_test: 
			misclassified += 1
			if y_test == 1:
				false_bad += 1
			else:
				false_good += 1
		else:
			if y_test == 1:
				true_good += 1
			else:
				true_bad += 1
	# report misclassification error
	print misclassified, true_good, true_bad, false_good, false_bad
	misclassification_error =  float(misclassified) / len(X)
	precision = float(true_bad) / (true_bad + false_bad)
	recall = float(true_bad) / (true_bad + false_good)
	return precision

# returns a list |X| of lists node features |phi_x|, a map from id_node 
# to corresponding index for the feature representation of that node in X,
# and a list |Y| of labels
def extract_node_features(nodes, multiclass=False):
	X = []
	Y = []
	index_map = {}
	gdb = GraphDatabase('http://ec2-54-187-76-157.us-west-2.compute.amazonaws.com:7474/db/data/')
	for i, node in enumerate(nodes):
		# phi = [handle_length, num_non_alpha in handle, belief, num_links, |indicators for source urls|]
		phi = []
		node_handle = node['node_handle']
		# handle_length
		phi.append(len(node_handle))
		# num_non_alpha characters
		phi.append(len([c for c in node_handle if not c.isalpha()]))
		q = 'MATCH (n{handle:' + node_handle + '})-[r]-(x) RETURN r, n, x'
		links = gdb.query(q=q)
		source_urls = set()
		belief = 0
		neighbor_beliefs = []
		for link in links:
			s_url = link[0]['data']['source_url']
			source_urls.add(s_url)
			try:
				belief = link[1]['data']['belief']
			except KeyError:
				pass
		#belief
		phi.append(belief)
		# num_links
		phi.append(len(links))
		# indicator variables for urls
		for source in GRAPH_SOURCES:
			if source in source_urls:
				phi.append(1)
			else:
				phi.append(0)
		action_type = node['action_type']
		if not multiclass:
			# binary classification, 'GOOD_NODE' = 1
			if action_type == "'GOOD_NODE'":
				Y.append(1)
			else:
				Y.append(2)
		else:
			# multiclass classification
			if action_type == "'GOOD_NODE'":
				Y.append(1)
			elif action_type == "'REMOVE_NODE'":
				Y.append(2)
			elif action_type == "'SPLIT_NODE'":
				Y.append(3)
			elif action_type == "'RENAME_NODE'":
				Y.append(4)
			else:
				print action_type
		index_map[node['id_node']] = i
		X.append(phi)
	return X, Y, index_map

def extract_link_features(links):
	X = []
	Y = []
	for i, link in enumerate(links):
		phi = []
		p = re.compile("'link(\d+)to(\d+)'")
		print link['id_node']
		m = p.match(link['id_node'])
		start_id = m.group(1)
		end_id = m.group(2)
		gdb = GraphDatabase('http://ec2-54-187-76-157.us-west-2.compute.amazonaws.com:7474/db/data/')
		q = "MATCH (n{id:'" + start_id + "'})-[r]-(x{id:'" + end_id + "'}) RETURN r, n, x"
		node_handle = "'shoe'"
		# q = 'MATCH (n{handle:' + node_handle + '})-[r]-(x) RETURN r, n, x'
		result = gdb.query(q=q)
		length = len(result)
		print length
		print "Before Loop"
		for x in range(0, length):
			print result[x]
		print "After Loop"
		print result
		break

# remaps an array of multiclass labels Y, to an equivalent list of binary labels
def multiclass_labels_to_binary(Y):
	return [int(y == 1) for y in Y]

# search for best parameters
def grid_search(X, Y, bin_Y, clf):
	kernels = ('linear', 'rbf')
	C = [0.1, 1, 5, 10, 50, 100]
	gamma = [0.0001, 0.001, 0.01, 0.1, 1] # for rbf kernel
	degree = [1, 2, 3, 4, 5] # for polynomial kernel
	parameters = {'kernel': kernels, 'C': C} #, 'gamma': gamma, 
			#'degree': degree}
	grid_fit = GridSearchCV(clf, parameters)
	grid_fit.fit(X, bin_Y)
	return grid_fit

# executed by python graph_clf_features.py
nodes, links = ingest_graph_feedback()
# extract_link_features(links)
X, Y, index_map = extract_node_features(nodes, multiclass=True)
bin_Y = multiclass_labels_to_binary(Y)
# svm_loocv_error = loocv_clf(X,Y)
# svm_rbf_clf = svm.SVC(kernel='rbf')
# bin_svm_loocv_error = loocv_clf(X, bin_Y)
# rbf_loocv_error = loocv_clf(X, Y, svm_rbf_clf)
# clf = MultinomialNB()
# nb_loocv_error = loocv_clf(X, Y, clf)
# bin_nb_loocv_error = loocv_clf(X, bin_Y, clf)
grid_fit_svm = grid_search(X, Y, bin_Y, svm.SVC())
grid_error = loocv_clf(X, bin_Y, grid_fit_svm)
# logistic_regression_clf = LogisticRegression()
# lr_loocv_error = loocv_clf(X, bin_Y, logistic_regression_clf)
# lda_clf = LDA()
# lda_loocv_error = loocv_clf(X, Y, lda_clf)
# bin_lda_loocv_error = loocv_clf(X, bin_Y, lda_clf)
print "misclassification error on nodes:"
print "multiclass:"
# print "LDA", lda_loocv_error
# print "Linear SVM", svm_loocv_error
# print "RBF SVM", rbf_loocv_error
# print "Naive Bayes", nb_loocv_error
print "binary classification:"
# print "logistic regression", lr_loocv_error
# print "linear SVM", bin_svm_loocv_error
# print "naive bayes", bin_nb_loocv_error
# print "LDA", bin_lda_loocv_error
print "grid search SVM", grid_error
print grid_fit_svm.get_params()
