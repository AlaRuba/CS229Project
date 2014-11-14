import json
import numpy as np
from neo4jrestclient.client import GraphDatabase
from sklearn import cross_validation
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

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
	score = 0
	loo = cross_validation.LeaveOneOut(len(X))
	for train, test in loo:
		X_train, X_test = [X[i] for i in train], X[test]
		y_train, y_test = [Y[i] for i in train], Y[test]
		clf.fit(X_train, y_train)
		y_hat = clf.predict(X_test)[0]
		print y_hat, y_test
		if y_hat != y_test: score += 1
	return float(score) / len(X)

# returns a list |X| of lists node features |phi_x|, a map from id_node 
# to corresponding index for the feature representation of that node in X,
# and a list |Y| of labels
def extract_node_features(nodes):
	X = []
	Y = []
	index_map = {}
	gdb = GraphDatabase('http://ec2-54-187-76-157.us-west-2.compute.amazonaws.com:7474/db/data/')
	for i, node in enumerate(nodes):
		# phi = [handle_length, num_non_alpha, num_links]
		phi = []
		node_handle = node['node_handle']
		# handle_length
		phi.append(len(node_handle))
		# num_non_alpha characters
		phi.append(len([c for c in node_handle if not c.isalpha()]))
		q = 'MATCH (n{handle:' + node_handle + '})-[r]-(x) RETURN r, n, x'
		links = gdb.query(q=q)
		# num_links
		phi.append(len(links))
		action_type = node['action_type']
		# binary classification, 'GOOD_NODE' = 1
		if action_type == "'GOOD_NODE'":
			Y.append(1)
		else:
			Y.append(0)
		index_map[node['id_node']] = i
		X.append(phi)
	return X, Y, index_map

# executed by python graph_clf_features.py
nodes, links = ingest_graph_feedback()
X, Y, index_map = extract_node_features(nodes)
svm_loocv_error = loocv_clf(X,Y)
clf = MultinomialNB()
nb_loocv_error = loocv_clf(X, Y, clf)
print svm_loocv_error
print nb_loocv_error
