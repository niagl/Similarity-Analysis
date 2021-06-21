import constants
import numpy as np
import pickle
import scipy.sparse as sparse
from util import Util

class Task3():
	def __init__(self):
		self.ut = Util()
		self.d = 0.85

	def pagerank(self, graph, K, seeds=[]):
		"""
		Pagerank is computed using iterative approach.
		Initialize a pagerank vector with uniformly distributed probabilities to ensure each node gets a chance of being
		randomly visited.
		Compute pagerank score for each node which is defined in terms of all the nodes pointing to the current node.
		If node A is pointed by node B and node C, then pagerank create_adj_mat_from_red_filer
		node A = (1 - d) + d * ( (pagerank(B)/outdegree(B)) + (pagerank(C)/out_degree_listtdegree(C)) ).
		Here d is damping factor with value 0.85.
		Continue this over for a number of times, default set to 50 or until two consecutive iterations yield same
		results.
		"""
		pagerank_vector = self.initialize_pagerank_vector(graph, seeds)
		graph_transpose = np.array(graph).transpose()
		out_degree_list = self.calculate_node_outdegree(graph)
		pointing_nodes_list = self.derive_pointing_nodes_list(graph_transpose)

		final_pagerank_vector = self.converge(pagerank_vector, out_degree_list, pointing_nodes_list)
		self.top_k(final_pagerank_vector, K)

	def initialize_pagerank_vector(self, graph, seeds):
		"""
		Initialize pagerank_vector with uniform probability distribution.
		"""
		return [1.0]*len(graph)

	def calculate_node_outdegree(self, graph):
		return [sum(row) for row in graph]

	def derive_pointing_nodes_list(self, graph):
		"""
		returns incoming nodes for a given node in the graph.
		"""
		pointing_nodes_list = []
		for row in graph:
			local_pointing_nodes_list = []
			for iter in range(len(row)):
				if row[iter] == 1:
					local_pointing_nodes_list.append(iter)
			pointing_nodes_list.append(local_pointing_nodes_list)
		return pointing_nodes_list

	def converge(self, pagerank_vector, out_degree_list, pointing_nodes_list, default_iterations=50):
		"""
		We have kept converging factor as default iterations of 50.
		"""
		iterations = 0
		pg_vectors = [pagerank_vector]
		while(iterations < default_iterations):
			pg_vector = [0]*len(pg_vectors[-1])
			for node in range(len(pg_vectors[0])):
				nodes_with_incoming_edges = pointing_nodes_list[node]
				right_operand = self.random_walk(nodes_with_incoming_edges, out_degree_list, pg_vectors[-1])
				pg_vector[node] = (1-self.d) + np.multiply(self.d, right_operand)
			pg_vectors.append(pg_vector)
			iterations += 1
		return pg_vectors[-1]

	def random_walk(self, nodes_with_incoming_edges, out_degree_list, pg_vector):
		sum_random_nodes = 0
		for node in nodes_with_incoming_edges:
			sum_random_nodes += (pg_vector[node] * 1.0)/out_degree_list[node]
		return sum_random_nodes

	def top_k(self, pagerank_score, K):
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]

		image_id_score_mapping = {}

		for iter in range(0, len(pagerank_score)):
			for image_id, index in image_id_mapping.items():
				if index == iter:
					image_id_score_mapping[image_id] = pagerank_score[iter]
		print("Top K images based on pagerank score\n")
		op = open(constants.TASK3_OUTPUT_FILE, "w")
		op.write("K most dominant images are:\n")
		for image_id, score in sorted(image_id_score_mapping.items(), key=lambda x: x[1], reverse=True)[:K]:
			op.write(str(image_id) + " " + str(round(score, 4)))
			op.write("\n")

		print(sorted(image_id_score_mapping.items(), key=lambda x: x[1], reverse=True)[:K])

	def runner(self):
		try:
			image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
			image_id_mapping = pickle.load(image_id_mapping_file)[1]
			seeds = []
			K = int(input("Enter the value of K: "))
			initial_k = int(input("Enter the initial value of k: "))

			graph = self.ut.create_adj_mat_from_red_file(initial_k)
			k_dominant_images = self.pagerank(graph, K, seeds)

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))