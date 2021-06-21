import constants
import pickle
import scipy
from util import Util

class Task2a():
	def __init__(self):
		self.ut = Util()

	def anglular_clustering(self, graph, c):
		"""
		1. perform SVD on the adjacency matrix.
		2. find top k singular vectors corresponding to the largest eigen values.
		3. k eigen vectors form the clusters.
		4. assign each node to the cluster by finding max value in the row for these vectors.
		"""
		top_singular_vector_matrix = self.fetch_singular_vectors(graph, c)
		c_clusters = self.partition(top_singular_vector_matrix)
		return c_clusters
	
	def fetch_singular_vectors(self, graph, c):
		"""
		returns top c singular vectors for the similarity graph passed.
		"""
		sparse_matrix = scipy.sparse.csc_matrix(graph, dtype=float)
		u, s, vt = scipy.sparse.linalg.svds(graph, k=c)
		return u

	def partition(self, top_singular_vector_matrix):
		"""
		form partitions using max eigen value found in the singular matrix for each node.
		"""
		clusters = {}
		for iter in range(len(top_singular_vector_matrix[0])):
			clusters[iter] = []

		for node in range(len(top_singular_vector_matrix)):
			max_value = max(top_singular_vector_matrix[node])
			index = list(top_singular_vector_matrix[node]).index(max_value)
			clusters[index].append(node)

		return clusters

	def pretty_print(self, c_clusters):
		"""
		prints to terminal and writes to file given as input to the visualization module.
		"""
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]
		id_image_mapping = { y:x for x,y in image_id_mapping.items() }
		count = 0

		op = open(constants.TASK2a_OUTPUT_FILE, "w")

		for cluster, image_ids in c_clusters.items():
			count += 1
			print("Cluster " + str(count) + "\n ########################## \n")
			op.write("Cluster " + str(count) + "\n")

			ids = [id_image_mapping[image_id] for image_id in image_ids]
			for temp in ids:
				op.write(temp + "\n")
			op.write("####\n")
			
			print("Cluster head: " + str(id_image_mapping[cluster]) + "\n" + "Clustering: " + str(ids) + "\n")

	def runner(self):
		try:
			initial_k = int(input("Enter the initial value of k: "))
			c = int(input("Enter the value of c (number of clusters): "))
			graph = self.ut.create_adj_mat_from_red_file(initial_k, True)
			c_clusters = self.anglular_clustering(graph, c)
			self.pretty_print(c_clusters)
		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))