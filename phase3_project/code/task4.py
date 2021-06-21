import constants
import numpy as np
import pickle
import scipy.sparse as sparse
from util import Util

class Task4():
	def __init__(self):
		self.ut = Util()
		self.d = 0.85

	def personalised_pagerank(self, graph, seeds=[]):
		"""
		Algorithm to compute PPR
		1. Let vq=0, for all its N entries, except a ’1’ for the q-th entry.
		2. Normalize the adjacency matrix of A(graph), by column. That is, make each column sum to 1.
		3. Initialize uq=vq
		4. while(uq has not converged)
		4.1 uq = (1-c)*A*uq + c*vq
		"""
		vq = self.initialize_vq(seeds, len(graph))
		uq = vq
		M = self.normalize_M(graph)
		uq = self.converge(uq, vq, M)
		return uq

	def initialize_vq(self, seeds, graph_len):
		vq = [0]*graph_len
		for iter in seeds:
			vq[iter] = 1.0
		return vq

	def normalize_M(self, graph):
		graph_transpose = zip(*graph)
		new_graph = []
		for row in graph_transpose:
			sum_row = sum(row)
			if sum_row == 0:
				new_graph.append(row)
				continue

			new_graph.append([value/sum_row for value in row])
		return np.transpose(new_graph)

	def converge(self, uq, vq, M):
		uq = self.compute_uq(uq, vq, M)
		uq_list = []
		uq_list.append(uq)
		converged = False
		count = 0
		while(count < 50):
			count += 1
			uq = self.compute_uq(uq_list[-1], vq, M)
			uq_list.append(uq)
		return uq_list[-1]

	def compute_uq(self, uq, vq, M):
		left_operand = np.matmul(M, uq)
		right_operand = np.multiply(vq, self.d)
		uq = np.multiply((1-self.d), left_operand) + right_operand
		return uq

	def top_k(self, pagerank_score, K):
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]

		image_id_score_mapping = {}

		for iter in range(0, len(pagerank_score)):
			for image_id, index in image_id_mapping.items():
				if index == iter:
					image_id_score_mapping[image_id] = pagerank_score[iter]
		print("Top K images based on pagerank score\n")
		op = open(constants.TASK4_OUTPUT_FILE, "w")
		op.write("K most dominant images are (First 3 images are the seeds):\n")
		for image_id, score in sorted(image_id_score_mapping.items(), key=lambda x: x[1], reverse=True)[:K + 3]:
			op.write(str(image_id) + " " + str(round(score, 4)))
			op.write("\n")

		print(sorted(image_id_score_mapping.items(), key=lambda x: x[1], reverse=True)[:K + 3])

	def runner(self):
		try:
			image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
			image_id_mapping = pickle.load(image_id_mapping_file)[1]
			seeds = []
			K = int(input("Enter the value of K: "))
			initial_k = int(input("Enter the initial value of k: "))
			print("Enter three image ids to compute PPR:\n")
			image_id1 = input("Image id1:")
			seeds.append(image_id_mapping[image_id1])
			image_id2 = input("Image id2:")
			seeds.append(image_id_mapping[image_id2])
			image_id3 = input("Image id3:")
			seeds.append(image_id_mapping[image_id3])

			graph = self.ut.create_adj_mat_from_red_file(initial_k, True)
			personalized_pagerank_vector = self.personalised_pagerank(graph, seeds)
			self.top_k(personalized_pagerank_vector, K)

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))