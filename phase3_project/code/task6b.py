from collections import OrderedDict
import constants
import numpy as np
import pickle
import re
import scipy.sparse as sparse
from task4 import Task4
from util import Util


class Task6b():
	def __init__(self):
		self.ut = Util()
		self.ppr = Task4()
		self.image_id_mapping = None

	def get_normalized_seed_vector(self,seed_vector,graph_len):
		"""
		Method : Returns the normalized seed vector such the norm of the vector is 1
		seed_vector : Vector initialized with 1 for a labeled image
		graph_len : Length of the graph
		"""
		seed_vector = self.ppr.initialize_vq(seed_vector,graph_len)

		#normalize seed_vector
		seed_vector = [i/np.linalg.norm(seed_vector) for i in seed_vector]

		return np.array(seed_vector)

	def ppr_classifier(self, graph, image_label_map):
		"""
		Algorithm to classify images based on PPR
		1. Fetch seeds to compute personalized pagerank for each label
		2. Find pagerank vector for each label
		3. Argmax for finding label for the unclassified instace
		"""
		seed_label_map = self.fetch_seeds(image_label_map)
		label_ppr_map = {}
		graph_len = len(graph)
		M = self.ppr.normalize_M(graph)
		for label, seed_list in seed_label_map.items():
			seed_vector = self.get_normalized_seed_vector(seed_list,graph_len)
			rank_vector = seed_vector
			rank_vector = self.ppr.converge(rank_vector, seed_vector, M)
			label_ppr_map[label] = np.array(rank_vector)
		classified_image_label_map = self.classify(label_ppr_map)
		return classified_image_label_map

	def fetch_seeds(self,image_label_map):
		"""
		Method : Returns the map (image-> <list of labels associated with the image>)
		image_label_map : Map of (image -> label)
		"""
		seed_list_map = {}
		for key, value in image_label_map.items():
			if value in seed_list_map.keys():
				seed_list_map[value].append(key)
			else:
				seed_list_map[value] = [key]
		return seed_list_map

	def get_label_index_map(self,labels):
		"""
		Method: Returns the map of (label_value -> label_iterator) given
		the list of labels
		"""
		label_index_map = {label:i for i,label in enumerate(labels)}
		return label_index_map

	def get_labels_from_indexes(self,label_indexes,index_label_map):
		"""
		Method : Returns the list of labels given the label indexes from the index label map
		"""
		label_list = []

		for iter in label_indexes:
			label_list.append(index_label_map[iter])

		return label_list


	def classify(self, label_ppr_map):
		"""
		Method: Returns the map of classified images (image-> label) given
		the map of( label-> personalized page rank vector)
		label_ppr_map : Map of (label -> personalized page rank vector)
		"""
		label_list = label_ppr_map.keys()
		pagerank_vectors = label_ppr_map.values()

		#stacking the page rank vectors vertically
		pagerank_matrix = np.vstack(pagerank_vectors)
		label_index_map = self.get_label_index_map(label_list)
		index_label_map = dict((v,k) for k,v in label_index_map.items())

		image_label_matrix = pagerank_matrix.T
		classified_image_label_map = {}

		for i,v in enumerate(image_label_matrix):
			max_score = np.amax(v)

			#Gets the indexes(for labels) where the score is maximum, accounts for multiple max score
			label_indexes = np.argwhere(v == max_score).flatten().tolist()
			if max_score == 0:
				#Assigning the first label where 0 has occured.
				label_indexes = [label_indexes[0]]

			#Gets the labels for the obtained indexes
			computed_labels = self.get_labels_from_indexes(label_indexes,index_label_map)

			classified_image_label_map[i] = (computed_labels,max_score)

		return classified_image_label_map

	def pretty_print(self,label_image_map):
		op = open(constants.TASK6b_OUTPUT_FILE, "w")
		count = 0

		for label, image_scores in label_image_map.items():
			count += 1
			print("Label " + str(count) + "\n ########################## \n")
			op.write("Label " + label + "\n")

			image_scores = sorted(image_scores,key=lambda x:x[1],reverse=True)

			sorted_image_ids = [im[0] for im in image_scores]

			id_image_mapping = { y:x for x,y in self.image_id_mapping.items() }

			ids = [id_image_mapping[image_id] for image_id in sorted_image_ids]
			for temp in ids:
				op.write(temp + "\n")
			op.write("####\n")

	def runner(self):
		try:
			image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
			self.image_id_mapping = pickle.load(image_id_mapping_file)[1]
			image_label_map = OrderedDict({})

			f = open(constants.TASK6_INPUT_FILE1,"r")
			file_content = f.readlines()[2:]

			for row in file_content:
				row_entry = row.split(" ")
				row_entry = [item.strip() for item in row_entry if re.match('\\W?\\w+', item)]
				image_id = self.image_id_mapping[row_entry[0]]
				label = row_entry[1]
				image_label_map[image_id] = label

			initial_k = int(input("Enter the initial value of k: "))
			graph = self.ut.create_adj_mat_from_red_file(initial_k, True)
			classified_image_label_map = self.ppr_classifier(graph, image_label_map)

			label_image_map = {}

			for k,v in classified_image_label_map.items():
				for i in v[0]:
					if i in label_image_map:
						label_image_map[i].append((k,v[1]))
					else:
						label_image_map[i] = [(k,v[1])]

			self.pretty_print(label_image_map)

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))