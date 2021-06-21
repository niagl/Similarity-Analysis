import constants
import pickle
import random
from util import Util

class Task2b():
	def __init__(self):
		self.ut = Util()

	def max_a_min_partitioning(self, graph, c, k):
		"""
		1. fix random object as leader 1
		2. select c-1 farthest points from these leaders
		3. perform pass iteration - add images closest to a leader to cluster represented by the leader
		"""
		leaders = self.leader_selection(graph, c, k)
		clusters_heads = self.leader_fixation(leaders)
		final_clusters = self.pass_iteration(clusters_heads, graph)
		return final_clusters

	def leader_selection(self, graph, c, k):
		"""
		1. first cluster head randomly selected.
		2. remaining cluster heads should be unique and equal to the number of clusters passed.
		"""
		leaders = []
		leaders.append(random.randint(0,len(graph))) #initialize first leader
		for i in range(c-1):
			temp_row = sorted(graph[leaders[i]], reverse=True)
			leader = graph[leaders[i]].index(temp_row[k-2])
			count = 3
			while(count < k):
				leader = graph[leaders[i]].index(temp_row[k-count])
				if leader not in leaders:
					break 
				count += 1
			leaders.append(leader)

		return leaders

	def leader_fixation(self, leaders):
		clusters = {}
		for leader in leaders:
			clusters[leader] = []

		return clusters

	def pass_iteration(self, clusters, graph):
		"""
		Assign each node in the graph to cluster based on similarity score. Also, we maintain a balance within the
		clusters. Each pair of clusters can have maximum difference of 100 nodes.
		"""
		leaders = list(clusters.keys())
		dict_graph = self.ut.fetch_dict_graph()
		for image_iter in range(len(graph)):
			image_out_links = graph[image_iter]
			leader_image_sim_list = [dict_graph[image_iter][(image_iter, leader)] for leader in leaders]
			cluster_head = leader_image_sim_list.index(max(leader_image_sim_list))
			new_cluster = self.preserve_cluster_balance(clusters)
			if new_cluster == -1:
				clusters[leaders[cluster_head]].append(image_iter)
			else:
				index = leaders.index(new_cluster)
				clusters[leaders[index]].append(image_iter)

		return clusters

	def preserve_cluster_balance(self, clusters):
		"""
		returns -1 if a given node is not a cluster head already.
		returns the new cluster head ensuring that balance between clusters is maintained, controlled by
		inter_cluster_threshold.
		"""
		inter_cluster_threshold = 100
		cluster_length_map = {}
		for key, value in clusters.items():
			cluster_length_map[key] = len(value)
		cluster_lengths = list(cluster_length_map.values())
		cluster_lengths.sort()

		if cluster_lengths[-1] - cluster_lengths[0] > inter_cluster_threshold:
			for key, value in clusters.items():
				if len(value) == cluster_lengths[0]:
					return key

		return -1

	def pretty_print(self, c_clusters):
		"""
		prints output to terminal and writes to a file, which then is visualized using the presentation layer.
		"""
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]
		id_image_mapping = { y:x for x,y in image_id_mapping.items() }
		count = 0

		op = open(constants.TASK2b_OUTPUT_FILE, "w")
	
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
			c_clusters = self.max_a_min_partitioning(graph, c, initial_k)
			self.pretty_print(c_clusters)
		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))