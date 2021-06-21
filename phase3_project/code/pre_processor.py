"""
This module contains data preprocessing and data parsing methods.
This would be run before starting our driver program. It ensures that the raw dataset of visual descriptors is processed
and stored under processed directory.
This module contains one time tasks for pre-processing of data.
"""
from collections import OrderedDict
import constants
from data_extractor import DataExtractor
import glob
import pandas as pd
import pickle
from task1 import Task1

class PreProcessor(object): 
	def __init__(self):
		self.models = constants.MODELS
		data_extractor = DataExtractor()
		mapping = data_extractor.location_mapping()
		self.locations = list(mapping.values())
		self.task1  = Task1()

	def pre_process(self):
		"""
		Any other preprocessing needed can be called from pre_process method.
		"""
		self.remove_duplicates_from_visual_descriptor_dataset()
		self.rename_image_ids_from_visual_descriptor_dataset()
		self.add_missing_objects_to_dataset()
		self.transform_graph_file_to_dict_graph()
		# self.transform_edgelist_to_list_of_list_graph() -> not used

	def remove_duplicates_from_visual_descriptor_dataset(self):
		"""
		Method: remove_duplicates_from_visual_descriptor_dataset removes any duplicate image id in a model file
		"""

		files = glob.glob(constants.VISUAL_DESCRIPTORS_DIR_PATH_REGEX)
		for file in files:
			raw_file_contents = open(file, "r").readlines()
			global_image_ids = []
			file_name = file.split("/")[-1]
			output_file = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + file_name, "w")
			for row in raw_file_contents:
				image_id = row.split(",")[0]
				if image_id not in global_image_ids:
					output_file.write(row)

	def rename_image_ids_from_visual_descriptor_dataset(self):
		"""
		Method: rename_image_ids_from_visual_descriptor_dataset finds out if there are similar image ids across two
		locations and updates image id of one of the locations.
		FIX: 1) In case more than 1 location has same image ids with respect to a location, this method can"t handle such
		updates. (will need volunteer for fixing this :P)
		"""

		global_image_map = OrderedDict({})

		for location in self.locations:
			location_model_image_ids = []
			for model in self.models:
				location_model_file = location + " " + model + ".csv"
				data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()
				location_model_image_ids += [row.split(",")[0] for row in data]
			global_image_map[location] = list(set(location_model_image_ids))

		location_files_to_be_cleaned = []
		dataset = list(global_image_map.values())

		for iterator1 in range(0, len(dataset)):
			for iterator2 in range(iterator1+1, len(dataset)):
				common_image_ids = set(dataset[iterator1]).intersection(dataset[iterator2])
				if len(common_image_ids) > 0:
					location_files_to_be_cleaned.append([self.locations[iterator1], self.locations[iterator2],
														 common_image_ids])

		for iterator in location_files_to_be_cleaned:
			for model in self.models:
				location_model_file = iterator[1] + " " + model + ".csv"
				data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()
				output_file = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "w")
				for row in data:
					values = row.split(",")
					if values[0] in iterator[2]:
						image_id = values[0] + "_1"
						row = image_id + "," + ",".join(values[1:])
					output_file.write(row)

	def add_missing_objects_to_dataset(self):
		"""
		Method: add_missing_objects_to_dataset finds out missing objects across locations and models and adds them to
		relevant files ensuring data consistency
		"""
		location_image_id_map = {}
		for location in self.locations:
			for model in self.models:
				location_model_file = location + " " + model + ".csv"
				file = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r")
				data = file.readlines()
				file.close()
				location_model_image_ids = [row.split(",")[0] for row in data]
				for image_id in location_model_image_ids:
					if location in location_image_id_map.keys():
						if image_id not in location_image_id_map[location]:
							location_image_id_map[location].append(image_id)
					else:
						location_image_id_map[location] = [image_id]

		for location in self.locations:
			for model in self.models:
				location_model_file = location + " " + model + ".csv"
				data = pd.read_csv(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, dtype = \
						{0:"str"}, header=None)
				location_model_image_ids = data[0]
				missing_image_ids = set(location_image_id_map[location]) -\
									set(location_model_image_ids).intersection(set(location_image_id_map[location]))
				if len(missing_image_ids) > 0:
					file_des = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "a")
					for image_id in missing_image_ids:
						feature_values = [str(data[iterator].mean()) for iterator in range(1, len(data.columns))]
						print("New object inserted : " + str(image_id) + " in " + location_model_file)
						file_des.write(str(image_id) + "," + ",".join(feature_values) + "\n")

	def transform_edgelist_to_list_of_list_graph(self):
		"""
		Note: works for reduced graph.
		"""
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]
		graph_list = []
		with open(constants.ENTIRE_GRAPH_FILE, "r") as graph_file:
			image1 = ""
			for line in graph_file:
				temp = line.replace("\n", "").split(" ")
				if temp[0] != image1:
					image1 = temp[0]
					edges_list = []
					edges_list.append((image_id_mapping[temp[0]], image_id_mapping[temp[1]], float(temp[2])))
					graph_list.append(edges_list)
				else:
					graph_list[-1].append((image_id_mapping[temp[0]], image_id_mapping[temp[1]], float(temp[2])))

		list_of_list_graph_pickle_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "list_of_list_graph.pickle", "wb")
		pickle.dump(graph_list, list_of_list_graph_pickle_file)
		list_of_list_graph_pickle_file.close()	

	def transform_graph_file_to_dict_graph(self):
		"""
		Note: This method generates a pickle to be used in task2b.
		"""
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]
		graph_dict = []
		with open(constants.ENTIRE_GRAPH_FILE, "r") as graph_file:
			image1 = ""
			cnt = -1
			for line in graph_file:
				temp = line.replace("\n", "").split(" ")
				if temp[0] != image1:
					image1 = temp[0]
					cnt += 1
					edges_dict = {(image_id_mapping[temp[0]], image_id_mapping[temp[1]]): float(temp[2])}
					graph_dict.append(edges_dict)
				else:
					graph_dict[cnt].update({(image_id_mapping[temp[0]], image_id_mapping[temp[1]]): float(temp[2])})

		graph_dict_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "entire_graph_dict.pickle", "wb")

		pickle.dump(["Object", graph_dict], graph_dict_file)
		graph_dict_file.close()

object = PreProcessor()
object.pre_process()