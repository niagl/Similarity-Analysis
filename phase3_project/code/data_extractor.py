"""
This module contains data parsing methods.
"""
from collections import OrderedDict
import constants
import xml.etree.ElementTree as et

class DataExtractor(object): 
	def location_mapping(self):
		#parse the xml file of the locations
		tree = et.parse(constants.DEVSET_TOPICS_DIR_PATH)
		#get the root tag of the xml file
		doc = tree.getroot()
		mapping = OrderedDict({})
		#map the location id(number) with the location name
		for topic in doc:
			mapping[topic.find("number").text] = topic.find("title").text

		return mapping

	def prepare_dataset_for_task1(self, mapping):
		"""
		Method: Combining all the images across locations.
		"""
		locations = list(mapping.values())
		image_feature_map = OrderedDict({})
		model = constants.VISUAL_DESCRIPTOR_MODEL_FOR_GRAPH_CREATION

		for location in locations:
			location_model_file = location + " " + model + ".csv"
			data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()

			for row in data:
				row_data = row.strip().split(",")
				feature_values = list(map(float, row_data[1:]))
				image_id = row_data[0]
				image_feature_map[image_id] = feature_values

		return image_feature_map

	def prepare_dataset_for_task6(self, mapping):
		"""
		Method: Combining all the images across locations.
		"""
		locations = list(mapping.values())
		image_feature_map = OrderedDict({})
		models = constants.MODELS_6A

		for location in locations:
			for model in models:
				location_model_file = location + " " + model + ".csv"
				data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()

				for row in data:
					row_data = row.strip().split(",")
					feature_values = list(map(float, row_data[1:]))
					image_id = row_data[0]
					if image_id in image_feature_map:
						image_feature_map[str(image_id)].extend(feature_values)
					else:
						image_feature_map[str(image_id)] = feature_values

		return image_feature_map