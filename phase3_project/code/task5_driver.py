from task5_hash_table import Task5HashTable
from task5_LSH import Task5LSH
import constants
import webbrowser, os
from urllib.request import pathname2url

class Task5Driver:
	def __init__(self):
		self.L = int()
		self.k = int()
		self.lsh = None
		self.query_imageid = ''
		self.t = int()

	def runner(self):
		self.L = int(input('Enter the number of layers (L): '))
		self.k = int(input('Enter the number of Hashes per layer (k): '))
		self.lsh = Task5LSH(self.L, self.k)

		for table_instance in self.lsh.hash_tables:
			print('Number of hash codes/buckets for the given layer: ', len(list(table_instance.hash_table.keys()))) #, ' Max size of any given bucket: ', max(list(table_instance.hash_table.values())))
			print('------------')
		print('')

		t_nearest_neighbors = list()
		returned_dict = dict()
		
		while True:
			self.query_imageid = int(input('Enter the image ID: '))
			self.t = int(input('Enter the number of nearest neighbors desired (t): '))
			returned_dict = self.lsh.get_atleast_t_candidate_nearest_neighbors(self.query_imageid, self.t)
			print('Total images considered: ', returned_dict['total_images_considered'])
			print('Unique images considered: ', returned_dict['unique_images_considered'])

			nearest_neighbors_list = self.lsh.get_t_nearest_neighbors(self.query_imageid, returned_dict['result_list'], self.t)
			for nearest_neighbor in nearest_neighbors_list: # Get the image IDs alone
				t_nearest_neighbors.append(nearest_neighbor['image_id'])

			op = open(constants.TASK5_OUTPUT_FILE, "w")
			op.truncate(0)
			op.write("The T nearest neighbor images for the query image (first one from the left) are:\n")
			op.write(str(self.query_imageid))
			op.write("\n")
			for image_id in t_nearest_neighbors:
				op.write(str(image_id))
				op.write("\n")
			
			url = 'file:{}'.format(pathname2url(os.path.abspath('../output/task5.html')))
			webbrowser.open(url)

			op.close()

			print('The T nearest neighbors: ', t_nearest_neighbors, '\n')
			runagain = input('Run again? (Y/N): ')
			t_nearest_neighbors.clear()
			returned_dict.clear()
			if runagain == 'N':
				break