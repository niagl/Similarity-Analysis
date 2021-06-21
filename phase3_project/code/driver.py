"""
This module is a driver program which selects a particular task.
"""
from task1 import Task1
from task2a import Task2a
from task2b import Task2b
from task3 import Task3
from task4 import Task4
from task5_driver import Task5Driver
from task6a import Task6a
from task6b import Task6b

class Driver():

	def input_task_num(self):
		print("Tasks: 1, 2a(Angular), 2b(Max-a-min), 3(PageRank), 4(PPR), 5(LSH), 6a, 6b")
		task_num = input("Enter the Task no.: ")
		self.select_task(task_num)

	def select_task(self, task_num):
		# Plugin class names for each task here
		tasks = { "1": Task1(), "2a": Task2a(),  "2b": Task2b(), "3": Task3(), "4": Task4(), "5": Task5Driver(), "6a":Task6a(), "6b":Task6b()}
		# Have a runner method in all the task classes
		tasks.get(task_num).runner()

flag = True
while(flag):
	choice = int(input("Enter your choice:\t1) Execute tasks \t2) Exit\n"))

	if choice == 2:
		flag = False
	else:
		t = Driver()
		t.input_task_num()