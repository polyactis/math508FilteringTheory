#!/usr/bin/env python
""""
Usage: math508.py -y TestCaseType [OPTIONS]

Option:
	-y ..., --type=...	which test case should be invoked.
	-h, --help              show this help

Examples:
	math508.py -y 2

2007-01-26
	Problems from Math508, Spring 2007, "Filtering Theory" by Prof Remigijus Mikulevicius
1: Math508_HW2_1
2: Math508_HW2_2
"""
import sys, os, math
bit_number = math.log(sys.maxint)/math.log(2)
if bit_number>40:       #64bit
	sys.path.insert(0, os.path.expanduser('~/lib64/python'))
	sys.path.insert(0, os.path.join(os.path.expanduser('~/script64/annot/bin')))
else:   #32bit
	sys.path.insert(0, os.path.expanduser('~/lib/python'))
	sys.path.insert(0, os.path.join(os.path.expanduser('~/script/annot/bin')))
import unittest, os, sys, getopt, csv

class Math508_HW2_1(unittest.TestCase):
	"""
	2007-01-26
		Homework 2, No 1 of Math508 (Filtering Theory)
	"""
	def setUp(self):
		print
	
	def plot_trace(self, trace_list, N, title):
		import pylab
		pylab.clf()
		x_list = []
		y_list = []
		for row in trace_list:
			x_list.append(row[0])
			y_list.append(row[1])
		pylab.plot(x_list, y_list, '-')
		pylab.title(title)
		pylab.show()
	
	def vector_plus(self, vector1, vector2):
		vector3 = []
		for i in range(len(vector1)):
			vector3.append(vector1[i] + vector2[i])
		return vector3
	
	def test_hw2_1_a(self):
		import random
		N = raw_input("Please specify N (#iterations):")
		if N:
			N = int(N)
		else:
			N = 100
		trace_list = [[0,0]]
		i = 0
		while i<N:
			u = random.random()
			if u>=0 and u<0.25:
				step = [1, 0]
			elif u>=0.25 and u<0.5:
				step = [-1, 0]
			elif u>=0.5 and u<0.75:
				step = [0, 1]
			else:
				step = [0, -1]
			trace_list.append(self.vector_plus(trace_list[-1], step))
			i += 1
		self.plot_trace(trace_list, N, "HW2-1-a, simple r.w. on Z^2. N=%s"%(N))
	
	def test_hw2_1_b(self):
		import random
		N = raw_input("Please specify N (#iterations):")
		if N:
			N = int(N)
		else:
			N = 100
		trace_list = [[0,0]]
		i = 0
		while i<N:
			u = random.random()
			if u>=0 and u<0.2:
				step = [1, 0]
			elif u>=0.2 and u<0.5:
				step = [-1, 0]
			elif u>=0.5 and u<0.75:
				step = [0, 1]
			else:
				step = [0, -1]
			trace_list.append(self.vector_plus(trace_list[-1], step))
			i += 1
		self.plot_trace(trace_list, N, "HW2-1-b, non-simple r.w. on Z^2. N=%s"%(N))

class Math508_HW2_2(unittest.TestCase):
	"""
	2007-01-26
		Homework 2, No 2 of Math508 (Filtering Theory)
	"""
	def setUp(self):
		print
	
	def plot_TL_list(self, TL_list, title):
		import pylab
		pylab.clf()
		x_list = range(1, len(TL_list)+1)
		pylab.plot(x_list, TL_list, 'o')
		pylab.title(title)
		pylab.show()
		pylab.clf()
		pylab.hist(TL_list, 100)
		pylab.show()
	
	def test_hw2_2(self):
		import random
		m = raw_input("Please specify m (#trajectories):")
		if m:
			m = int(m)
		else:
			m = 20
		L = raw_input("Please specify L (maximum length of trajectory):")
		if L:
			L = int(L)
		else:
			L = 100
		TL_list = []
		i = 0
		while i<m:
			X = 0
			j = 0
			while j<L:
				u = random.random()
				if u>=0 and u<0.5:
					step = 1
				else:
					step = -1
				X += step
				if X==0:
					break
				j+=1
			TL_list.append(j+1)
			i+=1
		self.plot_TL_list(TL_list, "HW2-2, T_L=min{T, L} of simple r.w. on Z. m=%s, L=%s"%(m, L))

if __name__ == '__main__':
	if len(sys.argv) == 1:
		print __doc__
		sys.exit(2)
	
	long_options_list = ["help", "type="]
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hy:", long_options_list)
	except:
		print __doc__
		sys.exit(2)
	
	TestCaseDict = {1: Math508_HW2_1,
		2: Math508_HW2_2}
	type = 0
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print __doc__
			sys.exit(2)
		elif opt in ("-y", "--type"):
			type = int(arg)
			
	if type:
		suite = unittest.TestSuite()
		"""
		add one by one
		"""
		#suite.addTest(TestGeneStatPlot("test_return_distinct_functions"))
		#suite.addTest(TestGeneStatPlot("test_L1_match"))
		#suite.addTest(TestGeneStat("test__gene_stat_leave_one_out"))
		#suite.addTest(TestGeneStat("test_submit"))
		#suite.addTest(TestGeneStat("test_common_ancestor_deep_enough"))
		"""
		add all
		"""
		suite.addTest(unittest.makeSuite(TestCaseDict[type]))
		unittest.TextTestRunner(verbosity=2).run(suite)

	else:
		print __doc__
		sys.exit(2)
