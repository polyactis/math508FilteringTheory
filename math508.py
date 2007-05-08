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
3: Math508_HW3_1
4: Math508_HW5_1
5: Math508_HW5_2
6: Math508_HW6_1
7: Math508_HW6_2
8: Math508_HW7_1
9: Math508_HW7_2
10: Math508_HW8_a
11: Math508_HW8_b
12: Math508_HW10_1
13: Math508_HW10_2
14: Math508_HW11_3
15: Math508_HW12_3
16: Math508_Final_Exam_3
17: Math508_Final_Exam_4
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

class Math508_HW3_1(unittest.TestCase):
	"""
	2007-02-01
		Homework 3, No 1 of Math508 (Filtering Theory)
	"""
	def setUp(self):
		print
	
	def plot_TL_list(self, TL_list, title):
		import pylab
		pylab.clf()
		x_list = range(1, len(TL_list)+1)
		pylab.plot(x_list, TL_list, '-')
		pylab.title(title)
		pylab.show()
		pylab.clf()
		pylab.hist(TL_list, 100)
		pylab.show()
	
	def test_hw3_1_a(self):
		import random
		L = raw_input("Please specify L (maximum length of trajectory(default=200)):")
		if L:
			L = int(L)
		else:
			L = 200
		x = raw_input("Please specify x (the initial state (default=5)):")
		if x:
			x = int(x)
		else:
			x = 5
		initial_x = x
		TL_list = []
		j = 0
		while j<L:
			u = random.random()
			if x>0 and x<20:
				if u>=0 and u<0.5:
					step = 1
				else:
					step = -1
			elif x==0 or x==20:
				if u>=0 and u<0.5:
					step = 0
				else:
					step = int(x<10) - int(x>10)
			x += step
			j+=1
			TL_list.append(x)
		self.plot_TL_list(TL_list, "HW3-1, X0=%s, N=%s, simple reflected r.w. on Z[0,20]."%(initial_x, L))
	
	def test_hw3_1_b(self):
		import random
		L = raw_input("Please specify L (maximum length of trajectory(default=200)):")
		if L:
			L = int(L)
		else:
			L = 200
		x = raw_input("Please specify x (the initial state (default=5)):")
		if x:
			x = int(x)
		else:
			x = 5
		initial_x = x
		TL_list = []
		j = 0
		while j<L:
			if x>0 and x<20:
				u = random.random()
				if u>=0 and u<0.5:
					step = 1
				else:
					step = -1
				x += step
			j+=1
			TL_list.append(x)
		self.plot_TL_list(TL_list, "HW3-1, X0=%s, N=%s, simple r.w. on Z[0,20] with absorption."%(initial_x, L))

class Math508_HW5_1(unittest.TestCase):
	"""
	2007-02-15
	"""
	def setUp(self):
		print
	
	def plot_list(self, Xn_list, Xn_hat_list, Yn_list, title, figure_fname):
		import pylab
		pylab.clf()
		x_list = range(1, len(Xn_list)+1)
		pylab.plot(x_list, Xn_list, 'b')
		pylab.plot(x_list, Xn_hat_list, 'g')
		pylab.plot(x_list, Yn_list, 'r')
		pylab.title(title)
		label_list = ['Xn_list', 'Xn_hat_list', 'Yn_list']
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=150)
		pylab.savefig('%s.eps'%figure_fname, dpi=150)
		pylab.savefig('%s.png'%figure_fname, dpi=150)
		#pylab.show()
	
	def simulate_Xn_Yn(self, K=20, L_list=[1,3,5,10,15,18,20], chain_length=200, output_fname='hw5_1_simulation.out'):
		"""
		2007-02-22
			the chain is from 0 to chain_length, which is 201
		"""
		import csv, random
		writer = csv.writer(open(output_fname, 'w'))
		for L in L_list:
			Xn_list = []
			Yn_list = []
			for i in range(chain_length+1):
				if i==0:
					#Xi = random.randint(0, K)
					Xi = 10	#starting at 10
				else:
					x = Xn_list[i-1]
					u = random.random()
					if x>0 and x<20:
						if u>=0 and u<0.5:
							step = 1
						else:
							step = -1
					elif x==0 or x==20:
						if u>=0 and u<0.5:
							step = 0
						else:
							step = int(x<10) - int(x>10)
					Xi = x + step
				Xn_list.append(Xi)
				u = random.random()
				if u>=0 and u<0.5:
					W = L
				else:
					W = -L
				Yi = min(max(Xi+W, 0), 20)
				Yn_list.append(Yi)
			header_row = ['L', L]
			writer.writerow(header_row)
			writer.writerow(Xn_list)
			writer.writerow(Yn_list)
		del writer
		
	def initialize_P_Y_array(self, K, L):
		"""
		2007-02-15
			this is the de novo version
			assuming X_0 is uniform [0,K]
		"""
		import Numeric
		P_Y_array = Numeric.zeros(K+1, Numeric.Float)
		for i in range(K+1):
			P_Y_array[min(i+L, K)] += 1/(2*(K+1.0))	#this is where downward(+L) direction reaches
			P_Y_array[max(i-L, 0)] += 1/(2*(K+1.0))	#this is where upward(-L) direction reaches
		return P_Y_array
	
	def initialize_P_Y_array_non_denovo(self, P_Y_given_X_array):
		"""
		X_0 is uniform[0,K]
		"""
		import Numeric
		K = Numeric.shape(P_Y_given_X_array)[0]-1
		P_X_array = Numeric.ones(K+1, Numeric.Float)/(K+1.0)
		P_Y_X_array = Numeric.transpose(P_Y_given_X_array)*P_X_array
		P_Y_array = Numeric.matrixmultiply(P_Y_X_array, Numeric.ones(K+1))	#row sum
		return P_Y_array
	
	def initialize_P_Y_given_X_array(self, K, L):
		"""
		for P_Y_given_X_array, row is X, column is Y
		for P_Y_X_array, row is Y, column is X
		"""
		import Numeric
		P_Y_given_X_array = Numeric.zeros([K+1, K+1], Numeric.Float)
		for i in range(K+1):
			P_Y_given_X_array[i, max(i-L,0)] = 1/2.0
			P_Y_given_X_array[i, min(i+L,K)] = 1/2.0
		return P_Y_given_X_array
	
	def initialize_P_X_given_Y_array(self, P_Y_given_X_array, P_Y_array):
		import Numeric
		K = Numeric.shape(P_Y_array)[0]-1
		P_X_array = Numeric.ones(K+1, Numeric.Float)/(K+1.0)
		
		#calculate the joint probability
		P_X_Y_array = P_Y_given_X_array*P_X_array	#the 2d versus 1d operation is column-wise, it's not good for this but as P_X_array is all same, doesn't matter
		P_X_given_Y_array = Numeric.transpose(P_X_Y_array/P_Y_array)	#column-wise divide 
		return P_X_given_Y_array
	
	def estimate_Xn(self, K, Yn_list, L):
		import sys,os
		print 'Working on L=%s...'%L
		P_Y_array = self.initialize_P_Y_array(K, L)
		print 'P_Y_array'
		print P_Y_array
		P_Y_given_X_array = self.initialize_P_Y_given_X_array(K, L)
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		P_X_given_Y_array = self.initialize_P_X_given_Y_array(P_Y_given_X_array, P_Y_array)
		print 'P_X_given_Y_array'
		print P_X_given_Y_array
		P_Y_array_non_denovo = self.initialize_P_Y_array_non_denovo(P_Y_given_X_array)
		print 'P_Y_array_non_denovo'
		print P_Y_array_non_denovo
		import Numeric
		n = len(Yn_list)
		fi_array = Numeric.zeros([n, K+1], Numeric.Float)
		Xn_hat_list = []	#the final estimated Xn
		#calculate fi_0
		P_Y = P_Y_array[Yn_list[0]]	#
		numerator = 0.0
		denominator = 0.0
		for j in range(K+1):
			fi_array[0,j] = P_X_given_Y_array[Yn_list[0], j]*P_Y
			numerator += j*fi_array[0,j]
			denominator += fi_array[0,j]
		Xn_hat_list.append(numerator/denominator)
		#calculate fi_1, fi_2, ...
		for i in range(1, n):
			numerator = 0.0
			denominator = 0.0
			for j in range(K+1):
				if j==0:
					previous_X_candidate_list = [0,1]
				elif j==20:
					previous_X_candidate_list = [19,20]
				else:
					previous_X_candidate_list = [j-1, j+1]
				for previous_X_candidate in previous_X_candidate_list:
					fi_array[i, j] += P_Y_given_X_array[j, Yn_list[i]]*0.5*fi_array[i-1,previous_X_candidate]	#the transition prob of M.C. Xn is always 0.5
				numerator += j*fi_array[i, j]
				denominator += fi_array[i, j]
			Xn_hat_list.append(numerator/denominator)
		print 'Xn_hat_list'
		print Xn_hat_list
		print 'Done.\n'
		return Xn_hat_list
	
	def calculate_mse(self, Xn_list, Xn_hat_list):
		import Numeric
		diff_array = Numeric.array(Xn_list)-Numeric.array(Xn_hat_list)
		mse = sum(diff_array*diff_array)/(len(Xn_list))
		return mse
	
	def test_simple_rw_HMM(self):
		K=20; L_list=[1,3,5,10,15,18,20]; chain_length=200
		output_fname = raw_input("Please specify file containing/to contain data(default='hw5_1_simulation'):")
		if not output_fname:
			output_fname='hw5_1_simulation.out'
		simulate_yes = raw_input("Do you need to simulate data into the file(y/N)?")
		if not simulate_yes:
			simulate_yes='n'
		if simulate_yes=='y':
			self.simulate_Xn_Yn(K, L_list, chain_length, output_fname)
		
		import csv, Numeric
		reader = csv.reader(open(output_fname))
		row = reader.next()
		while row:
			if row[0]=='L':
				L = int(row[1])
				Xn_list = reader.next()
				Xn_list = map(int, Xn_list)
				Yn_list = reader.next()
				Yn_list = map(int, Yn_list)
				Xn_hat_list = self.estimate_Xn(K, Yn_list, L)
				mse = self.calculate_mse(Xn_list, Xn_hat_list)
				print 'mse=',mse
				title = 'simple reflected r.w. on A=[0,%s], P(W=+-%s)=1/2, n=%s, mse=%2.2f'%(K,L,chain_length, mse)
				figure_fname = 'hw5_1_K_%s_L_%s_n_%s'%(K,L,chain_length)
				self.plot_list(Xn_list, Xn_hat_list, Yn_list, title, figure_fname)
			row = reader.next()

class Math508_HW5_2(unittest.TestCase):
	"""
	2007-02-16
	"""
	def setUp(self):
		print
	
	def plot_list(self, Xn_hat_list, Yn_list, title, figure_fname):
		import pylab
		pylab.clf()
		x_list = range(1, len(Xn_hat_list)+1)
		pylab.plot(x_list, Xn_hat_list, 'g')
		pylab.title(title)
		label_list = ['Xn_hat_list']
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=150)
		pylab.savefig('%s.eps'%figure_fname, dpi=150)
		pylab.savefig('%s.png'%figure_fname, dpi=150)
		#pylab.show()
	
	def initialize_P_Y_array_non_denovo(self, P_Y_given_X_array, P_X_array):
		"""
		X_0 is uniform
		"""
		import Numeric
		no_of_X_states = P_Y_given_X_array.shape[0]
		P_Y_X_array = Numeric.transpose(P_Y_given_X_array)*P_X_array
		P_Y_array = Numeric.matrixmultiply(P_Y_X_array, Numeric.ones(no_of_X_states))	#row sum
		return P_Y_array
	
	def initialize_P_Y_given_X_array(self):
		"""
		for P_Y_given_X_array, row is X, column is Y
		for P_Y_X_array, row is Y, column is X
		"""
		import Numeric
		P_Y_given_X_array = Numeric.array([[0.3, 0.7], [0.5,0.5], [0.7,0.3]])
		return P_Y_given_X_array
	
	def initialize_P_X_given_Y_array(self, P_Y_given_X_array, P_Y_array, P_X_array):
		import Numeric
		#calculate the joint probability
		P_Y_X_array = Numeric.transpose(P_Y_given_X_array)*P_X_array
		P_X_given_Y_array = Numeric.transpose(Numeric.transpose(P_Y_X_array)/P_Y_array)	#column-wise divide 
		return P_X_given_Y_array
	
	def estimate_Xn(self, Yn_list, no_of_X_states=3):
		import sys,os
		import Numeric
		P_Y_given_X_array = self.initialize_P_Y_given_X_array()
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		P_X_array = Numeric.ones(no_of_X_states, Numeric.Float)/(float(no_of_X_states))
		P_Y_array = self.initialize_P_Y_array_non_denovo(P_Y_given_X_array, P_X_array)
		print 'P_Y_array'
		print P_Y_array
		P_X_given_Y_array = self.initialize_P_X_given_Y_array(P_Y_given_X_array, P_Y_array, P_X_array)
		print 'P_X_given_Y_array'
		print P_X_given_Y_array
		
		map_of_Y_state = {'H':0, 'T':1}
		
		n = len(Yn_list)
		fi_array = Numeric.zeros([n, no_of_X_states], Numeric.Float)
		pi_array = Numeric.zeros([n, no_of_X_states], Numeric.Float)
		Xn_hat_list = []	#the final estimated Xn
		#calculate fi_0
		P_Y_0 = P_Y_array[map_of_Y_state[Yn_list[0]]]	#
		numerator = 0.0
		denominator = 0.0
		for j in range(no_of_X_states):
			fi_array[0,j] = P_X_given_Y_array[map_of_Y_state[Yn_list[0]], j]*P_Y_0
			numerator += j*fi_array[0,j]
			denominator += fi_array[0,j]
		pi_array[0,] = fi_array[0,]/denominator
		Xn_hat_list.append(numerator/denominator)
		#calculate fi_1, fi_2, ...
		for i in range(1, n):
			numerator = 0.0
			denominator = 0.0
			for j in range(no_of_X_states):
				previous_X_candidate = j	#constant Xn
				fi_array[i, j] = P_Y_given_X_array[j, map_of_Y_state[Yn_list[i]]]*fi_array[i-1,previous_X_candidate]	#transition prob=1 for Xn
				numerator += j*fi_array[i, j]
				denominator += fi_array[i, j]
			pi_array[i,] = fi_array[i,]/denominator
			Xn_hat_list.append(numerator/denominator)
		print 'Xn_hat_list'
		print Xn_hat_list
		print 'fi_array'
		print fi_array
		print 'pi_array'
		print pi_array
		print 'Done.\n'
		return Xn_hat_list
	
	def test_simple_rw_HMM(self):
		import csv, Numeric
		Yn_list = ['H', 'H', 'H', 'T', 'T']
		no_of_X_states = 3
		Xn_hat_list = self.estimate_Xn(Yn_list, no_of_X_states)
		title = 'hw5-2 constant coin HMM'
		figure_fname = 'hw5_2_constant_coin_HMM'
		self.plot_list(Xn_hat_list, Yn_list, title, figure_fname)

class Math508_HW6_1(Math508_HW5_1):
	def initialize_P_X_given_X_array(self, K):
		import Numeric
		P_X_given_X_array = Numeric.zeros([K+1, K+1], Numeric.Float)
		for i in range(K+1):
			j = i-1
			if j<0:	#the boundary
				j = i
			P_X_given_X_array[i,j] = 1/2.0
			j = i+1
			if j>K:	#the boundary
				j = i
			P_X_given_X_array[i,j] = 1/2.0
		return P_X_given_X_array
		
	def backward_B_W(self, P_Y_given_X_array, P_X_given_X_array, Yn_list, n, T, K):
		import Numeric
		mu_array = Numeric.zeros([T-n+1, K+1], Numeric.Float)	#it's inverse, row 0 is mu_{T}, row 1 is mu_{T-1}, etc
		for j in range(K+1):
			mu_array[0,j] = 1
		for i in range(1, T-n+1):
			Yn_index = T -i +1
			for j in range(K+1):
				for k in range(K+1):
					mu_array[i, j] += P_Y_given_X_array[k,Yn_list[Yn_index]]*P_X_given_X_array[j, k]*mu_array[i-1, k]
		return mu_array
		
		
	def forward_B_W(self, Yn_list, K, P_Y_array, P_Y_given_X_array, P_X_given_X_array, P_X_given_Y_array, n):
		"""
		2007-02-22
			watch n+1, and K+1
		"""
		import Numeric
		fi_array = Numeric.zeros([n+1, K+1], Numeric.Float)
		#calculate fi_0
		P_Y = P_Y_array[Yn_list[0]]	#
		for j in range(K+1):
			fi_array[0,j] = P_X_given_Y_array[Yn_list[0], j]*P_Y
		#calculate fi_1, fi_2, ...
		for i in range(1, n+1):
			for j in range(K+1):
				for k in range(K+1):
					fi_array[i, j] += P_Y_given_X_array[j, Yn_list[i]]*P_X_given_X_array[k,j]*fi_array[i-1,k]
		return fi_array
	
	def estimate_X_hat_n_T(self, Yn_list, K, L, T_list, n):
		"""
		2007-02-22
			smoothing, n<T
		"""
		P_Y_array = self.initialize_P_Y_array(K, L)
		print 'P_Y_array'
		print P_Y_array
		P_Y_given_X_array = self.initialize_P_Y_given_X_array(K, L)
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		P_X_given_Y_array = self.initialize_P_X_given_Y_array(P_Y_given_X_array, P_Y_array)
		print 'P_X_given_Y_array'
		print P_X_given_Y_array
		P_Y_array_non_denovo = self.initialize_P_Y_array_non_denovo(P_Y_given_X_array)
		print 'P_Y_array_non_denovo'
		print P_Y_array_non_denovo
		P_X_given_X_array = self.initialize_P_X_given_X_array(K)
		print 'P_X_given_X_array'
		print P_X_given_X_array
		fi_array = self.forward_B_W(Yn_list, K, P_Y_array, P_Y_given_X_array, P_X_given_X_array, P_X_given_Y_array, n)
		print 'fi_array'
		print fi_array
		
		X_hat_n_T_list = []
		for T in T_list:
			print 'Working on K=%s, L=%s, T=%s, n=%s...'%(K, L, T, n)
			mu_array = self.backward_B_W(P_Y_given_X_array, P_X_given_X_array, Yn_list, n, T, K)
			print 'mu_array'
			print mu_array
			numerator = 0.0
			denominator = 0.0
			for i in range(K+1):
				numerator += i*fi_array[n, i]*mu_array[T-n, i]
				denominator += fi_array[n, i]*mu_array[T-n, i]
			X_hat = numerator/denominator
			X_hat_n_T_list.append(X_hat)
		print 'X_hat_n_T_list'
		print X_hat_n_T_list
		return X_hat_n_T_list
	
	def plot_smoothing_X_hat_list(self, T_list, Xn, X_hat_n_T_list, title, figure_fname, xlabel='x'):
		import pylab
		pylab.clf()
		Xn_list = [Xn]*len(T_list)
		pylab.plot(T_list, Xn_list, 'b')
		pylab.plot(T_list, X_hat_n_T_list, 'g')
		pylab.title(r'%s'%title)
		pylab.xlabel(r'%s'%xlabel)
		label_list = ['Xn_list', 'Xn_hat_list']
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=150)
		pylab.savefig('%s.eps'%figure_fname, dpi=150)
		pylab.savefig('%s.png'%figure_fname, dpi=150)
		#pylab.show()
	
	def predict_X_hat_T_n(self, Yn_list, K, L, T, n_list):
		"""
		2007-02-22
			prediction
		"""
		P_Y_array = self.initialize_P_Y_array(K, L)
		print 'P_Y_array'
		print P_Y_array
		P_Y_given_X_array = self.initialize_P_Y_given_X_array(K, L)
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		P_X_given_Y_array = self.initialize_P_X_given_Y_array(P_Y_given_X_array, P_Y_array)
		print 'P_X_given_Y_array'
		print P_X_given_Y_array
		P_Y_array_non_denovo = self.initialize_P_Y_array_non_denovo(P_Y_given_X_array)
		print 'P_Y_array_non_denovo'
		print P_Y_array_non_denovo
		P_X_given_X_array = self.initialize_P_X_given_X_array(K)
		print 'P_X_given_X_array'
		print P_X_given_X_array
		fi_array = self.forward_B_W(Yn_list, K, P_Y_array, P_Y_given_X_array, P_X_given_X_array, P_X_given_Y_array, max(n_list))
		print 'fi_array'
		print fi_array
		
		X_hat_list = []
		import Numeric
		P_X_transition_array = Numeric.identity(K+1, Numeric.Float)
		
		#the n_list is from big to small, 
		for n in n_list:
			print 'Working on K=%s, L=%s, T=%s, n=%s...'%(K, L, T, n)
			numerator = 0.0
			denominator = 0.0
			fi_T_n_array = Numeric.zeros(K+1, Numeric.Float)
			if n!=T:	#incremental one by one
				P_X_transition_array = Numeric.matrixmultiply(P_X_transition_array, P_X_given_X_array)
			for i in range(K+1):
				if n==T:
					fi_T_n_array[i] = fi_array[n, i]
				else:
					for j in range(K+1):
						fi_T_n_array[i] += fi_array[n,j]*P_X_transition_array[j, i]
				numerator += i*fi_T_n_array[i]
				denominator += fi_T_n_array[i]
			X_hat = numerator/denominator
			X_hat_list.append(X_hat)
		print 'X_hat_list'
		print X_hat_list
		
		return X_hat_list
		
	def test_simple_rw_HMM(self):
		K=20; L_list=[1,3,5,10,15,18,20]; chain_length=200
		output_fname = raw_input("Please specify file containing/to contain data(default='hw6_1_simulation'):")
		if not output_fname:
			output_fname='hw6_1_simulation.out'
		simulate_yes = raw_input("Do you need to simulate data into the file(y/N)?")
		if not simulate_yes:
			simulate_yes='n'
		if simulate_yes=='y':
			self.simulate_Xn_Yn(K, L_list, chain_length, output_fname)
		
		import csv, Numeric
		reader = csv.reader(open(output_fname))
		row = reader.next()
		
		#for prediction
		n_list = range(100,201)
		n_list.reverse()	#from big to small
		T = 200
		
		while row:
			if row[0]=='L':
				L = int(row[1])
				Xn_list = reader.next()
				Xn_list = map(int, Xn_list)
				Yn_list = reader.next()
				Yn_list = map(int, Yn_list)
				if L==5:
					#smoothing
					n = 100
					T_list = range(100,201)
					X_hat_n_T_list = self.estimate_X_hat_n_T(Yn_list, K, L, T_list, n)
					title = r'smoothing $\hat{X_{%s,T}}$ r.w. on A=[0,%s], P(W=+-%s)=1/2'%(n,K,L)
					figure_fname = 'hw6_1_a_K_%s_L_%s_n_%s'%(K,L,n)
					self.plot_smoothing_X_hat_list(T_list, Xn_list[n], X_hat_n_T_list, title, figure_fname, xlabel='T')
				
				#prediction
				X_hat_T_n_list = self.predict_X_hat_T_n(Yn_list, K, L, T, n_list)
				title = r'prediction $\hat{X_{%s,n}}$ r.w. on A=[0,%s], P(W=+-%s)=1/2'%(T,K,L)
				figure_fname = 'hw6_1_b_K_%s_L_%s_T_%s'%(K,L,T)
				self.plot_smoothing_X_hat_list(n_list, Xn_list[T], X_hat_T_n_list, title, figure_fname, xlabel='n')
			row = reader.next()
		del reader
	
class Math508_HW6_2(Math508_HW5_2):
	"""
	2007-02-23
	"""
	def setUp(self):
		print
	
	def initialize_P_X_given_X_array(self, no_of_X_states):
		import Numeric
		P_X_given_X_array = Numeric.zeros([no_of_X_states, no_of_X_states], Numeric.Float)
		for i in range(no_of_X_states):
			for j in range(no_of_X_states):
				P_X_given_X_array[i,j] = 1/3.0
		return P_X_given_X_array
		
	def backward_B_W(self, P_Y_given_X_array, P_X_given_X_array, Yn_list, n, T, no_of_X_states, map_of_Y_state):
		import Numeric
		mu_array = Numeric.zeros([T-n+1, no_of_X_states], Numeric.Float)	#it's inverse, row 0 is mu_{T}, row 1 is mu_{T-1}, etc
		for j in range(no_of_X_states):
			mu_array[0,j] = 1
		for i in range(1, T-n+1):
			Yn_index = T -i +1
			for j in range(no_of_X_states):
				for k in range(no_of_X_states):
					mu_array[i, j] += P_Y_given_X_array[k,map_of_Y_state[Yn_list[Yn_index]]]*P_X_given_X_array[j, k]*mu_array[i-1, k]
		return mu_array
		
		
	def forward_B_W(self, Yn_list, no_of_X_states, P_Y_array, P_Y_given_X_array, P_X_given_X_array, P_X_given_Y_array, n, map_of_Y_state):
		"""
		2007-02-22
			watch n+1, and K+1
		"""
		import Numeric
		fi_array = Numeric.zeros([n+1, no_of_X_states], Numeric.Float)
		#calculate fi_0
		P_Y = P_Y_array[map_of_Y_state[Yn_list[0]]]
		for j in range(no_of_X_states):
			fi_array[0,j] = P_X_given_Y_array[map_of_Y_state[Yn_list[0]], j]*P_Y
		#calculate fi_1, fi_2, ...
		for i in range(1, n+1):
			for j in range(no_of_X_states):
				for k in range(no_of_X_states):
					fi_array[i, j] += P_Y_given_X_array[j,map_of_Y_state[Yn_list[i]]]*P_X_given_X_array[k,j]*fi_array[i-1,k]
		return fi_array
	
	def smoothe_X_hat_n_T(self, Yn_list, no_of_X_states, T_list, n):
		"""
		2007-02-22
			smoothing, n<T
		"""
		import Numeric
		P_Y_given_X_array = self.initialize_P_Y_given_X_array()
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		P_X_array = Numeric.ones(no_of_X_states, Numeric.Float)/(float(no_of_X_states))
		P_Y_array = self.initialize_P_Y_array_non_denovo(P_Y_given_X_array, P_X_array)
		print 'P_Y_array'
		print P_Y_array
		P_X_given_Y_array = self.initialize_P_X_given_Y_array(P_Y_given_X_array, P_Y_array, P_X_array)
		print 'P_X_given_Y_array'
		print P_X_given_Y_array
		
		map_of_Y_state = {'H':0, 'T':1}
		
		P_X_given_X_array = self.initialize_P_X_given_X_array(no_of_X_states)
		print 'P_X_given_X_array'
		print P_X_given_X_array
		
		fi_array = self.forward_B_W(Yn_list, no_of_X_states, P_Y_array, P_Y_given_X_array, P_X_given_X_array, P_X_given_Y_array, n, map_of_Y_state)
		print 'fi_array'
		print fi_array
		
		pi_array = Numeric.zeros([len(T_list), no_of_X_states], Numeric.Float)
		for j in range(len(T_list)):
			T = T_list[j]
			print 'Working on no_of_X_states=%s, T=%s, n=%s...'%(no_of_X_states, T, n)
			mu_array = self.backward_B_W(P_Y_given_X_array, P_X_given_X_array, Yn_list, n, T, no_of_X_states, map_of_Y_state)
			print 'mu_array'
			print mu_array
			denominator = 0.0
			for i in range(no_of_X_states):
				pi_array[j,i] = fi_array[n, i]*mu_array[T-n, i]
				denominator += fi_array[n, i]*mu_array[T-n, i]
			pi_array[j] = pi_array[j]/denominator
		print 'pi_array'
		print pi_array
		return pi_array
	
	def predict_X_hat_T_n(self, Yn_list, no_of_X_states, T, n_list):
		"""
		2007-02-22
			prediction
		"""
		import Numeric
		P_Y_given_X_array = self.initialize_P_Y_given_X_array()
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		P_X_array = Numeric.ones(no_of_X_states, Numeric.Float)/(float(no_of_X_states))
		P_Y_array = self.initialize_P_Y_array_non_denovo(P_Y_given_X_array, P_X_array)
		print 'P_Y_array'
		print P_Y_array
		P_X_given_Y_array = self.initialize_P_X_given_Y_array(P_Y_given_X_array, P_Y_array, P_X_array)
		print 'P_X_given_Y_array'
		print P_X_given_Y_array
		
		map_of_Y_state = {'H':0, 'T':1}
		
		P_X_given_X_array = self.initialize_P_X_given_X_array(no_of_X_states)
		print 'P_X_given_X_array'
		print P_X_given_X_array
		
		fi_array = self.forward_B_W(Yn_list, no_of_X_states, P_Y_array, P_Y_given_X_array, P_X_given_X_array, P_X_given_Y_array, max(n_list), map_of_Y_state)
		print 'fi_array'
		print fi_array
		
		
		pi_array = Numeric.zeros([len(n_list), no_of_X_states], Numeric.Float)
		#the n_list is from big to small, 
		for m in range(len(n_list)):
			n = n_list[m]
			print 'Working on no_of_X_states=%s, T=%s, n=%s...'%(no_of_X_states, T, n)
			denominator = 0.0
			fi_T_n_array = Numeric.zeros(no_of_X_states, Numeric.Float)
			P_X_transition_array = Numeric.identity(no_of_X_states, Numeric.Float)
			for i in range(T-n):
				P_X_transition_array = Numeric.matrixmultiply(P_X_transition_array, P_X_given_X_array)
				print 'P_X_transition_array'
				print P_X_transition_array
			for i in range(no_of_X_states):
				if n==T:
					fi_T_n_array[i] = fi_array[n, i]
				else:
					for j in range(no_of_X_states):
						fi_T_n_array[i] += fi_array[n,j]*P_X_transition_array[j, i]
				print 'fi_T_n_array'
				print fi_T_n_array
				pi_array[m,i] = fi_T_n_array[i]
				denominator += fi_T_n_array[i]
			pi_array[m] = pi_array[m]/denominator
		print 'pi_array'
		print pi_array
		
		return pi_array
	
	def test_simple_rw_HMM(self):
		import csv, Numeric
		Yn_list = 'HHTHHTHTTTT'
		no_of_X_states = 3
		
		#for smoothing
		T_list = [8,10]
		n = 5
		pi_array = self.smoothe_X_hat_n_T(Yn_list, no_of_X_states, T_list, n)
		
		#for prediction
		n_list = [6,5]
		T = 10
		pi_array = self.predict_X_hat_T_n(Yn_list, no_of_X_states, T, n_list)

class viterbi_algorithm:
	"""
	2007-03-02
	"""
	def init_score_and_trace_matrix(self, no_of_states, no_of_Yns):
		sys.stderr.write("Initialize score_matrix and trace_matrix ... ")
		import Numeric
		score_matrix = Numeric.zeros([no_of_states, no_of_Yns], Numeric.Float)
		trace_matrix = []
		for i in range(no_of_states):
			trace_matrix.append([])
			for j in range(no_of_Yns):
				trace_matrix[i].append(0)
		sys.stderr.write("Done.\n")
		return score_matrix, trace_matrix
	
	def run_viterbi_algorithm(self, score_matrix, trace_matrix, Yn_list, P_Y_given_X_array, P_X_given_X_array):
		sys.stderr.write("Running viterbi algorithm ...")
		no_of_states, no_of_Yns = score_matrix.shape
		for j in range(1, no_of_Yns):
			for i in range(no_of_states):
				candidate_score_value_list = []
				max_score = 0
				for k in range(no_of_states):
					candidate_score = P_Y_given_X_array[i, Yn_list[j]]*P_X_given_X_array[k,i]*score_matrix[k,j-1]
					candidate_score_value_list.append(candidate_score)
					if candidate_score>max_score:
						max_score = candidate_score
				score_matrix[i,j] = max_score
				source_of_max_score_list = []
				for k in range(no_of_states):
					if candidate_score_value_list[k] == max_score:
						source_of_max_score_list.append(k)
				trace_matrix[i][j] = source_of_max_score_list
		sys.stderr.write("Done.\n")
		return score_matrix, trace_matrix
	
	def recursive_trace(self, trace_matrix, i, j, func):
		X_state_list = []
		if j>0:
			prev_X_state = func(trace_matrix[i][j])
			X_state_list = self.recursive_trace(trace_matrix, prev_X_state, j-1, func)
			X_state_list.append(prev_X_state)
		return X_state_list
	
	def trace(self, score_matrix, trace_matrix, func):
		sys.stderr.write("Tracing the route ...")
		no_of_Yns = score_matrix.shape[1]
		X_state_for_last_Yn = 0
		max_last_score = 0
		for i in range(score_matrix.shape[0]):
			if func==min:
				if score_matrix[i, no_of_Yns-1]>max_last_score:
					max_last_score = score_matrix[i, no_of_Yns-1]
					X_state_for_last_Yn = i
			elif func == max:
				if score_matrix[i, no_of_Yns-1]>=max_last_score:	#the only difference is >= or >
					max_last_score = score_matrix[i, no_of_Yns-1]
					X_state_for_last_Yn = i
		X_state_list = self.recursive_trace(trace_matrix, X_state_for_last_Yn, no_of_Yns-1, func)
		X_state_list.append(X_state_for_last_Yn)
		sys.stderr.write("Done.\n")
		return X_state_list

class Math508_HW7_1(viterbi_algorithm, Math508_HW6_2):
	"""
	2007-03-02
	"""
	def setUp(self):
		print
	
	def initialize_P_Y_given_X_array(self):
		"""
		for P_Y_given_X_array, row is X, column is Y
		for P_Y_X_array, row is Y, column is X
		"""
		import Numeric
		P_Y_given_X_array = Numeric.array([[0.75, 0.25], [0.5,0.5], [0.25,0.75]])
		return P_Y_given_X_array
	
	def initlize_1st_column(self, score_matrix, P_Y_given_X_array, P_X_array, Yn_list):
		for i in range(score_matrix.shape[0]):
			score_matrix[i,0] = P_Y_given_X_array[i, Yn_list[0]]*P_X_array[i]
		return score_matrix
	
	def test_simple_rw_HMM(self):
		import csv, Numeric
		old_Yn_list = 'HHHHTHTTTT'
		no_of_X_states = 3
		
		P_Y_given_X_array = self.initialize_P_Y_given_X_array()
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		
		P_X_array = Numeric.ones(no_of_X_states, Numeric.Float)/(float(no_of_X_states))
		P_Y_array = self.initialize_P_Y_array_non_denovo(P_Y_given_X_array, P_X_array)
		print 'P_Y_array'
		print P_Y_array
		
		P_X_given_X_array = self.initialize_P_X_given_X_array(no_of_X_states)
		print 'P_X_given_X_array'
		print P_X_given_X_array
		
		map_of_Y_state = {'H':0, 'T':1}
		
		Yn_list = []
		for i in range(len(old_Yn_list)):
			Yn_list.append(map_of_Y_state[old_Yn_list[i]])
		
		no_of_Yns = len(Yn_list)
		score_matrix, trace_matrix = self.init_score_and_trace_matrix(no_of_X_states, no_of_Yns)
		score_matrix = self.initlize_1st_column(score_matrix, P_Y_given_X_array, P_X_array, Yn_list)
		print 'score_matrix after 1st column initilization'
		print score_matrix
		score_matrix, trace_matrix = self.run_viterbi_algorithm(score_matrix, trace_matrix, Yn_list, P_Y_given_X_array, P_X_given_X_array)
		print 'score_matrix'
		print score_matrix
		print 'trace_matrix'
		for i in range(no_of_X_states):
			print trace_matrix[i]
		max_X_state_list = self.trace(score_matrix, trace_matrix, func=max)
		print 'max_X_state_list'
		print max_X_state_list
		min_X_state_list = self.trace(score_matrix, trace_matrix, func=min)
		print 'min_X_state_list'
		print min_X_state_list

class Math508_HW7_2(viterbi_algorithm, Math508_HW6_1):
	"""
	2007-03-02
	"""
	def setUp(self):
		print
	
	def initlize_1st_column(self, score_matrix, P_Y_given_X_array, initial_x_state, Yn_list):
		for i in range(score_matrix.shape[0]):
			if i==initial_x_state:
				score_matrix[i,0] = P_Y_given_X_array[i, Yn_list[0]]
			else:
				score_matrix[i,0] = 0
		return score_matrix
	
	def plot(self, Xn_list, X_hat_n_T_list, max_X_state_list, min_X_state_list, title, figure_fname):
		import pylab, Numeric
		pylab.clf()
		x_index_list = range(len(Xn_list))
		pylab.plot(x_index_list, Xn_list, 'b')
		pylab.plot(x_index_list, X_hat_n_T_list, 'g')
		pylab.plot(x_index_list, max_X_state_list, 'r')
		pylab.plot(x_index_list, min_X_state_list, 'c')
		pylab.plot(x_index_list, (Numeric.array(min_X_state_list)+Numeric.array(max_X_state_list))/2, 'k')
		pylab.title(r'%s'%title)
		pylab.xlabel('n')
		label_list = ['Xn_list', 'Xn_hat_list', 'max_X_state_list', 'min_X_state_list', '(max+min)/2']
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=200)
		pylab.savefig('%s.eps'%figure_fname, dpi=200)
		pylab.savefig('%s.png'%figure_fname, dpi=200)
		#pylab.show()
	
	def smoothe_X_hat_n_T(self, Yn_list, K, L, T, n_list):
		"""
		2007-02-22
			smoothing, n<T
		"""
		P_Y_array = self.initialize_P_Y_array(K, L)
		print 'P_Y_array'
		print P_Y_array
		P_Y_given_X_array = self.initialize_P_Y_given_X_array(K, L)
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		P_X_given_Y_array = self.initialize_P_X_given_Y_array(P_Y_given_X_array, P_Y_array)
		print 'P_X_given_Y_array'
		print P_X_given_Y_array
		P_Y_array_non_denovo = self.initialize_P_Y_array_non_denovo(P_Y_given_X_array)
		print 'P_Y_array_non_denovo'
		print P_Y_array_non_denovo
		P_X_given_X_array = self.initialize_P_X_given_X_array(K)
		print 'P_X_given_X_array'
		print P_X_given_X_array
		max_n = max(n_list)
		fi_array = self.forward_B_W(Yn_list, K, P_Y_array, P_Y_given_X_array, P_X_given_X_array, P_X_given_Y_array, max_n)
		print 'fi_array'
		print fi_array
		min_n = min(n_list)
		mu_array = self.backward_B_W(P_Y_given_X_array, P_X_given_X_array, Yn_list, min_n, T, K)
		print 'mu_array'
		print mu_array
		
		X_hat_n_T_list = []
		for n in n_list:
			print 'Working on K=%s, L=%s, T=%s, n=%s...'%(K, L, T, n)
			numerator = 0.0
			denominator = 0.0
			for i in range(K+1):
				numerator += i*fi_array[n, i]*mu_array[T-n, i]
				denominator += fi_array[n, i]*mu_array[T-n, i]
			X_hat = numerator/denominator
			X_hat_n_T_list.append(X_hat)
		print 'X_hat_n_T_list'
		print X_hat_n_T_list
		return X_hat_n_T_list
	
	def find_optimal_path(self, Yn_list, K, L, initial_x_state):
		P_Y_given_X_array = self.initialize_P_Y_given_X_array(K, L)
		print 'P_Y_given_X_array'
		print P_Y_given_X_array
		
		#P_X_array = Numeric.ones(no_of_X_states, Numeric.Float)/(float(no_of_X_states))
		P_Y_array = self.initialize_P_Y_array(K, L)
		print 'P_Y_array'
		print P_Y_array
		
		P_X_given_X_array = self.initialize_P_X_given_X_array(K)
		print 'P_X_given_X_array'
		print P_X_given_X_array
		
		no_of_X_states = K + 1
		no_of_Yns = len(Yn_list)
		score_matrix, trace_matrix = self.init_score_and_trace_matrix(no_of_X_states, no_of_Yns)
		score_matrix = self.initlize_1st_column(score_matrix, P_Y_given_X_array, initial_x_state, Yn_list)
		print 'score_matrix after 1st column initilization'
		print score_matrix
		score_matrix, trace_matrix = self.run_viterbi_algorithm(score_matrix, trace_matrix, Yn_list, P_Y_given_X_array, P_X_given_X_array)
		print 'score_matrix'
		print score_matrix
		print 'trace_matrix'
		for i in range(no_of_X_states):
			print trace_matrix[i]
		max_X_state_list = self.trace(score_matrix, trace_matrix, func=max)
		print 'max_X_state_list'
		print max_X_state_list
		min_X_state_list = self.trace(score_matrix, trace_matrix, func=min)
		print 'min_X_state_list'
		print min_X_state_list
		
		return max_X_state_list, min_X_state_list
	
	def test_simple_rw_HMM(self):
		K=20; L_list=[10,14,16,17]; chain_length=400
		output_fname = raw_input("Please specify file containing/to contain data(default='hw7_2_simulation'):")
		if not output_fname:
			output_fname='hw7_2_simulation.out'
		simulate_yes = raw_input("Do you need to simulate data into the file(y/N)?")
		if not simulate_yes:
			simulate_yes='n'
		if simulate_yes=='y':
			self.simulate_Xn_Yn(K, L_list, chain_length, output_fname)
		
		import csv, Numeric
		reader = csv.reader(open(output_fname))
		row = reader.next()
		#smoothing
		n_list = range(0, 400+1)
		T = 400
		
		while row:
			if row[0]=='L':
				L = int(row[1])
				Xn_list = reader.next()
				Xn_list = map(int, Xn_list)
				Yn_list = reader.next()
				Yn_list = map(int, Yn_list)
				X_hat_n_T_list = self.smoothe_X_hat_n_T(Yn_list, K, L, T, n_list)
				max_X_state_list, min_X_state_list = self.find_optimal_path(Yn_list, K, L, initial_x_state=10)
				title = r'Optimal Path by Viterbi X_hat_{n,%s} r.w. on A=[0,%s], P(W=+-%s)=1/2'%(T,K,L)
				figure_fname = 'hw7_2_K_%s_L_%s_T_%s'%(K,L,T)
				self.plot(Xn_list, X_hat_n_T_list, max_X_state_list, min_X_state_list, title, figure_fname)
			row = reader.next()
		del reader

class ConstantMCFiltering:
	def filtering(self, Yn_list, P_X_array, X_state_list, prob_W, func_to_cal_W, sigma):
		import Numeric
		no_of_X_states = len(X_state_list)
		pi_array = Numeric.zeros([len(Yn_list), no_of_X_states], Numeric.Float)
		pi_array[0,:] = P_X_array
		for i in range(1, len(Yn_list)):
			for j in range(no_of_X_states):
				denominator  = 0.0
				for k in range(no_of_X_states):
					W = func_to_cal_W(Yn_list[i], X_state_list[k], Yn_list[i-1], sigma)
					denominator += prob_W(W)*pi_array[i-1, k]
				W = func_to_cal_W(Yn_list[i], X_state_list[j], Yn_list[i-1], sigma)
				pi_array[i,j] = prob_W(W)*pi_array[i-1,j]/denominator
		return pi_array
	
	def func_to_cal_Y(self, X_n, Y_n, W, sigma):
		return X_n*Y_n+sigma*W
	
	def func_to_cal_W(self, Y_n_plus_1, X_n, Y_n, sigma):
		return (Y_n_plus_1-X_n*Y_n)/sigma
	
	def simulate(self, X_state_list, Y_0, sample_W, sigma, func_to_cal_Y, chain_length=11):
		import random
		X_0 = random.sample(X_state_list,1)[0]
		Wn_list= []
		Xn_list = [X_0]
		Yn_list = [Y_0]
		for i in range(1, chain_length):
			Xn_list.append(Xn_list[i-1])
			W = sample_W()
			Wn_list.append(W)
			Y_n_plus_1 = func_to_cal_Y(Xn_list[i-1], Yn_list[i-1], W, sigma)
			Yn_list.append(Y_n_plus_1)
		return Xn_list, Wn_list, Yn_list

class Math508_HW8_a(ConstantMCFiltering, unittest.TestCase):
	"""
	2007-03-23
	"""
	def setUp(self):
		print
	
	def prob_W(self, W):
		if W==1 or W==-1:
			return 0.5
		else:
			return 0.0
	
	def sample_W(self):
		import random
		u = random.random()
		if u>0.5:
			return 1
		else:
			return -1
	
	def test_simulate_filtering(self):
		X_state_list = xrange(6)
		Y_0 = 0
		sigma = 2
		chain_length=11
		Xn_list, Wn_list, Yn_list = self.simulate(X_state_list, Y_0, self.sample_W, sigma, self.func_to_cal_Y, chain_length)
		print 'Xn_list'
		print Xn_list
		print 'Wn_list'
		print Wn_list
		print 'Yn_list'
		print Yn_list
		P_X_array = [1.0/len(X_state_list)]*len(X_state_list)
		pi_array = self.filtering(Yn_list, P_X_array, X_state_list, self.prob_W, self.func_to_cal_W, sigma)
		print 'pi_array'
		print pi_array

class Math508_HW8_b(ConstantMCFiltering, unittest.TestCase):
	"""
	2007-03-23
	"""
	def setUp(self):
		print
	
	def prob_W(self, W):
		import rpy
		return rpy.r.dnorm(W)
	
	def sample_W(self):
		import random
		return random.gauss(0,1)
	
	def test_simulate_filtering(self):
		X_state_list = xrange(6)
		Y_0 = 0
		sigma = 1
		chain_length=11
		Xn_list, Wn_list, Yn_list = self.simulate(X_state_list, Y_0, self.sample_W, sigma, self.func_to_cal_Y, chain_length)
		print 'Xn_list'
		print Xn_list
		print 'Wn_list'
		print Wn_list
		print 'Yn_list'
		print Yn_list
		P_X_array = [1.0/len(X_state_list)]*len(X_state_list)
		pi_array = self.filtering(Yn_list, P_X_array, X_state_list, self.prob_W, self.func_to_cal_W, sigma)
		print 'pi_array'
		print pi_array

class KalmanFilter:
	"""
	2007-04-05
	"""
	def simulate(self, X_0, sample_V, sample_W, chain_length, alpha, epsilon, delta):
		import sys
		sys.stderr.write("Simulating ...")
		Xn_list = [X_0]
		Vn_list = []
		Yn_list = [Xn_list[0]+delta*self.sample_W()]
		Wn_list = []
		for i in range(1, chain_length):
			Vn = sample_V()
			Xn = alpha*Xn_list[i-1]+epsilon*Vn
			Wn = sample_W()
			Yn = Xn + delta*Wn
			Vn_list.append(Vn)
			Xn_list.append(Xn)
			Wn_list.append(Wn)
			Yn_list.append(Yn)
		sys.stderr.write("Done.\n")
		return Vn_list, Xn_list, Wn_list, Yn_list
	
	def decode(self, Yn_list, E_X_0_squared, alpha, epsilon, delta):
		import sys
		sys.stderr.write("Decoding ...")
		c = epsilon*epsilon/(epsilon*epsilon+delta*delta)
		Pn_list = [(c-1)*(c-1)*E_X_0_squared + c*c*delta*delta]
		X_hat_list = [c*Yn_list[0]]
		for i in range(1, len(Yn_list)):
			common_coeff = (alpha*alpha*Pn_list[i-1]+epsilon*epsilon)/(alpha*alpha*Pn_list[i-1]+epsilon*epsilon+delta*delta)
			Pn = delta*delta*common_coeff
			Xn_hat = alpha*X_hat_list[i-1] + common_coeff*(Yn_list[i]-alpha*X_hat_list[i-1])
			Pn_list.append(Pn)
			X_hat_list.append(Xn_hat)
		sys.stderr.write("Done.\n")
		return Pn_list, X_hat_list
	
	def plot(self, Xn_list, X_hat_list, Yn_list, title, figure_fname):
		import pylab, Numeric
		pylab.clf()
		x_index_list = range(len(Xn_list))
		pylab.plot(x_index_list, Xn_list, 'b')
		pylab.plot(x_index_list, X_hat_list, 'g')
		pylab.plot(x_index_list, Yn_list, 'r')
		pylab.title(r'%s'%title)
		pylab.xlabel('n')
		label_list = ['Xn_list', 'Xn_hat_list', 'Yn_list']
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=200)
		pylab.savefig('%s.eps'%figure_fname, dpi=200)
		pylab.savefig('%s.png'%figure_fname, dpi=200)
		pylab.show()
	
	def plot_Pn(self, Pn_list, title, figure_fname):
		import pylab, Numeric
		pylab.clf()
		x_index_list = range(len(Pn_list))
		pylab.plot(x_index_list, Pn_list, 'b')
		pylab.title(r'%s'%title)
		pylab.xlabel('n')
		label_list = ['Pn_list']
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=200)
		pylab.savefig('%s.eps'%figure_fname, dpi=200)
		pylab.savefig('%s.png'%figure_fname, dpi=200)
		pylab.show()
		

class Math508_HW10_1(KalmanFilter, unittest.TestCase):
	"""
	2007-04-06
	"""
	def setUp(self):
		print
	
	def sample_W(self):
		import random
		return random.gauss(0,1)
	
	def simulate_decode(self, chain_length, alpha, epsilon, delta):
		X_0 = self.sample_W()
		self.sample_V = self.sample_W
		Vn_list, Xn_list, Wn_list, Yn_list = self.simulate(X_0, self.sample_V, self.sample_W, chain_length, alpha, epsilon, delta)
		print 'Vn_list'
		print Vn_list
		print 'Xn_list'
		print Xn_list
		print 'Wn_list'
		print Wn_list
		print 'Yn_list'
		print Yn_list
		E_X_0_squared = 1.0
		Pn_list, X_hat_list = self.decode(Yn_list, E_X_0_squared, alpha, epsilon, delta)
		print 'Pn_list'
		print Pn_list
		print 'X_hat_list'
		print X_hat_list
		title = r'KalmanFilter on X0, V1, W1~N(0,1), N=%s, a=%s, e=%s, d=%s'%(chain_length, alpha, epsilon, delta)
		figure_fname = 'hw10_1_N_%s_a_%s_e_%s_d_%s'%(chain_length, alpha, epsilon, delta)
		self.plot(Xn_list, X_hat_list, Yn_list, title, figure_fname)
		
		title = r'KalmanFilter error on X0, V1, W1~N(0,1), N=%s, a=%s, e=%s, d=%s'%(chain_length, alpha, epsilon, delta)
		figure_fname = 'hw10_1_error_N_%s_a_%s_e_%s_d_%s'%(chain_length, alpha, epsilon, delta)
		self.plot_Pn(Pn_list, title, figure_fname)
		
	def test_simulate_decode(self):
		chain_length = 201
		alpha = 0.9
		epsilon = 0.3
		delta = 1
		self.simulate_decode(chain_length, alpha, epsilon, delta)
		
		chain_length = 201
		alpha = 0.8
		epsilon = 0.9
		delta = 2
		self.simulate_decode(chain_length, alpha, epsilon, delta)

class Math508_HW10_2(KalmanFilter, unittest.TestCase):
	"""
	2007-04-06
	"""
	def setUp(self):
		print
	
	def sample_W(self):
		import random
		u = random.random()
		if u>0.5:
			return 1
		else:
			return -1
	
	def simulate_decode(self, chain_length, alpha, epsilon, delta):
		X_0 = 1
		self.sample_V = self.sample_W
		Vn_list, Xn_list, Wn_list, Yn_list = self.simulate(X_0, self.sample_V, self.sample_W, chain_length, alpha, epsilon, delta)
		print 'Vn_list'
		print Vn_list
		print 'Xn_list'
		print Xn_list
		print 'Wn_list'
		print Wn_list
		print 'Yn_list'
		print Yn_list
		E_X_0_squared = 1.0
		Pn_list, X_hat_list = self.decode(Yn_list, E_X_0_squared, alpha, epsilon, delta)
		print 'Pn_list'
		print Pn_list
		print 'X_hat_list'
		print X_hat_list
		title = r'KalmanFilter on X0=1, P(V1=+-1)=P(W1=+-1)=1/2, N=%s, a=%s, e=%s, d=%s'%(chain_length, alpha, epsilon, delta)
		figure_fname = 'hw10_2_N_%s_a_%s_e_%s_d_%s'%(chain_length, alpha, epsilon, delta)
		self.plot(Xn_list, X_hat_list, Yn_list, title, figure_fname)
		
		title = r'KalmanFilter error on X0, V1, W1~N(0,1), N=%s, a=%s, e=%s, d=%s'%(chain_length, alpha, epsilon, delta)
		figure_fname = 'hw10_2_error_N_%s_a_%s_e_%s_d_%s'%(chain_length, alpha, epsilon, delta)
		self.plot_Pn(Pn_list, title, figure_fname)
		
	def test_simulate_decode(self):
		chain_length = 201
		alpha = 0.9
		epsilon = 0.3
		delta = 1
		self.simulate_decode(chain_length, alpha, epsilon, delta)

class Math508_HW11_3(unittest.TestCase):
	"""
	2007-04-13
	"""
	def setUp(self):
		print
	
	def simulate_eta_1_2(self, sigma_1, sigma_2):
		import random
		return (random.gauss(0,sigma_1*sigma_1), random.gauss(0, sigma_2*sigma_2))
	
	def simulate_real_part_of_theta_n_Xn_Yn(self, N, M, a, lambda_1, lambda_2, eta_1, eta_2):
		sys.stderr.write("Simulating ...")
		import math
		Real_theta_n_list = []
		Real_X_n_list = []
		Real_Y_n_list = []
		for i in range(0, N+1):
			theta_n = eta_1*math.cos(lambda_1*i)
			Real_theta_n_list.append(theta_n)
			X_n = theta_n + eta_2*math.cos(lambda_2*i)
			Real_X_n_list.append(X_n)
			part_of_real_Y_n = 0
			for j in range(0, M+1):
				part_of_real_Y_n += eta_2*math.cos(lambda_2*i+(lambda_1-lambda_2)*j)
			Real_Y_n = (1-a*a)*((M+1)*eta_1*math.cos(lambda_1*i)+part_of_real_Y_n)
			Real_Y_n_list.append(Real_Y_n)
		sys.stderr.write("Done.\n")
		return Real_theta_n_list, Real_X_n_list, Real_Y_n_list
	
	def plot(self, Xn_list, Yn_list, label_list, title, figure_fname):
		import pylab, Numeric
		pylab.clf()
		x_index_list = range(len(Xn_list))
		pylab.plot(x_index_list, Xn_list, 'b')
		pylab.plot(x_index_list, Yn_list, 'r')
		pylab.title(r'%s'%title)
		pylab.xlabel('n')
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=200)
		pylab.savefig('%s.eps'%figure_fname, dpi=200)
		pylab.savefig('%s.png'%figure_fname, dpi=200)
		pylab.show()

	def test_simulate(self):
		import math
		sigma_1 = 1
		sigma_2 = 2
		N = 50
		M = 10
		a = 0.8
		lambda_1 = math.pi/4
		lambda_2 = math.pi*2/3
		eta_1, eta_2 = self.simulate_eta_1_2(sigma_1, sigma_2)
		Real_theta_n_list, Real_X_n_list, Real_Y_n_list = self.simulate_real_part_of_theta_n_Xn_Yn(N, M, a, lambda_1, lambda_2, eta_1, eta_2)
		label_list = ['theta_n', 'Xn']
		title = r'Real parts of theta_n, Xn, eta_1=%s, eta_2=%s'%(eta_1, eta_2)
		figure_fname = 'hw11_3_theta_n_Xn'
		self.plot(Real_theta_n_list, Real_X_n_list, label_list, title, figure_fname)
		
		label_list = ['theta_n', 'Yn']
		title = r'Real parts of theta_n, Yn, eta_1=%s, eta_2=%s'%(eta_1, eta_2)
		figure_fname = 'hw11_3_theta_n_Yn'
		self.plot(Real_theta_n_list, Real_Y_n_list, label_list, title, figure_fname)

class Math508_HW12_3(unittest.TestCase):
	"""
	2007-04-22
	"""
	def setUp(self):
		print
	
	def sample_W(self):
		import random
		return random.gauss(0,1)
	
	def sample_X_0(self):
		import random
		return random.gauss(7, 0.5)
	
	def simulate(self, sample_X_0, sample_V, sample_W, chain_length, a=1.004, b=0.06, sigma=2.0):
		import sys
		sys.stderr.write("Simulating ...")
		Xn_list = []
		Vn_list = []
		Yn_list = []
		Wn_list = []
		for i in range(chain_length):
			Vn = sample_V()
			if i==0:
				Xn = sample_X_0()
			else:
				Xn = a*Xn_list[i-1]+b*Xn_list[i-1]*Vn
			Wn = sample_W()
			Yn = Xn + sigma*Wn
			Vn_list.append(Vn)
			Xn_list.append(Xn)
			Wn_list.append(Wn)
			Yn_list.append(Yn)
		sys.stderr.write("Done.\n")
		return Vn_list, Xn_list, Wn_list, Yn_list
	
	def prob_func_Y_given_X(self, X_symbol, Y, sigma=2):
		"""
		Yn|Xn ~ N(Xn, 2^2)
		"""
		import swiginac
		return 1/(sigma*swiginac.sqrt(2*swiginac.Pi))*swiginac.exp(-((Y-X_symbol)**2)/(2*sigma**2))
	
	def prob_func_X_0(self, X_symbol):
		"""
		X_0 ~ N(7, 0.5^2)
		"""
		import swiginac
		return 2/(swiginac.sqrt(2*swiginac.Pi))*swiginac.exp(-2*((X_symbol-7)**2))
	
	def prob_func_X_given_X(self, old_X_symbol, new_X_symbol, a=1.004, b=0.06):
		"""
		Xn|Xn-1 ~ N(a*Xn-1, b*Xn-1)
		"""
		import swiginac
		return 1/(a*old_X_symbol*swiginac.sqrt(2*swiginac.Pi))*swiginac.exp(-(new_X_symbol-a*old_X_symbol)**2/(2*(b*old_X_symbol)**2))
	
	def integral_by_riemann_sum(self, symbol, lower, upper, func, no_of_samples=1E3):
		integral = 0
		gap = (upper-lower)/no_of_samples
		x_i_1 = lower
		x_i = x_i_1 + gap
		i = 0
		while i<no_of_samples:
			x = (x_i_1+x_i)/2
			integral += func.subs(symbol==x)*gap
			x_i_1 += gap
			x_i += gap
			i += 1
		return integral
	
	def filter(self, Yn_list, prob_func_Y_given_X, prob_func_X_0,prob_func_X_given_X, X_integral_region=[-30, 40], a=1.004, b=0.06, sigma=2):
		"""
		2007-04-22
			either integral_by_riemann_sum or swiginac.integral doesn't work
			integral_by_riemann_sum encounters "Floating point underflow" or "Floating point overflow"
			swiginac.integral doesn't solve multi integral (return the inner integral unevaluated)
		2007-04-23 deprecated
		"""
		import sys, swiginac
		sys.stderr.write("Filtering ...\n")
		X_symbol_ls = []
		fi_array = []
		Xn_hat_list = []
		for i in range(len(Yn_list)):
			X_symbol_ls.append(swiginac.symbol("x%s"%i))
			#import pdb
			#pdb.set_trace()
			if i==0:
				fi_array.append(prob_func_Y_given_X(X_symbol_ls[i], Yn_list[i], sigma)*prob_func_X_0(X_symbol_ls[i]))
			else:
				fi_int_part = self.integral_by_riemann_sum(X_symbol_ls[i-1], X_integral_region[0], X_integral_region[1], prob_func_X_given_X(X_symbol_ls[i-1], X_symbol_ls[i], a, b)*fi_array[i-1])
				fi_func = prob_func_Y_given_X(X_symbol_ls[i], Yn_list[i], sigma)*fi_int_part.evalf()
				fi_func.evalf()
				fi_array.append(fi_func)
			Xn_hat = swiginac.integral(X_symbol_ls[i], X_integral_region[0], X_integral_region[1], X_symbol_ls[i]*fi_array[i])/swiginac.integral(X_symbol_ls[i], X_integral_region[0], X_integral_region[1], fi_array[i])
			Xn_hat_list.append(Xn_hat.evalf())
			sys.stderr.write("%s%s"%('\x08'*10, i))
		sys.stderr.write("Done.\n")
		return Xn_hat_list
	
	def normal_density(self, X, mean, sd):
		"""
		2007-04-23
			rpy.r.dnorm() sometimes return NaN
		"""
		import math
		return 1/(sd*math.sqrt(2*math.pi))*math.exp(-(X-mean)*(X-mean)/(2.0*sd*sd))
	
	def prob_Y_given_X(self, X, Y, sigma=2):
		return self.normal_density(Y, X, sigma)
	
	def prob_X_given_X(self, old_X, new_X, a=1.004, b=0.06):
		return self.normal_density(new_X, a*old_X, abs(b*old_X))
	
	def prob_X_0(self, X):
		return self.normal_density(X, 7, 0.5)
	
	def filter_by_discretization(self, Yn_list, prob_Y_given_X, prob_X_given_X, prob_X_0, X_integral_region=[-30, 40], gap=0.01, a=1.004, b=0.06, sigma=2):
		"""
		2007-04-23
			use discretization and Riemann-sum to solve the integral
		"""
		import sys, Numeric
		sys.stderr.write("Filtering ...\n")
		X_array = []
		x_i = X_integral_region[0]
		while x_i < X_integral_region[1]:
			X_array.append(x_i)
			x_i+= gap
		no_of_X_states = len(X_array)
		no_of_Yns = len(Yn_list)
		fi_array = Numeric.zeros([no_of_Yns, no_of_X_states], Numeric.Float)
		numerator = 0.0
		denominator = 0.0
		Xn_hat_list = []
		import pdb
		#pdb.set_trace()
		for i in range(no_of_X_states):
			fi_array[0, i] = prob_Y_given_X(X_array[i], Yn_list[0])*prob_X_0(X_array[i])
			numerator += X_array[i]*fi_array[0,i]
			denominator += fi_array[0,i]
		Xn_hat_list.append(numerator/denominator)
		#pdb.set_trace()
		for i in range(1,no_of_Yns):
			numerator = 0.0
			denominator = 0.0
			for j in range(no_of_X_states):
				for k in range(no_of_X_states):
					fi_array[i,j] += prob_Y_given_X(X_array[j], Yn_list[i])*prob_X_given_X(X_array[k], X_array[j])*fi_array[i-1,k]*gap
				numerator += X_array[j]*fi_array[i,j]
				denominator += fi_array[i,j]
			Xn_hat_list.append(numerator/denominator)
			sys.stderr.write("%s%s"%('\x08'*10, i))
		sys.stderr.write("Done.\n")
		return Xn_hat_list
	
	def sample_X_0(self):
		"""
		2007-04-24
			for SIR
		"""
		import random
		return random.gauss(7, 0.5)
	
	def sample_X_given_X(self, old_X, a=1.004, b=0.06):
		"""
		2007-04-24
			for SIR
		"""
		import random
		return random.gauss(a*old_X, abs(b*old_X))
	
	def resampling_by_residual_sampling(self, signal_X_matrix, trace_matrix, importance_vector, index, sum_importance):
		import Numeric, random, math
		no_of_samplings = len(importance_vector)
		residual_importance_vector = Numeric.zeros(no_of_samplings, Numeric.Float)
		no_of_samples_sampled_so_far = 0
		i = 0
		cumulative_df_ls = []
		while i<no_of_samplings:
			importance_vector[i] = importance_vector[i]/sum_importance
			no_of_copies_for_i = int(math.floor(no_of_samplings*importance_vector[i]))
			for j in range(no_of_copies_for_i):
				signal_X_matrix[2*index+1, no_of_samples_sampled_so_far] = signal_X_matrix[2*index, i]
				trace_matrix[2*index+1, no_of_samples_sampled_so_far] = i
				no_of_samples_sampled_so_far += 1
			residual_importance_vector[i] = no_of_samplings*importance_vector[i] - no_of_copies_for_i
			if i==0:
				cumulative_df_ls.append(residual_importance_vector[i])
			else:
				cumulative_df_ls.append(cumulative_df_ls[i-1]+residual_importance_vector[i])
			i += 1
		no_of_samples_left = no_of_samplings - no_of_samples_sampled_so_far
		for i in range(no_of_samples_left):
			u = random.random()*cumulative_df_ls[-1]	#cumulative_df_ls is not normalized
			j = 0
			while u > cumulative_df_ls[j]:
				j+=1
			signal_X_matrix[2*index+1, no_of_samples_sampled_so_far+i] = signal_X_matrix[2*index, j]
			trace_matrix[2*index+1, no_of_samples_sampled_so_far+i] = j
	
	def filter_by_SIR(self, Yn_list, sample_X_given_X, sample_X_0, prob_Y_given_X,a=1.004, b=0.06, sigma=2, no_of_samplings=int(1E4)):
		"""
		2007-04-24
			sequential importance resampling (SIR) (theoretic stuff see hw12.tex)
		"""
		import sys, Numeric
		sys.stderr.write("Filtering ...\n")
		no_of_Yns = len(Yn_list)
		signal_X_matrix = Numeric.zeros([2*no_of_Yns-1, no_of_samplings], Numeric.Float)
		trace_matrix = Numeric.zeros([2*no_of_Yns-1, no_of_samplings], Numeric.Int)
		importance_vector = Numeric.zeros(no_of_samplings, Numeric.Float)
		stepwise_Xn_hat_list = []
		#import pdb
		#pdb.set_trace()
		for i in range(no_of_Yns):
			j= 0
			stepwise_Xn_hat = 0.0
			sum_importance = 0.0
			while j<no_of_samplings:
				if i==0:
					x_sample = sample_X_0()
				else:
					x_sample = sample_X_given_X(signal_X_matrix[2*i-1,j], a, b)
				signal_X_matrix[2*i,j] = x_sample
				trace_matrix[2*i,j] = j
				importance_vector[j] = prob_Y_given_X(x_sample, Yn_list[i], sigma)
				stepwise_Xn_hat += x_sample*importance_vector[j]
				sum_importance += importance_vector[j]
				j += 1
			stepwise_Xn_hat = stepwise_Xn_hat/sum_importance
			stepwise_Xn_hat_list.append(stepwise_Xn_hat)
			if i!=no_of_Yns-1:	#the last step doesn't need to do resampling
				self.resampling_by_residual_sampling(signal_X_matrix, trace_matrix, importance_vector, i, sum_importance)
			sys.stderr.write("%s%s"%('\x08'*10, i))
		sys.stderr.write("Done.\n")
		return stepwise_Xn_hat_list
	
	def plot(self, Xn_list, X_hat_list, Yn_list, title, figure_fname):
		import pylab, Numeric
		pylab.clf()
		x_index_list = range(len(Xn_list))
		pylab.plot(x_index_list, Xn_list, 'b')
		pylab.plot(x_index_list, X_hat_list, 'g')
		pylab.plot(x_index_list, Yn_list, 'r')
		pylab.title(r'%s'%title)
		pylab.xlabel('n')
		label_list = ['Xn_list', 'Xn_hat_list', 'Yn_list']
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=200)
		pylab.savefig('%s.eps'%figure_fname, dpi=200)
		pylab.savefig('%s.png'%figure_fname, dpi=200)
		#pylab.show()
	
	def calculate_mse(self, Xn_list, Xn_hat_list):
		import Numeric
		diff_array = Numeric.array(Xn_list)-Numeric.array(Xn_hat_list)
		mse = sum(diff_array*diff_array)/(len(Xn_list))
		return mse
	
	def test_simulate_filter(self):
		chain_length = 251
		a = 1.004
		b = 0.06
		sigma = 2
		Vn_list, Xn_list, Wn_list, Yn_list = self.simulate(self.sample_X_0, self.sample_W, self.sample_W, chain_length, a, b, sigma)
		print 'Vn_list'
		print Vn_list
		print 'Xn_list'
		print Xn_list
		print 'Wn_list'
		print Wn_list
		print 'Yn_list'
		print Yn_list
		"""
		X_integral_region = [-20, 30]
		gap = 0.15	#gap=0.5 could easily results in zero float division in normal_density()
		Xn_hat_list = self.filter_by_discretization(Yn_list, self.prob_Y_given_X, self.prob_X_given_X, self.prob_X_0, X_integral_region, gap, a, b, sigma)
		#X_hat_list = self.filter(Yn_list, self.prob_func_Y_given_X, self.prob_func_X_0, self.prob_func_X_given_X, X_integral_region, a, b, sigma)
		mse = self.calculate_mse(Xn_list, Xn_hat_list)
		print 'Xn_hat_list:', Xn_hat_list
		print 'mse:', mse
		title = r'Hw12, No 3. continuous state space filtering. gap=%s, mse=%2.4f'%(gap, mse)
		figure_fname = 'hw12_3_gap_%s'%(gap)
		self.plot(Xn_list, Xn_hat_list, Yn_list, title, figure_fname)
		"""
		no_of_samplings = int(1E4)
		Xn_hat_list = self.filter_by_SIR(Yn_list, self.sample_X_given_X, self.sample_X_0, self.prob_Y_given_X, a=1.004, b=0.06, sigma=2, no_of_samplings=no_of_samplings)
		mse = self.calculate_mse(Xn_list, Xn_hat_list)
		print 'Xn_hat_list:', Xn_hat_list
		print 'mse:', mse
		title = r'Hw12, No 3. continuous state space filtering by SIR(%s), mse=%2.4f'%(no_of_samplings, mse)
		figure_fname = 'hw12_3_SIR_%s'%(no_of_samplings)
		self.plot(Xn_list, Xn_hat_list, Yn_list, title, figure_fname)

class Math508_Final_Exam_3(unittest.TestCase):
	"""
	2007-05-06
	"""
	def setUp(self):
		print
	
	def sample_W(self):
		import random
		return random.gauss(0,1)
	
	def sample_X_0(self):
		import random
		return random.gauss(0, 0.1)
	
	def simulate(self, sample_X_0, sample_V, sample_W, chain_length, h=0.01):
		import sys, math
		sys.stderr.write("Simulating ...")
		Xn_list = []
		Vn_list = []
		Yn_list = []
		Wn_list = []
		for i in range(chain_length):
			Vn = sample_V()
			if i==0:
				Xn = sample_X_0()
			else:
				Xn = Xn_list[i-1]+0.1*math.cos(2*Xn_list[i-1])*h + 0.14*math.sqrt(h)*Vn
			Wn = sample_W()
			Yn = math.atan(Xn)*h + 0.04*math.sqrt(h)*Wn
			Vn_list.append(Vn)
			Xn_list.append(Xn)
			Wn_list.append(Wn)
			Yn_list.append(Yn)
		sys.stderr.write("Done.\n")
		return Vn_list, Xn_list, Wn_list, Yn_list
	
	def normal_density(self, X, mean, sd):
		"""
		2007-04-23
			rpy.r.dnorm() sometimes return NaN
		"""
		import math
		return 1/(sd*math.sqrt(2*math.pi))*math.exp(-(X-mean)*(X-mean)/(2.0*sd*sd))
	
	def cal_inverse_lambda_func(self, Y, X, h, sigma_square):
		import math
		c_r = math.atan(X)*h
		exponent = c_r*Y/sigma_square - (c_r*c_r)/(2*sigma_square)
		return math.exp(exponent)
	
	def prob_X_given_X(self, old_X, new_X, h=0.01):
		import math
		return self.normal_density(new_X, old_X+0.1*math.cos(2*old_X)*h, abs(0.14*math.sqrt(h)))
	
	def prob_X_0(self, X):
		return self.normal_density(X, 0, 0.1)
	
	def filter_by_discretization(self, Yn_list, cal_inverse_lambda_func, prob_X_given_X, prob_X_0, X_integral_region=[-1, 1], gap=0.01, h=0.01):
		"""
		2007-05-06
			Yn_list[0] is 0 (place holder)
			use discretization and Riemann-sum to solve the integral
		"""
		import sys, Numeric
		sys.stderr.write("Filtering ...\n")
		X_array = []
		x_i = X_integral_region[0]
		while x_i < X_integral_region[1]:
			X_array.append(x_i)
			x_i+= gap
		no_of_X_states = len(X_array)
		no_of_Yns = len(Yn_list)
		#the last row, no_of_Yns+1 is only for Hn_hat_list, not for Xn_hat_list
		gamma_array = Numeric.zeros([no_of_Yns, no_of_X_states], Numeric.Float)
		numerator = 0.0
		denominator = 0.0
		Xn_hat_list = []
		max_post_prob_Xn_hat_list = []
		#note H_n = \sum_{k=1}^{k=n} X_{k-1}, so H_{n+1} would include X_n
		gamma_H_array = Numeric.zeros([no_of_Yns, no_of_X_states], Numeric.Float)
		Hn_hat_list = [0]	#the first one is 0
		
		#the normalized FDF
		pi_array = Numeric.zeros([no_of_Yns-1, no_of_X_states], Numeric.Float)
		
		sigma_square = 0.04*0.04*h	#for cal_inverse_lambda_func
		#import pdb
		#pdb.set_trace()
		for i in range(no_of_X_states):	#gamma_H_array[0] is all zero, already intialized
			gamma_array[0, i] = prob_X_0(X_array[i])
			numerator += X_array[i]*gamma_array[0,i]
			denominator += gamma_array[0,i]
		max_post_prob_Xn_index = Numeric.argmax(gamma_array[0])
		max_post_prob_Xn_hat_list.append(X_array[max_post_prob_Xn_index])
		Xn_hat_list.append(numerator/denominator)
		pi_array[0] = gamma_array[0]/denominator
		#pdb.set_trace()
		for i in range(1,no_of_Yns):
			numerator = 0.0
			denominator = 0.0
			numerator_H = 0.0
			#denominator_H = 0.0	#same as denominator
			for j in range(no_of_X_states):
				for k in range(no_of_X_states):
					p_x_given_x = prob_X_given_X(X_array[k], X_array[j])
					inverse_lambda = cal_inverse_lambda_func(Yn_list[i], X_array[k], h, sigma_square)	#index of Yn_list is i
					gamma_array[i,j] += p_x_given_x*inverse_lambda*gamma_array[i-1,k]*gap
					gamma_H_array[i,j] += p_x_given_x*inverse_lambda*(gamma_H_array[i-1,k] + X_array[k]*gamma_array[i-1,k])*gap
				numerator += X_array[j]*gamma_array[i,j]
				numerator_H += gamma_H_array[i,j]
				denominator += gamma_array[i,j]
			if i!=no_of_Yns-1:	#Xn_hat_list doesn't need the last one, which is X_{n+1}
				max_post_prob_Xn_index = Numeric.argmax(gamma_array[i])
				max_post_prob_Xn_hat_list.append(X_array[max_post_prob_Xn_index])
				Xn_hat_list.append(numerator/denominator)
				pi_array[i] = gamma_array[i]/denominator
			Hn_hat_list.append(numerator_H/denominator)
			sys.stderr.write("%s%s"%('\x08'*10, i))
		sys.stderr.write("Done.\n")
		return Xn_hat_list, Hn_hat_list, max_post_prob_Xn_hat_list, pi_array
	
	def filter_by_discretization_corrected(self, Yn_list, cal_inverse_lambda_func, prob_X_given_X, prob_X_0, X_integral_region=[-1, 1], gap=0.01, h=0.01):
		"""
		2007-05-06
			use discretization and Riemann-sum to solve the integral
			difference from filter_by_discretization()
				filter_by_discretization() is Xn|Y_[1,n], (one step of prediction considering the dependency structure, Y_{n+1} depends on Xn
				filter_by_discretization_corrected() is Xn|Y_[1,n+1]
			
			detail formulas see final_answer.tex
		"""
		import sys, Numeric
		sys.stderr.write("Filtering ...\n")
		X_array = []
		x_i = X_integral_region[0]
		while x_i < X_integral_region[1]:
			X_array.append(x_i)
			x_i+= gap
		no_of_X_states = len(X_array)
		no_of_Yns = len(Yn_list)
		#the last row, no_of_Yns+1 is only for Hn_hat_list, not for Xn_hat_list
		gamma_array = Numeric.zeros([no_of_Yns, no_of_X_states], Numeric.Float)
		numerator = 0.0
		denominator = 0.0
		Xn_hat_list = []
		max_post_prob_Xn_hat_list = []
		#note H_n = \sum_{k=1}^{k=n} X_{k-1}, so H_{n+1} would include X_n
		gamma_H_array = Numeric.zeros([no_of_Yns, no_of_X_states], Numeric.Float)
		Hn_hat_list = [0]	#the first one is 0
		#import pdb
		#pdb.set_trace()
		
		#the normalized FDF
		pi_array = Numeric.zeros([no_of_Yns-1, no_of_X_states], Numeric.Float)
		
		sigma_square = 0.04*0.04*h	#for cal_inverse_lambda_func
		
		for i in range(no_of_X_states):	#gamma_H_array[0] is all zero, already intialized
			gamma_array[0, i] = cal_inverse_lambda_func(Yn_list[1], X_array[i], h, sigma_square)*prob_X_0(X_array[i])	#index for Yn_list is 1
			numerator += X_array[i]*gamma_array[0,i]
			denominator += gamma_array[0,i]
		max_post_prob_Xn_index = Numeric.argmax(gamma_array[0])
		max_post_prob_Xn_hat_list.append(X_array[max_post_prob_Xn_index])
		Xn_hat_list.append(numerator/denominator)
		pi_array[0] = gamma_array[0]/denominator
		#pdb.set_trace()
		
		for i in range(1,no_of_Yns):
			numerator = 0.0
			denominator = 0.0
			numerator_H = 0.0
			denominator_H = 0.0	#different from denominator
			if i!=no_of_Yns-1:	#Xn_hat_list doesn't need the last one, which is X_{n+1}
				for j in range(no_of_X_states):
					for k in range(no_of_X_states):
						p_x_given_x = prob_X_given_X(X_array[k], X_array[j])
						inverse_lambda = cal_inverse_lambda_func(Yn_list[i+1], X_array[j], h, sigma_square)	#note: i+1 is the index of Yn_list
						gamma_array[i,j] += p_x_given_x*inverse_lambda*gamma_array[i-1,k]*gap
					numerator += X_array[j]*gamma_array[i,j]
					denominator += gamma_array[i,j]
				max_post_prob_Xn_index = Numeric.argmax(gamma_array[i])
				max_post_prob_Xn_hat_list.append(X_array[max_post_prob_Xn_index])
				Xn_hat_list.append(numerator/denominator)
				pi_array[i] = gamma_array[i]/denominator
			for j in range(no_of_X_states):
				for k in range(no_of_X_states):
					p_x_given_x = prob_X_given_X(X_array[k], X_array[j])
					inverse_lambda = cal_inverse_lambda_func(Yn_list[i], X_array[j], h, sigma_square)	#note: i is the index of Yn_list
					gamma_H_array[i,j] += p_x_given_x*inverse_lambda*gamma_H_array[i-1,k]*gap 
				gamma_H_array[i,j] += X_array[j]*gamma_array[i-1,j]
				numerator_H += gamma_H_array[i,j]
				denominator_H += gamma_array[i-1,j]
			Hn_hat_list.append(numerator_H/denominator_H)
			sys.stderr.write("%s%s"%('\x08'*10, i))
		sys.stderr.write("Done.\n")
		return Xn_hat_list, Hn_hat_list, max_post_prob_Xn_hat_list, pi_array
	
	def cal_U_list_given_Y_list(self, Yn_list):
		"""
		2007-05-06
			U_0 = 0
			Y_{n+1} = U_{n+1} - U_n
		"""
		Un_list = [0]
		for i in range(1, len(Yn_list)):
			Un_list.append(Un_list[i-1]+Yn_list[i])
		return Un_list
	
	def cal_H_list_given_X_list(self, Xn_list):
		"""
		2007-05-06
			H_n = \sum_{k=1}^{k=n} X_{k-1}, so H_{n+1} would include X_n
			H_0=0
		"""
		Hn_list = [0]
		for i in range(1, len(Xn_list)+1):
			Hn_list.append(Hn_list[i-1] + Xn_list[i-1])
		return Hn_list
	
	def plot(self, list_of_data_ls, label_list, title, figure_fname):
		import pylab, Numeric
		pylab.clf()
		x_index_list = range(len(list_of_data_ls[0]))
		color_list = ['b','g','r']
		for i in range(len(list_of_data_ls)):
			pylab.plot(x_index_list, list_of_data_ls[i], color_list[i])
		pylab.title(r'%s'%title)
		pylab.xlabel('n')
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=200)
		pylab.savefig('%s.eps'%figure_fname, dpi=200)
		pylab.savefig('%s.png'%figure_fname, dpi=200)
		pylab.show()
	
	def plot3D_pi_array(self, pi_array, X_integral_region, gap, title, figure_fname):
		"""
		2007-05-07
			matplotlib version >= 0.87.5
		"""
		import pylab
		import matplotlib.axes3d as p3
		pylab.clf()
		n = pylab.arange(pi_array.shape[0])
		x = pylab.arange(X_integral_region[0], X_integral_region[1], gap)
		X,N = pylab.meshgrid(x,n)
		fig=pylab.figure()
		ax = p3.Axes3D(fig)
		ax.plot_wireframe(N, X, pi_array)
		ax.set_xlabel('n')
		ax.set_ylabel('X')
		ax.set_zlabel('pi')
		pylab.title(title)
		pylab.savefig('%s.svg'%figure_fname, dpi=200)
		pylab.savefig('%s.eps'%figure_fname, dpi=200)
		pylab.savefig('%s.png'%figure_fname, dpi=200)
		pylab.show()
	
	def calculate_mse(self, Xn_list, Xn_hat_list):
		import Numeric
		diff_array = Numeric.array(Xn_list)-Numeric.array(Xn_hat_list)
		mse = sum(diff_array*diff_array)/(len(Xn_list))
		return mse
	
	def filter_wrapper(self, filter_func, Xn_list, Yn_list, h, figure_output_fname_prefix='Final_3'):
		X_integral_region = [-1, 1]
		gap = 0.01
		Xn_hat_list, Hn_hat_list, max_post_prob_Xn_hat_list, pi_array = filter_func(Yn_list, self.cal_inverse_lambda_func, self.prob_X_given_X, self.prob_X_0, X_integral_region, gap, h)
		
		mse = self.calculate_mse(Xn_list, Xn_hat_list)
		print 'Xn_hat_list:', Xn_hat_list
		print 'mse:', mse
		title = r'%s cont. state space filtering. gap=%s, mse=%2.4f'%(figure_output_fname_prefix, gap, mse)
		figure_fname = '%s_Xn_Un_Xn_hat_gap_%s'%(figure_output_fname_prefix, gap)
		Un_list = self.cal_U_list_given_Y_list(Yn_list)
		real_Un_list = Un_list[1:]	#discard the 1st 0
		list_of_data_ls = [Xn_list, real_Un_list, Xn_hat_list]
		label_list = ['Xn', 'Un', 'Xn_hat']
		self.plot(list_of_data_ls, label_list, title, figure_fname)
		
		Hn_list = self.cal_H_list_given_X_list(Xn_list)
		mse = self.calculate_mse(Hn_list, Hn_hat_list)
		print 'Hn_hat_list:', Hn_hat_list
		print 'mse:', mse
		title = r'%s cont. state space filtering. gap=%s, mse=%2.4f'%(figure_output_fname_prefix, gap, mse)
		figure_fname = '%s_Hn_Hn_hat_gap_%s'%(figure_output_fname_prefix, gap)
		list_of_data_ls = [Hn_list, Hn_hat_list]
		label_list = ['Hn', 'Hn_hat']
		self.plot(list_of_data_ls, label_list, title, figure_fname)
		
		mse = self.calculate_mse(Xn_list, max_post_prob_Xn_hat_list)
		print 'max_post_prob_Xn_hat_list:', max_post_prob_Xn_hat_list
		print 'mse:', mse
		title = r'%s cont. state space filtering. gap=%s, mse=%2.4f'%(figure_output_fname_prefix, gap, mse)
		figure_fname = '%s_Xn_Xn_hat_Xn_hat_max_gap_%s'%(figure_output_fname_prefix, gap)
		list_of_data_ls = [Xn_list, Xn_hat_list, max_post_prob_Xn_hat_list]
		label_list = ['Xn', 'Xn_hat', 'Xn_hat_max']
		self.plot(list_of_data_ls, label_list, title, figure_fname)
		
		title = '%s_3D of FDF'%figure_output_fname_prefix
		figure_fname = '%s_3d_pi_FDF'%figure_output_fname_prefix
		self.plot3D_pi_array(pi_array, X_integral_region, gap, title, figure_fname)
	
	def test_simulate_filter(self):
		chain_length = 201
		h = 0.01
		Vn_list, Xn_list, Wn_list, real_Yn_list = self.simulate(self.sample_X_0, self.sample_W, self.sample_W, chain_length, h)
		Yn_list = [0] + real_Yn_list	#to be compatible with the problem
		print 'Vn_list'
		print Vn_list
		print 'Xn_list'
		print Xn_list
		print 'Wn_list'
		print Wn_list
		print 'Yn_list'
		print Yn_list
		figure_output_fname_prefix = 'Final_3'
		self.filter_wrapper(self.filter_by_discretization, Xn_list, Yn_list, h, figure_output_fname_prefix)
		figure_output_fname_prefix = 'Final_3_corrected'
		self.filter_wrapper(self.filter_by_discretization_corrected, Xn_list, Yn_list, h, figure_output_fname_prefix)

class Math508_Final_Exam_4(unittest.TestCase):
	"""
	2007-05-07
	"""
	def setUp(self):
		print
	
	def sample_W(self):
		import random
		return random.gauss(0,1)
	
	def simulate(self, t_list, a, b, B, sample_V, sample_W, delta_t=0.01):
		import sys, math
		sys.stderr.write("Simulating ...")
		Xn_list = [1]
		Vn_list = []
		Yn_list = [0]
		Wn_list = []
		for i in range(1, len(t_list)):
			Vn = sample_V()
			Xn = Xn_list[i-1]*(1+a*delta_t) + b*math.sqrt(delta_t)*Vn
			Wn = sample_W()
			Yn = Yn_list[i-1] + Xn_list[i-1]*delta_t + B*math.sqrt(delta_t)*Wn
			Vn_list.append(Vn)
			Xn_list.append(Xn)
			Wn_list.append(Wn)
			Yn_list.append(Yn)
		sys.stderr.write("Done.\n")
		return Vn_list, Xn_list, Wn_list, Yn_list
	
	def normal_density(self, X, mean, sd):
		"""
		2007-04-23
			rpy.r.dnorm() sometimes return NaN
		"""
		import math
		return 1/(sd*math.sqrt(2*math.pi))*math.exp(-(X-mean)*(X-mean)/(2.0*sd*sd))
	
	def prob_X_given_X(self, old_X, new_X, h=0.01):
		import math
		return self.normal_density(new_X, old_X+0.1*math.cos(2*old_X)*h, abs(0.14*math.sqrt(h)))
	
	def prob_X_0(self, X):
		return self.normal_density(X, 0, 0.1)
	
	def cal_alpha_ratio(self, t_i_plus_1, t_i, a, B, lambda_1, delta_t, K, C):
		import math
		return math.exp((a-lambda_1/(B*B))*delta_t)*math.exp(delta_t*C)*(K*math.exp(t_i/C)+1)/(K*math.exp(t_i_plus_1/C)+1)
	
	def filter_by_discretization(self, Yn_list, cal_alpha_ratio, t_list, a, b, B, delta_t=0.01):
		"""
		2007-05-07
			use discretization and Riemann-sum to solve the integral
		"""
		import sys, math
		sys.stderr.write("Filtering ...\n")
		no_of_Yns = len(Yn_list)
		Xn_hat_list = [1]	#X0=1 and \hat{X_0} = EX0 = 1
		Xn_hat_with_exact_alpha_ratio_list = [1]
		Pn_list = [0]	#X_0 is constant, so P_0 is 0
		
		a_B_sqr = a*B*B
		lambda_common = a_B_sqr*a_B_sqr + b*b*B*B
		lambda_1 = math.sqrt(lambda_common)+a_B_sqr
		lambda_2 = -math.sqrt(lambda_common)+a_B_sqr
		C = B*B/(lambda_1-lambda_2)
		K = -lambda_1/lambda_2	#different from the final exam
		approx_alpha_ratio_list = []
		exact_alpha_ratio_list1 = []
		exact_alpha_ratio_list2 = []
		for i in range(1, len(t_list)):
			approx_alpha_ratio = math.exp((a-Pn_list[i-1]/(B*B))*delta_t)
			approx_alpha_ratio_list.append(approx_alpha_ratio)
			exact_alpha_ratio1 = self.cal_alpha_ratio(t_list[i], t_list[i], a, B, lambda_1, delta_t, K, C)
			exact_alpha_ratio2 = self.cal_alpha_ratio(t_list[i], t_list[i], a, B, lambda_2, delta_t, K, C)
			exact_alpha_ratio_list1.append(exact_alpha_ratio1)
			exact_alpha_ratio_list2.append(exact_alpha_ratio2)
			
			Xn_hat_list.append(approx_alpha_ratio*(Xn_hat_list[i-1]+Pn_list[i-1]/(B*B)*(Yn_list[i]-Yn_list[i-1])))
			Xn_hat_with_exact_alpha_ratio_list.append(exact_alpha_ratio1*(Xn_hat_list[i-1]+Pn_list[i-1]/(B*B)*(Yn_list[i]-Yn_list[i-1])))
			delta_P = 2*a*Pn_list[i-1] + b*b - (Pn_list[i-1]*Pn_list[i-1])/(B*B)
			Pn_list.append(delta_P*delta_t + Pn_list[i-1])
			sys.stderr.write("%s%s"%('\x08'*10, i))
		sys.stderr.write("Done.\n")
		return Xn_hat_list, Xn_hat_with_exact_alpha_ratio_list, Pn_list, approx_alpha_ratio_list, exact_alpha_ratio_list1, exact_alpha_ratio_list2
	
	def plot(self, list_of_data_ls, label_list, title, figure_fname):
		import pylab, Numeric
		pylab.clf()
		x_index_list = range(len(list_of_data_ls[0]))
		color_list = ['b','g','r', 'c']
		for i in range(len(list_of_data_ls)):
			pylab.plot(x_index_list, list_of_data_ls[i], color_list[i])
		pylab.title(r'%s'%title)
		pylab.xlabel('n')
		pylab.legend(label_list)
		pylab.savefig('%s.svg'%figure_fname, dpi=200)
		pylab.savefig('%s.eps'%figure_fname, dpi=200)
		pylab.savefig('%s.png'%figure_fname, dpi=200)
		pylab.show()
	
	def calculate_mse(self, Xn_list, Xn_hat_list):
		import Numeric
		diff_array = Numeric.array(Xn_list)-Numeric.array(Xn_hat_list)
		mse = sum(diff_array*diff_array)/(len(Xn_list))
		return mse
	
	def simulate_filter_wrapper(self, t_list, delta_t, a, b, B, figure_output_fname_prefix='Final_4'):
		
		Vn_list, Xn_list, Wn_list, Yn_list = self.simulate(t_list, a, b, B, self.sample_W, self.sample_W, delta_t)
		print 'Vn_list'
		print Vn_list
		print 'Xn_list'
		print Xn_list
		print 'Wn_list'
		print Wn_list
		print 'Yn_list'
		print Yn_list
		
		Xn_hat_list, Xn_hat_with_exact_alpha_ratio_list, Pn_list, approx_alpha_ratio_list, exact_alpha_ratio_list1, exact_alpha_ratio_list2 = self.filter_by_discretization(Yn_list, self.cal_alpha_ratio, t_list, a, b, B, delta_t)
		
		mse1 = self.calculate_mse(Xn_list, Xn_hat_list)
		mse2 = self.calculate_mse(Xn_list, Xn_hat_with_exact_alpha_ratio_list)
		print 'Xn_hat_list:', Xn_hat_list
		print 'Xn_hat_with_exact_alpha_ratio_list:', Xn_hat_with_exact_alpha_ratio_list
		print 'mse1:', mse1
		print 'mse2:', mse2
		
		title = r'%s K-B filter. a=%s b=%s B=%s gap=%s  mse1=%2.4f mse2=%2.4f'%(figure_output_fname_prefix, a, b, B, delta_t, mse1, mse2)
		figure_fname = '%s_Xn_Yn_Xn_hat_a_%s_b_%s_B_%s_gap_%s'%(figure_output_fname_prefix, a, b, B, delta_t)
		list_of_data_ls = [Xn_list, Yn_list, Xn_hat_list, Xn_hat_with_exact_alpha_ratio_list]
		label_list = ['Xn', 'Yn', 'Xn_hat', 'Xn_hat_exact']
		self.plot(list_of_data_ls, label_list, title, figure_fname)
		
		mse1 = self.calculate_mse(approx_alpha_ratio_list, exact_alpha_ratio_list1)
		mse2 = self.calculate_mse(approx_alpha_ratio_list, exact_alpha_ratio_list2)
		print 'approx_alpha_ratio_list:', approx_alpha_ratio_list
		print 'exact_alpha_ratio_list1:', exact_alpha_ratio_list1
		print 'exact_alpha_ratio_list2:', exact_alpha_ratio_list2
		print 'mse1:', mse1
		print 'mse2:', mse2
		title = r'%s a_ratio approx vs exact. a=%s b=%s B=%s gap=%s, mse1=%2.4f, mse2=%2.4f'%(figure_output_fname_prefix, a, b, B, delta_t, mse1, mse2)
		figure_fname = '%s_alpha_ratio_comp_a_%s_b_%s_B_%s_gap_%s'%(figure_output_fname_prefix, a, b, B, delta_t)
		list_of_data_ls = [approx_alpha_ratio_list, exact_alpha_ratio_list1, exact_alpha_ratio_list2]
		label_list = ['approx_alpha_ratio', 'exact_alpha_ratio1', 'exact_alpha_ratio2']
		self.plot(list_of_data_ls, label_list, title, figure_fname)
	
	def test_simulate_filter(self):
		t_region = [0, 10]
		delta_t = 0.01
		t_list = []
		t_i = t_region[0]
		while t_i <= t_region[1]:
			t_list.append(t_i)
			t_i += delta_t
		#import pdb
		#pdb.set_trace()
		a = 0.1
		b = 1
		B = 0.3
		self.simulate_filter_wrapper(t_list, delta_t, a, b, B)
		
		a = 0.1
		b = 1
		B = 10
		self.simulate_filter_wrapper(t_list, delta_t, a, b, B)
		
		a = -1
		b = 1
		B = 0.3
		self.simulate_filter_wrapper(t_list, delta_t, a, b, B)
		
		a = -1
		b = 2
		B = 0.3
		self.simulate_filter_wrapper(t_list, delta_t, a, b, B)

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
		2: Math508_HW2_2,
		3: Math508_HW3_1,
		4: Math508_HW5_1,
		5: Math508_HW5_2,
		6: Math508_HW6_1,
		7: Math508_HW6_2,
		8: Math508_HW7_1,
		9: Math508_HW7_2,
		10: Math508_HW8_a,
		11: Math508_HW8_b,
		12: Math508_HW10_1,
		13: Math508_HW10_2,
		14: Math508_HW11_3,
		15: Math508_HW12_3,
		16: Math508_Final_Exam_3,
		17: Math508_Final_Exam_4}
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
