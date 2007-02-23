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
		pylab.show()
	
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
				if L==500:
					n = 100
					T_list = range(100,201)
					X_hat_n_T_list = self.estimate_X_hat_n_T(Yn_list, K, L, T_list, n)
					title = r'smoothing $\hat{X_{%s,T}}$ r.w. on A=[0,%s], P(W=+-%s)=1/2'%(n,K,L)
					figure_fname = 'hw6_1_a_K_%s_L_%s_n_%s'%(K,L,n)
					self.plot_smoothing_X_hat_list(T_list, Xn_list[n], X_hat_n_T_list, title, figure_fname, xlabel='T')
				
				X_hat_T_n_list = self.predict_X_hat_T_n(Yn_list, K, L, T, n_list)
				title = r'prediction $\hat{X_{%s,n}}$ r.w. on A=[0,%s], P(W=+-%s)=1/2'%(T,K,L)
				figure_fname = 'hw6_1_b_K_%s_L_%s_T_%s'%(K,L,T)
				self.plot_smoothing_X_hat_list(n_list, Xn_list[T], X_hat_T_n_list, title, figure_fname, xlabel='n')
			row = reader.next()
		del reader
	
	
	
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
		6: Math508_HW6_1}
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
