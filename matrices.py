class Matrix:
	def __init__(self, *args):
		"""
		creates a matrix.
		accepts either a tuple containing the dimensions of the matrix, or a list containing the matrix entries.
		"""
		if len(args)==1:
			if isinstance(args[0], list) and isinstance(args[0][0], list):
				#it must be a list of all the rows
				#e.g [[1, 2, 3], [4, 5, 6]] gives
				# 1 2 3
				# 4 5 6
				#as a matrix
				self.list = args[0]
				self.rows, self.columns = self.get_dimensions()
				self.dimensions = (self.rows, self.columns)
				self.is_square = self.rows==self.columns
			elif isinstance(args[0], tuple):
				self.list = [[0 for _ in range(args[0][1])] for _ in range(args[0][0])]
				self.rows, self.columns = args[0]
				self.dimensions = args[0]
				self.is_square = self.rows==self.columns
	
	def copy(self):
		"""
		returns a copy of a given matrix.
		"""
		#why can't python assignment be normal
		A = Matrix(self.dimensions)
		for i in range(self.rows):
			for j in range(self.columns):
				A[i,j] = self[i,j]
		return A
	
	def __eq__(self, other):
		"""tests if two matrices are equal."""
		if self.dimensions!=other.dimensions: return False
		for i in range(self.rows):
			for j in range(self.columns):
				if self[i,j]!=other[i,j]: return False
		return True
	
	def __ne__(self, other):
		"""tests if two matrices are not equal."""
		return not self==other
	
	def __str__(self):
		max_len = 0
		for i in range(self.rows):
			for j in range(self.columns):
				if len(str(self[i,j]))>max_len:
					max_len = len(str(self[i,j]))
		
		string = ""
		for i in range(self.rows):
			string += "["
			for j in range(self.columns):
				string += str(self[i,j])+(" "*(max_len-len(str(self[i,j])))) + " "
			string = string[:-1] + "]\n"
		return string
	
	def __repr__(self):
		return str(self.list)
	
	def __list__(self):
		return self.list
	
	def __getitem__(self, indices):
		if isinstance(indices[0], int):
			return self.list[indices[0]][indices[1]]
		elif isinstance(indices[0], slice):
			row_start = indices[0].start
			column_start = indices[1].start
			row_stop = indices[0].stop
			column_stop = indices[1].stop
			row_step = indices[0].step
			column_step = indices[1].step
			
			if row_step is None: row_step = 1
			if column_step is None: column_step = 1
			if row_start is None: row_start = 0 if row_step>0 else self.dimensions[0]-1
			if row_stop is None: row_stop = self.dimensions[0] if row_step>0 else -1
			if column_start is None: column_start = 0 if column_step>0 else self.dimensions[1]-1
			if column_stop is None: column_stop = self.dimensions[1] if column_step>0 else -1
			
			A = Matrix((abs(row_stop-row_start), abs(column_stop-column_start)))
			
			for i in range(row_start, row_stop, row_step):
				for j in range(column_start, column_stop, column_step):
					A[abs(i-row_start), abs(j-column_start)] = self[i,j]
			
			return A	
	
	def __setitem__(self, indices, value):
		self.list[indices[0]][indices[1]] = value
	
	def fraction(self):
		"""converts all matrix entries to fractions."""
		from fractions import Fraction
		A = self.copy()
		for i in range(A.rows):
			for j in range(A.columns):
				A[i,j] = Fraction(A[i,j])
		return A
	
	def getrow(self, index):
		"""returns specified row of a matrix."""
		row = [[]]
		for i in range(self.columns):
			row[0].append(self.list[index][i])
		return Matrix(row)
	
	def setrow(self, index, value):
		"""sets specified row of a matrix to given list."""
		if not isinstance(value, list) or len(value)!=self.columns:
			raise ValueError("cannot set {value} as row {index}".format(value=value, index=index))
		self.list[index] = value
	
	def getcolumn(self, index):
		"""gets specified column vector of a matrix."""
		column = []
		for i in range(self.rows):
			column.append([self[i,index]])
		return Matrix(column)
	
	def setcolumn(self):
		#TODO
		pass
	
	def get_dimensions(self):
		"""returns a tuple of the dimensions of the matrix."""
		num_rows = len(self.list)
		num_columns = len(self.list[0])
		for i in self.list:
			if len(i)!=num_columns:
				raise ValueError("cannot use a non-rectangular matrix")
		return num_rows, num_columns
	
	def augment(self, other):
		"""augments a matrix with another."""
		if self.rows != other.rows:
			raise ValueError("Cannot augment two matrices with different amounts of rows")
		else:
			A = Matrix((self.rows, self.columns+other.columns))
			for i in range(self.rows):
				for j in range(self.columns):
					A[i,j] = self[i,j]
			for i in range(other.rows):
				for j in range(other.columns):
					A[i,j+self.columns] = other[i,j]
			return A
	
	def vertical_augment(self, other):
		"""vertically augments a given matrix with another."""
		if self.columns != other.columns:
			raise ValueError("Cannot vertically augment two matrices with different amounts of columns")
		else:
			A = self.list
			for i in range(other.rows):
				A.append(other.list[i])
			return Matrix(A)
	
	def index_of_leading_term(self, row):
		"""
		returns the index of the leading term for a given row.
		if the row is all zeros, it returns None.
		"""
		for i, n in enumerate(self.list[row]):
			if n!=0: return i
		return None
	
	def is_zero_row(self, row):
		"""tests if a given row is completely full of zeros."""
		for i in range(self.columns):
			if self[row,i]!=0: return False
		return True
	
	def swap_rows(self, r1, r2):
		"""swaps two given rows of a matrix."""
		self.list[r1], self.list[r2] = self.list[r2], self.list[r1]
	
	def multiply_row_by_scalar(self, row, num):
		"""multiplies a given row by a scalar."""
		self.list[row] = [num*i for i in self.list[row]]
	
	def add_multiply_two_rows(self, r1, r2, a):
		"""adds a times row 2 to row 1."""
		self.list[r1] = [ self.list[r1][i] + a*self.list[r2][i] for i in range(self.columns)]
	
	def row_echelon_form(self, return_determinant=False):
		"""
		returns the row-echelon form of a given matrix.
		if return_determinant is True, it returns the determinant of the matrix instead, because it uses the exact same algorithm.
		"""
		A = self.copy()
		Det = 1
		d = min(self.dimensions)
		
		for i in range(d):
			#get row with nonzero i-th entry and swap it with row i
			if A[i,i]==0:
				for r in range(i, A.rows):
					if A[i,r]!=0:
						A.swap_rows(i, r)
						Det *= -1
			
			try: Det *= A[i,A.index_of_leading_term(i)]; A.multiply_row_by_scalar(i, 1/A[i, A.index_of_leading_term(i)])
			except TypeError: continue
			#print(A, "\n")
			
			for j in range(i+1, d):
				try:
					A.add_multiply_two_rows(j, i, -A[j, A.index_of_leading_term(i)])
				except TypeError: continue
				#print(A, "\n")
		
		if return_determinant: return Det * A.diagonal_product()
		return A
	
	def reduced_row_echelon_form(self):
		"""returns the reduced row-echelon form of the given matrix."""
		A = self.row_echelon_form()
		d = min(self.dimensions)
		#converts ref into rref
		for i in range(d-2, -1, -1):
			for j in range(d-1, i, -1):
				try:
					A.add_multiply_two_rows(i, j, -A[i, A.index_of_leading_term(j)])
				except TypeError: continue
		
		return A
	
	def rank(self):
		"""returns the rank of the given matrix."""
		A = self.reduced_row_echelon_form()
		r = 0
		for i in range(A.rows):
			if not A.is_zero_row(i): r+=1
		return r
	
	def nullity(self):
		"""returns the nullity of a given matrix."""
		return self.columns - self.rank()
	
	def dot_product(self, other):
		"""returns the dot product of a given 1 by n and n by 1 matrix."""
		#so far self MUST be a 1 by n matrix (e.g [[2, 1]]) and other MUST be an n by 1 matrix (e.g [[3], [4]])
		if self.columns!=other.rows or self.rows!=1:
			raise ValueError("can only have dot product between 1 by n and n by 1 vectors")
		else:
			D = 0
			for i in range(self.columns):
				D += self.list[0][i]*other.list[i][0]
			return D
	
	def diagonal_product(self):
		"""returns the product of the main diagonal of a matrix. used when computing the determinant."""
		p = 1
		for i in range(min(self.dimensions)):
			p *= self[i,i]
		return p
	
	def trace(self):
		"""computes the diagonal sum of a given matrix. also the sum of the eigenvalues of the matrix."""
		if not self.is_square:
			raise ValueError("Cannot compute trace of a non-square matrix")
		else:
			t = 0
			for i in range(self.dimensions[0]):
				t += self[i,i]
			return t
	
	def __add__(self, other):
		"""adds two given matrices."""
		if self.rows!=other.rows or self.columns!=other.rows:
			raise ValueError("cannot add matrices with different sizes")
		else:
			S = Matrix(self.dimensions)
			for i in range(self.rows):
				for j in range(self.columns):
					S[i,j] = self[i,j] + other[i,j]
			return S
	
	def __sub__(self, other):
		"""subtracts a matrix from another matrix."""
		if self.rows!=other.rows or self.columns!=other.rows:
			raise ValueError("cannot subtract matrices with different sizes")
		else:
			S = Matrix(self.dimensions)
			for i in range(self.rows):
				for j in range(self.columns):
					S[i,j] = self[i,j] - other[i,j]
			return S
	
	def __mul__(self, other):
		"""multiplies a matrix by another matrix."""
		if type(other)==Matrix and self.columns!=other.rows:
			raise ValueError("cannot multiply matrices with wrong input dimensions")
		else:
			if type(other)==Matrix:
				P = Matrix((self.rows, other.columns))
				for i in range(self.rows):
					for j in range(other.columns):
						P[i,j] = self.getrow(i).dot_product(other.getcolumn(j))
				return P
			else:
				P = Matrix((self.rows, self.columns))
				for i in range(self.rows):
					for j in range(self.columns):
						P[i,j] = self[i,j] * other
				return P
	
	def __rmul__(self, other):
		"""multiplies a matrix by another matrix."""
		P = Matrix((self.rows, self.columns))
		for i in range(self.rows):
			for j in range(self.columns):
				P[i,j] = other * self[i,j]
		return P
	
	def __pow__(self, other):
		"""raises a matrix to given power."""
		if not self.is_square:
			raise ValueError("Cannot raise a non-square matrix to a power")
		else:
			d = self.dimensions[0]
			if isinstance(other, int):
				if other>0:
					t = Matrix.identity(d)
					for _ in range(other):
						t *= self
					return t
				elif other==0:
					return Matrix.identity(d)
				elif other<0:
					t = Matrix.identity(d)
					B = self.inverse()
					for _ in range(other):
						t *= B
					return t				
	
	def minor_matrix(self, row, column):
		"""returns a matrix without given row and column."""
		return Matrix([row[:column] + row[column+1:] for row in (self.list[:row]+self.list[row+1:])])
	
	def cofactor(self, row, column):
		"""returns the cofactor for a given row and column."""
		return ((-1)**(row+column)) * self.minor_matrix(row, column).determinant()
	
	def cofactor_matrix(self):
		"""returns the cofactor matrix for a given matrix."""
		C = Matrix(self.dimensions)
		for i in range(self.rows):
			for j in range(self.columns):
				C[i,j] = self.cofactor(i, j)
		return C
	
	def slow_determinant(self):
		"""DO NOT USE. RUNS IN O(n!) TIME."""
		# runs in O(n!) time 
		# DO NOT USE!!!
		if not self.is_square:
			raise ValueError("can only find the determinant of square matrices")
		else:
			d = self.dimensions[0]
			if d==0:
				#don't forget the trivial cases
				return 1
			elif d==1:
				return self[0,0]
			elif d==2:
				ad = self[0,0]*self[1,1]
				bc = self[0,1]*self[1,0]
				return ad-bc
			else:
				t = 0
				for i in range(d):
					t += self.cofactor(0,i)*self[0,i]
				return t
	
	def determinant(self):
		"""computes the determinant of the given matrix."""
		if not self.is_square:
			raise ValueError("can only find the determinant of square matrices")
		else:
			d = self.dimensions[0]
			if d==0:
				#don't forget the trivial cases
				return 1
			elif d==1:
				return self[0,0]
			elif d==2:
				ad = self[0,0] * self[1,1]
				bc = self[0,1] * self[1,0]
				return ad - bc
			else:
				return self.row_echelon_form(return_determinant=True)
	
	def transpose(self):
		"""returns the transpose matrix of given matrix."""
		T = Matrix((self.columns, self.rows))
		for i in range(self.columns):
			for j in range(self.rows):
				T[j,i] = self[i,j]
		return T
	
	def slow_inverse(self):
		"""returns the inverse of a given matrix using the transpose of the cofactor matrix."""
		if not self.is_square:
			raise ValueError("can only find the inverse of square matrices")
		else:
			return self.cofactor_matrix().transpose() * (1 / self.determinant())
	
	def inverse(self):
		"""returns the inverse of a given matrix using gaussian elimination."""
		if not self.is_square:
			#TODO: find left/right pseudoinverses of nonsquare matrices
			raise ValueError("Cannot find inverse of a non-square matrix")
		elif self.determinant()==0:
			raise ValueError("Cannot find inverse of determinant 0 matrix")
		else:
			A = self.augment(Matrix.identity(self.dimensions[0]))
			A = A.reduced_row_echelon_form()
			return A[:,self.dimensions[0]:]
	
	def identity(dimension):
		"""returns the square identity matrix for a given dimension."""
		I = Matrix((dimension, dimension))
		for i in range(dimension):
			I[i,i] = 1
		return I
	
	def random_matrix(rows, columns, value_range=range(-10, 10, 1)):
		"""returns a randomly-generated matrix for a given size and value range."""
		from random import choice
		A = Matrix((rows, columns))
		for i in range(rows):
			for j in range(columns):
				A[i,j] = choice(value_range)
		return A
	
	def eigenvalues(self):
		#TODO: add polynomial library to this
		pass
	
	def eigenvectors(self):
		pass
	
	def eigenbasis(self):
		pass
