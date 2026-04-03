import numpy as np
import copy


class Matrix:

    def __init__(self, data, dtype=float):
        self.data = np.array(data, dtype=dtype)
        self.LU = None

    def __repr__(self):
        return self.data.__repr__()
    
    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return Matrix(self.data.copy())

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise Exception("Error, other object not matrix")
        
        if self.shape[1] != other.shape[0]:
            raise Exception("matrix shapes are incompatible for matrix multiplication")
        
        if len(other.shape) > 2:
            raise Exception("matrix multiplication not defined for tensors (yet)")
        
        A = self
        B = other
        
        if len(other.shape) == 2: #check for matrix
            # say A has shape (m , n) and B has (n, p)
            # we want (A@B)_ij = sum_k A_ik * B_kj, with shape of (A@B) is (m, p)
            # by creating a matrix such that C_ikj = A_ik * B_kj, the multiplication can be done in vectorized way
            # by adding dimensions: A --> A[i, k, None], B --> B[None, k, j], multiplication is done correctly
            C = A[:, :, None] * B[None, :, :]
            # Now shape of C is (m, n, p) and C_ikj = A_ik * B_kj.
            # So to get matrix product we sum over the [1] axis.
            return np.sum(C, axis=1)   
        
        elif len(other.shape) == 1: #check for a vector
            C = A[:, :] * B[None, :] #same story but then for vectors
            return np.sum(C, axis=1)

        raise Exception("Matrix shape is not properly defined")
        

    def swap_rows(self, row1: int, row2: int):
        """
        swaps all elements of row1 with row2.
        """
        self[[row2, row1]] = self[[row1, row2]]


    def inverse(self):
        """
        returns inverse of matrix, using gauss jordan elimination.
        """
        mat = self.copy()
        if mat.shape[1] != mat.shape[0]:
            raise Exception("Matrix not square, does not have inverse")
        
        N = self.shape[0] #NxN matrix
        inv = Matrix(np.eye(N))

        for i in range(N): #loop over columns,
            
            for row in range(N)[i: ]:
                if mat[row, i] != 0: #select first row with nonzero pivot entry
                    if mat[row, i] != 1:
                        inv[row, :] /= mat[row, i]
                        mat[row, :] /= mat[row, i]
                    inv.swap_rows(row, i)
                    mat.swap_rows(row, i) #swap rows such that pivot is in [i, i]
                    break

                elif row == N - 1:
                    raise Exception("Matrix is singular")
        
            for row in range(N): #subtract pivot row from all other rows with non-zero elements in column i
                if row == i:
                    continue
                if mat[row, i] == 0:
                    continue
                else:
                    inv[row, :] -= mat[row, i] * inv[i, :]
                    mat[row, :] -= mat[row, i] * mat[i, :] 

        return inv
      

    def LU_decomposition(self):
        """
        returns LU matrix from LU_decomposition of matrix (with alpha_ii = 1, so can be stored in 1 matrix!).
        """

        LU = self.copy() #copy matrix to not modify the matrix itself
        if LU.shape[1] != LU.shape[0]: #check whether matrix is square
            raise Exception("Matrix not square")
        N = LU.shape[0]

        #implicit pivoting: find the largest entry on every row, we store its inverse
        inverse_max_vals = []
        for i in range(N): #loop over rows
            max_val = 0
            for j in range(N): #loop over columns
                if np.abs(LU[i, j]) > max_val: #check whether entry contains largest pivot candidate compared to previous rows
                    max_val = np.abs(LU[i,j])
            if max_val == 0: #check for matrix singularity
                raise Exception("matrix is singular")
            inverse_max_vals.append(1/max_val)

        self.LU_indx = Matrix(range(N), dtype=int) #placeholder vector to store indices of rows with largest pivot candidates

        for k in range(N): #loop over columns k
            i_max = None
            max_val = 0

            for i in range(k, N): #loop over rows i >= k
                # check whether entry contains largest pivot candidate compared to previous rows,
                # multiplied by inverse_max corresponding to that row
                if np.abs(LU[i, k]) * inverse_max_vals[i] > max_val: 
                    i_max = i
                    max_val = np.abs(LU[i,k]) * inverse_max_vals[i]

            if i_max != k: #for each column where row index containing largest entry is not equal to column index:
                LU.swap_rows(i_max, k) #swap rows to put largest weighted entry in pivot
                self.LU_indx.swap_rows(i_max, k)

            # LU[k+1:, k] /= LU[k, k] #for each row i > k, divide by beta_kk (= LU[k, k])
            # # to get LU_ik * LU_kj we can use the same trick as we did for matrix multiplication:
            # # prod = (LU[k+1:, :, None] * LU[None, :, k+1:])
            # # prod_indexed = prod[:, k, :]
            # LU[k+1:, k+1:] -= (LU[k+1:, :, None] * LU[None, :, k+1:])[:, k, :]

            for i in range(k+1, N):
                LU[i, k] /= LU[k, k]
                LU[i, k+1:] -= LU[i,k] * LU[k, k+1:]

            self.LU = LU
        
        return LU


    def solve(self, b):
        """
        Solves for x given the equation Ax = b, where A is current matrix object.
        b maybe be passed as a vector or as a matrix.

        In case b is a matrix. The code returns a matrix where column i is the solution 
        to the equation Ax = b_i where b_i is the column vector given by the ith column of the matrix b.
        """

        if self.shape[1] != b.shape[0]:
            raise Exception("shape of b (number of rows) does not match matrix (number of columns)")
        
        N = b.shape[0]
        if self.LU is None: #check whether LU matrix has been computed before
            self.LU_decomposition()
        
        x = b.copy()
        x = x[self.LU_indx.data] #swap rows of b similar to how we swapped rows when choosing pivots for LU decomposition

        #forward substitution
        for i in range(N):
            x[i] -= np.sum(self.LU[i, :i] * x[:i])

        #backward substitution
        for i in range(N-1, -1, -1): #start at N-1, go up to and including 0, with steps -1
            x[i] = 1/self.LU[i,i] * (x[i] - np.sum(self.LU[i, i+1:]*x[i+1:]) )

        return x




        

