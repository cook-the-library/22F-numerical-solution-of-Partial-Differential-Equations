import numpy as np
from scipy.special import factorial
from scipy import sparse
import math

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)
        x = np.zeros(len(values), dtype=float)
        for i in range(len(values)-1):
            x[i] = values[i+1]-values[i]
        x[-1] = -values[-1]+values[0]+length
        self.dx = x


class DifferenceUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.grid = grid # NonUniformPeriodicGrid type
        pass
    
    def matrix(self):
        h = self.grid.dx
        deri = self.derivative_order
        # where 1 located at b vector
        conv = self.convergence_order
        if deri % 2 ==0:
            deri_1 = deri -1
        else:
            deri_1 = deri
        conv = conv + deri_1
        # making conv as the size of the matrix
        
        # matrix that contains Taylor Series
        S = np.zeros([conv, conv], dtype = float)
        # center column: 1 0 0 0 ...
        center_column = np.zeros([conv], dtype = float)
        center_column[0] = 1 
        S[ : , int((conv-1)/2)] = center_column
        # n distance from center to right : (n*h)**k/k! k: row count from 0
        for n in range(1, int((conv-1)/2)+1):
            for k in range(conv):
                S[k ,int((conv-1)/2)+n] = (n*h)**k/math.factorial(k)
        # n distance from center to left  : (-1*n*h)**k/k! k: row count from 0
        for n in range(1, int((conv-1)/2)+1):
            for k in range(conv):
                S[k ,int(conv/2)-n] = (-1*n*h)**k/math.factorial(k)
        b = np.zeros([conv], dtype = float).T
        b[deri] = 1
        # need to set stencil from this matrix
        stencil = np.linalg.inv(S) @ b

        stencil_2 = stencil[:int((conv-1)/2)]

        stencil_1 = stencil[-int((conv-1)/2):]

        
        stencil = np.concatenate((stencil_1, stencil, stencil_2), axis = None)
        
        # for 1st deri, 2nd order accuracy, stencil = np.array([-1/(2*h), 0, 1/(2*h)])
        
        # build sparse matrix from stencil + add some more terms
        offset = np.array(range(-int((conv-1)/2), int((conv-1)/2)+1))
        offset_2 = np.array(range(-self.grid.N+1, -self.grid.N+int((conv-1)/2)+1))
        offset_3 = np.array(range(self.grid.N-int((conv-1)/2), self.grid.N))
        offset = np.concatenate((offset_2, offset, offset_3), axis = None)
        
        D = sparse.diags(stencil, offsets=offset, shape = [self.grid.N, self.grid.N])

 
        # why don't we add this part as offset, instead of typing them all?
        
        return D

    def __matmul__(self, other):
        return self.matrix() @ other


class DifferenceNonUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.grid = grid
        pass
    
    def matrix(self):
        
        deri = self.derivative_order
        # where 1 located at b vector
        conv = self.convergence_order
        if deri % 2 ==0:
            deri_1 = deri -1
        else:
            deri_1 = deri
        conv = conv + deri_1
        # making conv as the size of the matrix
        
        
        h = self.grid.dx
        
        #extending h because it is periodic function
        h_1 = h[-conv:]
        h_2 = h[:conv]
        
        h = np.concatenate((h_1, h, h_2))
        
        # making difference matrix row by row
        D = np.zeros([self.grid.N, self.grid.N], dtype = float)
        
        for i in range(self.grid.N):
            # matrix that contains Taylor Series
            S = np.zeros([conv, conv], dtype = float)
            # center column: 1 0 0 0 ...
            center_column = np.zeros([conv], dtype = float)
            center_column[0] = 1 
            S[ : , int((conv-1)/2)] = center_column
        
            # n distance from center to right : (n*h)**k/k! k: row count from 0
            for n in range(1, int((conv-1)/2)+1):
                for k in range(conv):
                    S[k ,int((conv-1)/2)+n] = (h[i+conv:i+n+conv].sum())**k/math.factorial(k)
            # n distance from center to left  : (-1*n*h)**k/k! k: row count from 0
            for n in range(1, int((conv-1)/2)+1):
                for k in range(conv):
                    S[k ,int(conv/2)-n] = (-1*h[i-n+conv:i+conv].sum())**k/math.factorial(k)
            b = np.zeros([conv], dtype = float).T
            b[deri] = 1
            # getting a row vector
            stencil = np.linalg.inv(S) @ b
            #caution: it could go around
            for s in range(len(stencil)):
                m = i + s - int((conv-1)/2)
                if m >= self.grid.N:
                    m = m - self.grid.N
                D[i, m] = stencil[s]
                
        return D

    def __matmul__(self, other):
        return self.matrix() @ other

