import numpy as np
import scipy.sparse as sparse
import sympy
import math
import scipy.sparse.linalg as spla

class Timestepper:

    def __init__(self, u, f):
        self.t = 0
        self.iter = 0
        self.u = u
        self.func = f
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1
        
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ForwardEuler(Timestepper):

    def _step(self, dt):
        return self.u + dt*self.func(self.u)


class LaxFriedrichs(Timestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.func(self.u)


class Leapfrog(Timestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.func(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.func(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, func1, func2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = func1
        self.f2 = func2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(Timestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b
        

    def _step(self, dt):
        k_set = np.zeros(shape = (self.stages, len(self.u)), dtype = float)
        for i in range(len(k_set)):
            k_set[i] = self.func(self.u + dt*self.a[i,:]@k_set)
            a = self.func(self.u + dt*self.a[i,:]@k_set)
        return self.u + dt * self.b @ k_set

class AdamsBashforth(Timestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt # constant
        self.A = np.empty((0,self.u.size))
        self.coefficient = np.array([])
        self.stage_old = 0
        

    def _step(self, dt):
        
        if self.iter+1 < self.steps:
            stage = self.iter+1
        else:
            stage = self.steps

        u_old = self.u
        # A stores previous values of f(u)
        
        A = np.vstack([self.func(u_old), self.A]) 
        
        if A.shape[0] > self.steps:
            A = np.delete(A,-1,0)
        
        self.A = A
        
        return self.u + dt*self._coefficient(stage) @ A
    
     
    def _coefficient(self, stage):
        if stage == self.stage_old:
            return self.coefficient
        else:
            self.stage_old = stage
            self.stage = stage
            coefficient = np.zeros(stage,dtype=float)
            x = sympy.symbols('x')
            f = "1"
            for i in range(stage):
                f +="*(x+{})".format(i)
            f_old = f
            for i in range(stage):
                f = f_old
                denominator = math.factorial(i)*math.factorial(stage-1-i)
                f +="/(x+{})".format(i)

                f = sympy.parsing.sympy_parser.parse_expr(f)
                a = sympy.integrate(f,(x,0,1))/denominator
                coefficient[i] = (-1)**i*sympy.Float(a)
                self.coefficient = coefficient
            return coefficient
        
class BackwardEuler(Timestepper):
    def __init__(self, u, L):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)
    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.func.matrix
            print(self.func.matrix.dtype)
            print(self.func)
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.u)
    
class CrankNicolson(Timestepper):
    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)
    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.func.matrix
            self.RHS = self.I + dt/2*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.RHS @ self.u)
    
class BackwardDifferentiationFormula(Timestepper):
    def __init__(self, u, L_op, steps):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)
        self.steps = steps
        self.A = np.empty((0,self.u.size))
        
        # stores previous time steps
        self.dt_series = np.array([])
        
        # list of numpy for nonuniform time steps
        self.dt_pack = []
        
        # dictionary not possible, so store in same order as list
        self.coefficient_pack = []
        
    def _step(self, dt):
        # define steps for inital stages
        if self.iter+1 < self.steps:
            steps = self.iter+1
        else:
            steps = self.steps
            
        # new dt at dt_series[0], oldest at [-1]
        self.dt_series = np.append(dt, self.dt_series)
        # deleting oldest dt if longer than stage
        if len(self.dt_series) > self.steps:   
            self.dt_series = self.dt_series[:-1]    
        
        # check whether self.dt_series is new
        boolean_1 = np.array([], dtype=bool)
        
        for i in self.dt_pack:
            boolean_1 = np.append(boolean_1, np.array_equal(i, self.dt_series))
        boolean_2 = boolean_1.any()
        # add into dt_pack if new and calculate coefficient
        if not boolean_2:
            self.dt_pack.append(self.dt_series)
            coefficient = self._coefficient(steps, self.dt_series)
            self.coefficient_pack.append(coefficient)
        else:   
        # ["foo", "bar", "baz"].index("bar")
            index_ = np.where(boolean_1)[0]
            coefficient = self.coefficient_pack[index_[0]]
        # have coefficient by now, need previous u
        # A stores previous values of u
        # new u at top
        u_old = self.u
        A = np.vstack([u_old, self.A]) 
        # delete old u if longer than self.steps
        if A.shape[0] > self.steps:
            A = np.delete(A,-1,0)
        self.A = A

        # store as matrix and call as vector
        # multiply coeff to u^n-1, ...
        RHS = (A * coefficient[1:, np.newaxis]).sum(axis = 0)
        
        # get u^n from inverse matrix
        # multiply coeff[0] to u^n
        # dt already divided at the coefficients
        LHS = self.func.matrix() - self.I * coefficient[0]
        self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        
        print("dt pack : ", self.dt_pack)
        print("coeff pack : ", self.coefficient_pack)
        
        return self.LU.solve(RHS)
        
    def _coefficient(self, steps, dt_series):
        # already checked whether dt_series are same & # of steps
        coefficient = np.zeros(steps+1, dtype=float)
        differentiation = np.zeros(steps+1, dtype = float)
        differentiation[1] = 1
        
        # matrix for differ. = matrix dot coeff.
        matrix = np.zeros([steps+1, steps+1], dtype = float)
        matrix[0,:] = 1
        
        for i in range(1, steps+1):
            for j in range(1, steps+1):
                matrix[i,j] = (-dt_series[:j].sum())**i/math.factorial(i)
        # get coefficient from inverse matrix
        coefficient = np.linalg.inv(matrix) @ differentiation 
        
        # a0 is u^n, so should consider at _step function
        return coefficient
        
        