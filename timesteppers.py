import numpy as np
import scipy.sparse as sparse
import sympy
import math
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque

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
        
class StateVector:
    
    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N*len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()
    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i+1)*self.N], var)
    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])   

class IMEXTimestepper:
    def __init__(self, eq_set):
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.dt = None
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)
    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1
        
class Euler(IMEXTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)
    
class CNAB(IMEXTimestepper):
    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt
            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data +3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)
    
class BDFExtrapolate(IMEXTimestepper):
    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.steps_old_ex = 0
        self.steps_old_im = 0
        
        self.X_old = np.empty((0,self.X.data.size))
        self.FX_old =np.empty((0,self.X.data.size))
        
    def _step(self, dt):
        # define steps for inital stages
        if self.iter+1 < self.steps:
            steps = self.iter+1
        else:
            steps = self.steps
           
        # X_old stores previous values of X as row vector
        # new X^n-1 at top
        X_old = np.vstack([self.X.data, self.X_old]) 
        # delete old X if longer than self.steps
        if X_old.shape[0] > self.steps:
            X_old = np.delete(X_old,-1,0)
        self.X_old = X_old
        
        
        # FX_old stores previous values of FX as row vector
        # new FX^n-1 at top
        FX_old = np.vstack([self.F(self.X).data, self.FX_old]) 
        # delete old F if longer than self.steps
        if FX_old.shape[0] > self.steps:
            FX_old = np.delete(FX_old,-1,0)
        self.FX_old = FX_old 
        
        # calculate matrix
        if self.steps_old_im != steps:
            LHS = self.M*self._coeff_im(steps, dt)[0][0] + self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        RHS = -self.M*(self._coeff_im(steps, dt)[1:] * self.X_old).sum(axis = 0) + (self._coeff_ex(steps, dt) * self.FX_old).sum(axis = 0)
        return self.LU.solve(RHS)
    
    def _coeff_ex(self, steps, dt):
        if self.steps_old_ex == steps:
            return self.coeff_ex
        else:
            self.steps_old_ex = steps
            i = (1 + np.arange(steps))[None, :]
            j = (1 + np.arange(steps))[:, None]
            S = (-i*dt)**(j-1)/factorial(j-1)
            b = 0*j
            b[0] = 1
            self.coeff_ex = np.linalg.solve(S, b)
            # self.coeff_ex is (steps,1) vector, first element to X^n-1
            return self.coeff_ex
        
    def _coeff_im(self, steps, dt):
        if self.steps_old_im == steps:
            return self.coeff_im
        else:
            self.steps_old_im = steps
            i = (np.arange(steps+1))[None, :]
            j = (1 + np.arange(steps+1))[:, None]
            S = (-i*dt)**(j-1)/factorial(j-1)
            b = 0*j
            b[1] = 1
            self.coeff_im = np.linalg.solve(S, b)
            self.coeff_im = self.coeff_im
            # self.coeff_im is (steps+1,1) vector, first element to X^n
            return self.coeff_im
        
        
        