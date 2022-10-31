import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque
from _array import axslice, apply_matrix

class StateVector:

    def __init__(self, variables, axis=0):
        self.axis = axis
        var0 = variables[0]
        shape = list(var0.shape)
        self.N = shape[axis]
        shape[axis] *= len(variables)
        self.shape = tuple(shape)
        self.data = np.zeros(shape)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[axslice(self.axis, i*self.N, (i+1)*self.N)], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[axslice(self.axis, i*self.N, (i+1)*self.N)])


class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1
    
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.F = eq_set.F


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.X.data + dt*self.F(self.X)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        N = len(X.data)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.X.data + dt*self.F(self.X)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.X_old = np.copy(self.X.data)
            return self.X.data + dt*self.F(self.X)
        else:
            X_temp = self.X_old + 2*dt*self.F(self.X)
            self.X_old = np.copy(self.X)
            return X_temp


class LaxWendroff(ExplicitTimestepper):

    def __init__(self, X, F1, F2):
        self.t = 0
        self.iter = 0
        self.X = X
        self.F1 = F1
        self.F2 = F2

    def _step(self, dt):
        return self.X.data + dt*self.F1(self.X) + dt**2/2*self.F2(self.X)


class Multistage(ExplicitTimestepper):

    def __init__(self, eq_set, stages, a, b):
        super().__init__(eq_set)
        self.stages = stages
        self.a = a
        self.b = b

        self.X_list = []
        self.K_list = []
        for i in range(self.stages):
            self.X_list.append(StateVector([np.copy(var) for var in self.X.variables]))
            self.K_list.append(np.copy(self.X.data))

    def _step(self, dt):
        X = self.X
        X_list = self.X_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(X_list[0].data, X.data)
        for i in range(1, stages):
            K_list[i-1] = self.F(X_list[i-1])

            np.copyto(X_list[i].data, X.data)
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                X_list[i].data += self.a[i, j]*dt*K_list[j]

        K_list[-1] = self.F(X_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            X.data += self.b[i]*dt*K_list[i]

        return X.data


def RK22(eq_set):
    a = np.array([[  0,   0],
                  [1/2,   0]])
    b = np.array([0, 1])
    return Multistage(eq_set, 2, a, b)


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, eq_set, steps, dt):
        super().__init__(eq_set)
        self.steps = steps
        self.dt = dt
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(np.copy(X.data))

    def _step(self, dt):
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.F(self.X)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.X.data += self.dt*coeff*self.f_list[i].data
        return self.X.data

    def _coeffs(self, num):

        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a


class ImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, axis):
        super().__init__()
        self.axis = axis
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L

    def _LUsolve(self, data):
        if self.axis == 0:
            return self.LU.solve(data)
        elif self.axis == len(data.shape)-1:
            return self.LU.solve(data.T).T
        else:
            raise ValueError("Can only do implicit timestepping on first or last axis")


class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self._LUsolve(self.X.data)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt/2*self.L
            self.RHS = self.M - dt/2*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))


class BackwardDifferentiationFormula(Timestepper):

    def __init__(self, u, L_op, steps):
        super().__init__()
        self.X = StateVector([u])
        self.func = L_op
        N = len(u)
        self.I = sparse.eye(N, N)
        self.steps = steps
        self.A = np.empty((0,u.size))
        
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
        u_old = self.X.data
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
        LHS = self.func.matrix - self.I * coefficient[0]
        self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
                
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
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
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

