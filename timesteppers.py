import numpy as np
import scipy.sparse as sparse
import sympy
import math

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
        return coefficient
        
