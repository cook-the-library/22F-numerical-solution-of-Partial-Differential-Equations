from scipy import sparse
from timesteppers import StateVector, CrankNicolson, RK22
import finite

class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):
        pass

    def step(self, dt):
        pass


class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        pass

    def step(self, dt):
        pass


class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, p0):
        
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        # is rho0 a vector or a scalar?
        if type(rho0) != int:
            D = sparse.diags(rho0)
            M00 = D.dot(I)
        else:
            M00 = I * rho0
        
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        # is p0 a vector or a scalar?
        L00 = Z
        L01 = d.matrix
        if type(p0) != int:
            K = sparse.diags(p0)
            L10 = K.dot(d.matrix())    
        else:
            L10 = d.matrix * p0
            
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data


class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        self.c = c
        self.X = StateVector([c])
        
        N = len(c)
        self.M = sparse.eye(N, N)
        self.L = -D*d2.matrix
        
        f = lambda X: X.data*(c_target - X.data)
        
        self.F = f


