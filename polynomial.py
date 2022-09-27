import numpy as np
import math

class Polynomial:
    def __init__(self, coefficient):
        self.coefficient = coefficient

    
    @staticmethod
    def from_string(string):
        # removing blank space
        string = string.replace(" ","")
        
        # adding 1* infront of x
        string = string.replace("*x","k")
        string = string.replace("x","1*x")
        string = string.replace("k","*x")
        # spliting into monomial terms
        string = string.replace("-", "+-")
        string_mono = string.split("+")
        
        # removing first one if it is blank
        if string_mono[0] == "":
            string_mono = string_mono[1:]
            
        # defining default coefficient & power
        idx = []
        coef = []
        coefficient=np
        
        # extracting coefficient & power
        for x in string_mono:
            x_power = x.split("^")
            
            # extracting constant & first power
            if x_power[0] == x:
                x1 = x.replace("x","")
                x1 = x1.replace("*","")
                
                # constant
                if x1 == x:
                    idx.append(0)
                    
                    # how am I going to handle rational coefficient?
                    if "/" in x:
                        x_part = x.split("/")
                        x_int = int(x_part[0]) // int(x_part[1])
                        coef.append(int(x_int))
                    else:
                        coef.append(int(x))
                    
                # first power
                else:
                    idx.append(1)
                    
                    # how am I going to handle rational coefficient?
                    if "/" in x:
                        x_part = x1.split("/")
                        x_int = int(x_part[0]) // int(x_part[1])
                        coef.append(int(x_int))
                    else:
                        coef.append(int(x1))
            
            # extracting power higher than two
            else:
                idx.append(int(x_power[1]))
                x_1 = x_power[0].replace("x","")
                x_1 = x_1.replace("*","")
                
                # how am I going to handle rational coefficient?
                if "/" in x:
                    x_part = x_1.split("/")
                    x_int = int(x_part[0]) // int(x_part[1])
                    coef.append(int(x_int))
                else:
                    coef.append(int(x_1))
                
        # implanting coefficients at right location
        coefficient = np.zeros(max(idx) + 1,dtype = int)  
        for (a, b) in zip(idx, coef):
            coefficient[a] = b  
        return Polynomial(coefficient)
        
    def __repr__(self):
        string = ""
        for idx, x in enumerate(self.coefficient):
            
            # skipping elements which is zero
            if x != 0:
                
                # adding constant to string
                if   idx == 0:
                    string += str(x)
                
                # adding first power to string
                elif idx == 1:
                    
                    # positive vs negative coefficient
                    if x > 0:
                        string += "+" + str(x) + "*x"
                    else:
                        string += str(x) + "*x"
                        
                # adding multipower to string
                else:
                    
                    # positive vs negative coefficient
                    if x> 0:
                        string += "+" + str(x) + "*x^" + str(idx)
                    else:
                        string += str(x) + "*x^" + str(idx)
                        
        return string
        
    def __add__(self, other):
        if len(self.coefficient) < len(other.coefficient):
            result = other.coefficient.copy()
            result[:len(self.coefficient)] += self.coefficient
        else:
            result = self.coefficient.copy()
            result[:len(other.coefficient)] += other.coefficient
        while result[-1] == 0:
            result = result[:-1]
            if result.size == 1:
                return 0
        return Polynomial(result)
    
    def __sub__(self, other):
        if len(self.coefficient) < len(other.coefficient):
            result = - other.coefficient.copy()
            result[:len(self.coefficient)] = self.coefficient + result[:len(self.coefficient)]
        else:
            result = self.coefficient.copy()
            result[:len(other.coefficient)] = result[:len(other.coefficient)] - other.coefficient
        while result[-1] == 0:
            result = result[:-1]
            if result.size == 1:
                return 0
        return Polynomial(result)
    
    def __mul__(self, other):
        first_length = self.coefficient.size
        secon_length = other.coefficient.size
        coef = np.zeros(first_length + secon_length - 1, dtype=int)
        idx = 0
        for x in self.coefficient:
            if x != 0:
                multip = x * other.coefficient
                idy = 0
                for y in range(secon_length):
                    coef[idy + idx] += multip[idy]
                    idy += 1
            idx += 1
        return Polynomial(coef)
    
    def __eq__(self, other):
        comp = self.coefficient == other.coefficient
        return comp.all()
    
    def __truediv__(self, other):
        numerator = self.coefficient
        denominator = other.coefficient
        return RationalPolynomial(numerator, denominator)

class RationalPolynomial():
    def __init__(self, numerator, denominator):
        self.numerator=numerator
        self.denominator=denominator
        
#    @staticmethod
#    def from_string(string):
    
#    def __repr__(self):
    
    def _reduce(self):
        gcd = math.gcd(self.numerator, self.denominator)
        self.numerator   = self.numerator   // gcd
        self.denominator = self.denominator // gcd
        if self.denominator < 0:
            self.numerator *= -1
            self.denominator *= -1
    
#    def __add__(self, other):

#    def __sub__(self, other):
    
#    def __mul__(self, other):
    
#    def __eq__(self, other):
    
#    def __div__(self, other):
        
    
    

 
    