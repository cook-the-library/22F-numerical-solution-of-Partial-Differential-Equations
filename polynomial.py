import numpy as np
import math
import sympy

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
            
        # remove maximum power's coefficient if it is zero
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
        
        # remove maximum power's coefficient if it is zero
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
        self._reduce()
        
    @staticmethod
    def from_string(string):
        # consider rational coefficient
        string = string.replace(" ","")
        string = string.split("/(")
        for idx, x in enumerate(string):
            string[idx] = x.replace("(","")
            string[idx] = string[idx].replace(")","")
        
        numerator = Polynomial.from_string(string[0])
        denominator = Polynomial.from_string(string[1])
        return RationalPolynomial(numerator, denominator)
    
    def _reduce(self): # finding greatest common factor of polynomials
        x = sympy.symbols("x")
        f = str(self.numerator)
        g = str(self.denominator)
        
        f = f.replace("^", "**")
        g = g.replace("^", "**")
        
        f = sympy.parsing.sympy_parser.parse_expr(f)
        g = sympy.parsing.sympy_parser.parse_expr(g)
        
        gcd = sympy.gcd(f, g)
        
        f = sympy.cancel(f/gcd)
        g = sympy.cancel(g/gcd)
        
        f = str(f).replace("**","^")
        g = str(g).replace("**","^")        
        
        self.numerator = Polynomial.from_string(f)
        self.denominator = Polynomial.from_string(g)
        
        if self.denominator.coefficient[-1] < 0:
            self.numerator *= Polynomial.from_string('-1')
            self.denominator *= Polynomial.from_string('-1')
        
        

    def __repr__(self):
        string = str(self.numerator) + " / " + str(self.denominator)
        return string
    
    def __add__(self, other):
        numerator = ( self.numerator * other.denominator
                   + other.numerator *  self.denominator)
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)

    def __sub__(self, other):
        numerator = ( self.numerator * other.denominator
                   - other.numerator *  self.denominator)
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)
    
    def __mul__(self, other):
        return RationalPolynomial(self.numerator*other.numerator,
                        self.denominator*other.denominator)
    
    def __truediv__(self, other):
        return RationalPolynomial(self.numerator*other.denominator,
                        self.denominator*other.numerator)
    
    def __eq__(self, other):
        if self.numerator == other.numerator:
            if self.denominator == other.denominator:
                return True
        return False
        
    
    

 
    