from scipy.linalg import *
from numpy import *
from queue import *
from statistics import *
from guppy import hpy
import time 
import os
import sys
from scipy.optimize import nnls,lsq_linear

class NNLSSPAR:
    def __init__(self,A,b,s):
        """ 
        initilization of parameters 
        -------------------------------------------------------------------------------------
        A       = m x n matrix consisting of m observations of n independent variables
        b       = m x 1 column vector consisting of m observed dependent variable
        s       = sparsity level
        
        Solver parameters
        -------------------------------------------------------------------------------------
        P       = indices of possible choices for independent variable
        C       = Chosen set of independent variables
        E       = indices of unconstrained least squares varaibles
        
        This Python cde implements the algorithms described in the paper "PROVABLY OPTIMAL SPARSE SOLUTIONS TO 
        OVERDETERMINED LINEAR SYSTEMS WITH NON-NEGATIVITY CONSTRAINTS IN A LEAST-SQUARES SENSE BY IMPLICIT 
        ENUMERATION" by Fatih S. AKTAŞ, Ömer EKMEKÇİOĞLU, Mustafa Ç. PINAR.
        
        If you get a pythonic error, it will probably because of how unfriendly is python with usage of 
        vectors, like lists cannot be sliced with indeces but arrays can be done, hstack converts everything
        to floating points, registering a vector and randomly trying to make it 2 dimensional takes too much
        unnecessary work, some functions do inverse of hstack, randomly converting numbers to integers etc.
        
        Please contact selim.aktas@ug.bilkent.edu.tr for any bugs, errors and recommendations.
        """
        
        for i in range(len(A)):
            for j in range(len(A[0])):
                if math.isnan(A[i,j]) or abs(A[i,j]) == Inf:
                    print("Matrix A has NAN or Inf values, it will give linear algebra errors")
                    break
        for i in range(len(A)):
            for j in range(len(A[0])):
                if type(A[i,j]) != float64:
                    print("Matrix A should be registered as float64, otherwise computations will be wrong\
for example; Variance will be negative")
            
        if shape(shape(b)) == (1,):
            print("you did not register the vector b as a vector of size m x 1, you are making a pythonic\
                  error, please reshape the vector b")
        elif shape(A)[0] != shape(b)[0]:
            print("Dimensions of A and b must match, A is",shape(A)[0],"x",shape(A)[1],"matrix", \
                  "b is",shape(b)[0],"x 1 vector, make sure" ,shape(A)[0],"=",shape(b)[0])
        elif shape(A)[0] <= shape(A)[1]:
            print("The linear system is supposed to be overdetermined, given matrix A of size m x n,",\
                  "the condition m > n should be satisfied") 
        elif shape(b)[1] != 1:
            print("input argument b should be a vector of size m x 1, where m is the row size of the \
                  A")
        elif type(A) != ndarray:
            print("A should be a numpy ndarray")
        elif type(s) != int:
            print("s should be an integer")
        else:
            self.A = A
            self.b = b
            self.bv = b[:,0]
            self.s = s
            """ registering the matrix A independent variable values, vector b dependent variable values 
            and s sparsity level """
            self.m = shape(A)[0]
            self.n = shape(A)[1]
            """ saving the shape of matrix A explicitly not to repeat use of shape() function """
            self.means = abs(A.mean(axis = 0))
            """ storing the mean values of each independent variable which will be used for 
            selection criteria in solver algortihms"""
            self.sterror = std(A,axis = 0)
            """ storing the standard deviation and variance of each independent variable which will
            be used for selection criteria in solver algortihms"""
            self.rem_qsize = []
            """ initializing remaining queue size after the algorithm finishes """
            self.nodes = []
            """ initializing number of nodes searched after the algorithm finishes """
            self.out = 1
            """ initializing the output choice of algorithm """
            self.SST = variance(b[:,0])*(self.m-1)
            """ total variabiliy in responde variable """
            self.original_stdout = sys.stdout
            """  saving original stdout, after manipulating it for changing values of out, we should be
            able to recover it """
            q,r = linalg.qr(A)
            self.q = q
            self.rtilde = r
            self.qb = self.q.T.dot(self.b)
            self.permanent_residual = norm(b) ** 2 - norm(self.qb) ** 2
            """ initial QR decomposition to shrink the system """
            self.tablelookup_nnls = {}
            self.tablelookup_nnls_elsq = {}
            """ lookup tables for solving all subset problem """
            self.many = 1
            """ initializing the number of solutions to be found"""
            self.residual_squared = []
            self.indices = []
            self.coefficients = []
            """ initializing the arrays of solution parameters """
            self.memory = 0
            """ initializing memory usage of the algorithm """
            self.cpu = 0
            """ initializing cpu usage of the algorithm """
            self.real = 0
            """ initializing wall time of the algoritmh """

    def nonnegative_IHT(self,ind,x = False):
        """
        This function implements the Iteratively Hard Thresholding Algortihm with non-negativity constraint with
        initial support given (ind) and initial guess (x).
        """
        l = len(ind)
        A = self.rtilde[:,ind]
        u,d,v = svd(A)
        step = 1/d[0] ** 2
        if x is False:
            x1 = zeros(l)
            x2 = copy(x1)
        else:
            x1 = copy(x)
            x2 = copy(x1)
        diff = 1
        t = 0
        while diff >= 1e-6 and t <= 400:
            print("dif",diff,"time",t)
            t += 1
            grad = self.rtilde[:,ind].T.dot(self.rtilde[:,ind].dot(x1) - self.qb[:,0])
            x1 = x1 - step * grad
            
            x1[where(x1 < 0)] = 0
            si = argsort(x1)
            x1[si[:-self.s]] = 0
            diff = norm(x2-x1,2)
            x2 = x1

        return x1

    def qr_nnls(self,ind):
        """ non-negative least squares solution to support (ind) after QR shrinking step """
        
        t = max(ind)+1
        """ since R is upper triangular, rows beyond t are 0 """
        sol = nnls(self.rtilde[:t,ind],self.qb[:t,0])
        res = sol[1] ** 2 + norm(self.qb[t:]) ** 2
        return [sol[0],res]
    
    def qr_nnlsl(self,ind):
        """ non-negative least squares solution to support (ind) after QR shrinking step for 
        all subsets problem, using table lookups"""
        
        check = str(ind)
        if check in self.tablelookup_nnls:
            return self.tablelookup_nnls[check]
        """ table look up step """
        t = max(ind)+1
        """ since R is upper triangular, rows beyond t are 0 """
        sol = nnls(self.rtilde[:t,ind],self.qb[:t,0])
        res = sol[1] ** 2 + norm(self.qb[t:]) ** 2
        """ using QR to solve a least squares problem """
        self.tablelookup_nnls[check] = [sol[0],res]
        """ registering the table look up """
        return [sol[0],res]
    
    def qr_nnls_elsq(self,ind):
        """ non-negative least squares solution to support (ind) and unconstrained varaible self.excluded_indices
        after QR shrinking step """
        
        l = len(ind)
        t = max(max(ind),self.excluded_indices_max)+1
        """ since R is upper triangular, rows beyond t are 0 """
        ub = array([inf] * (l + self.excluded_indices_length))
        lb = array([0] * l + [-inf] * self.excluded_indices_length)
        sol = lsq_linear(self.rtilde[:t,ind+self.excluded_indices],self.qb[:t,0],bounds = [lb,ub])
        """ least squares with linear inequalities solver """
        return [sol.x,sol.cost]


    def qr_nnls_elsql(self,ind):
        """ non-negative least squares solution to support (ind) and unconstrained varaible self.excluded_indices
        after QR shrinking step for all subsets problem with table lookups """
        check = str(ind)
        if check in self.tablelookup_nnls_elsq:
            return self.tablelookup_nnls_elsq[check]
        """ table look up step """
        l = len(ind)
        t = max(max(ind),self.excluded_indices_max)+1
        """ since R is upper triangular, rows beyond t are 0 """
        ub = array([inf] * (l + self.excluded_indices_length))
        lb = array([0] * l + [-inf] * self.excluded_indices_length)
        sol = lsq_linear(self.rtilde[:t,ind+self.excluded_indices],self.qb[:t,0],bounds = [lb,ub])
        """ least squares with linear inequalities solver """
        self.tablelookup_nnls_elsq[check] = [sol.x,sol.cost]
        
        return [sol.x,sol.cost]    
                    
    def solve_nnls(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let beta(i) be NNLS coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let std(i) be standard deviation of x(i), i'th independent variable
        if (|xbar(j)| + std(j))*beta(j) is the highest then j'th variable has the highest impact
        
        xbar*beta is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        std*beta is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high beta(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*beta(j) would not catch this variable
        
        This search rule basically finds the variable with the highest magnite in the scaled version of the non negative 
        least squares if the variable is not scaled prior. This is useful because it uses much less flops thansolving a 
        parallel non negative least squares with scaled variables.

        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        node = 0
        """ number of nodes searched so far  """
        f =  self.qr_nnls(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        count_best = 0
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            node += 1
            """ updating number of nodes searched """
            # print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
            if len_c < self.s:
                if  count_nonzero(coef) <= self.s:
                    self.residual_squared.append(low)
                    index = list(where(coef > 0)[0])
                    all_variables = P+C
                    real_indices = [all_variables[i] for i in index]
                    self.indices.append(real_indices)
                    self.coefficients.append(coef[index])
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        self.nodes.append(node)
                        break
                    """ termination condition of the algorithm """
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = (sdx+xbar)*coef[:len_p]
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef[l_index_bb:-1],coef[-1] = coef[l_index_bb+1:],coef[l_index_bb]
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indices"""
                    lower2 = self.qr_nnls(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indices ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_nnls(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indices.append(C)
                self.coefficients.append(coef)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    self.nodes.append(node)
                    break
                """ termination condition of the algorithm """

    def solve_mk0(self,P ,C = []):
        """ 
        
        This is a function to solve the all subsets problem with exploiting previously solved problems with table look ups.
        
        """
        
        P_0 = P[:]
        C_0 = C[:]        
        
        f = self.qr_nnls(C+P)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)
        """ initialization of the first problem """
        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            print("Started solving for sparsity level ",self.s)
            q = PriorityQueue()
            q.put([f[1],[P_0,C_0,f[0][lenc:],lenc,lenp]])
            """ starting from scratch with tablelookups """
            node = 0
            """ number of nodes searched so far  """
            count_best = 0
            """ number of solutions found so far """
            while q.qsize() >= 1:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = q.get()
                """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
                node += 1
                """ updating number of nodes searched """
                print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
                if len_c < self.s:
                    if  count_nonzero(coef) <= self.s:
                        self.residual_squared.append(low)
                        index = list(where(coef > 0)[0])
                        all_variables = P+C
                        real_indices = [all_variables[i] for i in index]
                        self.indices.append(real_indices)
                        self.coefficients.append(coef[index])
                        """ registering the solution"""
                        count_best += 1
                        if count_best == self.many:
                            self.rem_qsize.append(q.qsize())
                            self.nodes.append(node)
                            break
                        """ termination condition of the algorithm """
                    else:
                        xbar = self.means[P]
                        """ xbar is a vector length len(p), it retrieves the mean for each variable """
                        sdx = self.sterror[P]
                        """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                        bb_dec = (sdx+xbar)*coef[:len_p]
                        """ bb_dec is the decision vector, logic is explained above"""
                        l_index_bb = argmax(bb_dec)
                        """ find index of the largest value in decision vector """
                        r_index_bb = P[l_index_bb]
                        """ find index of the variable by index of the largest value in decision vector 
                        this is the chosen variable for this node """ 
                        C1 = C + [r_index_bb]
                        coef1 = copy(coef)
                        coef1[l_index_bb:-1],coef1[-1] = coef1[l_index_bb+1:],coef1[l_index_bb]
                        """ add the new chosen variable to the solution 
                        We also use old C, where chosen variable is not added""" 
                        P1 = P[:]
                        del P1[l_index_bb]
                        """ erasing the chosen variable from the possible variables' list 
                        reminder: this stores the variables by their indices"""
                        lower2 = self.qr_nnlsl(P1+C)
                        """ calculate lower bound of the second solution where C is the old chosen variable 
                        list and p1 is the possible variable (indices ) list where the chosen variable for 
                        this node is erased """
                        len_p -= 1
                        len_c1 = len_c +1 
                        if len_c1 == self.s:
                            """ fix here """
                            sol = self.qr_nnls(C1)
                            q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                            q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                        else:
                            """ if the length of the chosen variable list is not equal to sparsity level, 
                            then it is lower than sparsity level. We create two new nodes where first node
                            is the node where chosen variable for this node is in the solution and the second 
                            where the chosen variable for this node is erased from the problem """
                            q.put([low ,[P1,C1,coef1,len_c1,len_p]])
                            q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
                else:
                    self.residual_squared.append(low)
                    self.indices.append(C)
                    self.coefficients.append(coef)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        self.nodes.append(node)
                        break
                    """ termination condition of the algorithm """
                    
    def solve_nnls_elsq(self,P ,C = [],E = []):
        """ 
        
        This Algorithm solves the non-negative sparse least squares with extended least squares problem, 
        formulated as follows;
                
        min ||Ax + By - b||_2
        
            ||x||_0 <= s
            x >= 0
            y free

        """
        self.excluded_indices = E
        self.excluded_indices_max = max(E)
        self.excluded_indices_length = len(E)
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        node = 0
        """ number of nodes searched so far  """
        f =  self.qr_nnls_elsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        count_best = 0
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            node += 1
            """ updating number of nodes searched """
            # print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
            if len_c < self.s:
                if  count_nonzero(coef[:(len_c+len_p)]) <= self.s:
                    self.residual_squared.append(low)
                    index = list(where(coef[:(len_c+len_p)] > 0)[0])
                    all_variables = P+C
                    real_indices = [all_variables[i] for i in index]
                    self.indices.append(real_indices)
                    self.coefficients.append(coef[index+list(range(-1,-self.excluded_indices_length-1,-1))])
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        self.nodes.append(node)
                        break
                    """ termination condition of the algorithm """
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = (sdx+xbar)*coef[:len_p]
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef[l_index_bb:-1],coef[-1] = coef[l_index_bb+1:],coef[l_index_bb]
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indices"""
                    lower2 = self.qr_nnls_elsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indices ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_nnls_elsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indices.append(C)
                self.coefficients.append(coef)
                """ registerin the solution"""
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    self.nodes.append(node)
                    break
                """ termination condition of the algorithm """

    def solve_mk0_elsq(self,P ,C = [],E = []):
        """ 
        
        This is a function to solve the all subsets problem for extended least squares problem
        with exploiting previously solved problems with table look ups.
        
        """
        self.excluded_indices = E
        self.excluded_indices_max = max(E)
        self.excluded_indices_length = len(E)
        
        P_0 = P[:]
        C_0 = C[:]   
        
        f = self.qr_nnls(C+P)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)
        """ initialization of the first problem """
        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            print("Started solving for sparsity level ",self.s)
            q = PriorityQueue()
            q.put([f[1],[P_0,C_0,f[0][lenc:],lenc,lenp]])
            """ starting from scratch with tablelookups """
            node = 0
            """ number of nodes searched so far  """
            count_best = 0
            """ number of solutions found so far """
            while q.qsize() >= 1:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = q.get()
                """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
                node += 1
                """ updating number of nodes searched """
                print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
                if len_c < self.s:
                    if  count_nonzero(coef[:(len_c+len_p)]) <= self.s:
                        self.residual_squared.append(low)
                        index = list(where(coef[:(len_c+len_p)] > 0)[0])
                        all_variables = P+C
                        real_indices = [all_variables[i] for i in index]
                        self.indices.append(real_indices)
                        self.coefficients.append(coef[index+list(range(-1,-self.excluded_indices_length-1,-1))])
                        """ registering the solution"""
                        count_best += 1
                        if count_best == self.many:
                            self.rem_qsize.append(q.qsize())
                            self.nodes.append(node)
                            break
                        """ termination condition of the algorithm """
                    else:
                        xbar = self.means[P]
                        """ xbar is a vector length len(p), it retrieves the mean for each variable """
                        sdx = self.sterror[P]
                        """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                        bb_dec = (sdx+xbar)*coef[:len_p]
                        """ bb_dec is the decision vector, logic is explained above"""
                        l_index_bb = argmax(bb_dec)
                        """ find index of the largest value in decision vector """
                        r_index_bb = P[l_index_bb]
                        """ find index of the variable by index of the largest value in decision vector 
                        this is the chosen variable for this node """ 
                        C1 = C + [r_index_bb]
                        coef1 = copy(coef)
                        coef1[l_index_bb:-1],coef1[-1] = coef1[l_index_bb+1:],coef1[l_index_bb]
                        """ add the new chosen variable to the solution 
                        We also use old C, where chosen variable is not added""" 
                        P1 = P[:]
                        del P1[l_index_bb]
                        """ erasing the chosen variable from the possible variables' list 
                        reminder: this stores the variables by their indices"""
                        lower2 = self.qr_nnls_elsql(P1+C)
                        """ calculate lower bound of the second solution where C is the old chosen variable 
                        list and p1 is the possible variable (indices ) list where the chosen variable for 
                        this node is erased """
                        len_p -= 1
                        len_c1 = len_c +1 
                        if len_c1 == self.s:
                            """ fix here """
                            sol = self.qr_nnls_elsq(C1)
                            q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                            q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                        else:
                            """ if the length of the chosen variable list is not equal to sparsity level, 
                            then it is lower than sparsity level. We create two new nodes where first node
                            is the node where chosen variable for this node is in the solution and the second 
                            where the chosen variable for this node is erased from the problem """
                            q.put([low ,[P1,C1,coef1,len_c1,len_p]])
                            q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
                else:
                    self.residual_squared.append(low)
                    self.indices.append(C)
                    self.coefficients.append(coef)
                    """ registering the solution"""
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        self.nodes.append(node)
                        break
                    """ termination condition of the algorithm """
                    
    
    def solve(self,C = []):
        """ 
        This is wrapper for sparse non-negative least squares routines 
        """
        
        if self.out not in [0,1,2]:
            print("OUT parameter should be a integer >=0  and <= 2")
            return None
        elif self.n < self.s:
            print("sparsity level is higher than number of variables")
            return None
        elif self.many > math.factorial(self.n)/(math.factorial(self.s)*math.factorial(self.n-self.s)):
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        
        if not type(C) == list and C != []:
            print("C should be a list, C is taken empty list now ")
            C = []
        elif C != [] and (max(C) >= self.n or min(C) < 0):
            print("Values of C should be valid, in the range 0 <= n-1, C is taken empty list")
            C = []
        elif len(set(C)) != len(C):
            print("Values of C should be unique, C is taken empty list")
            C = []
        elif len(C) > self.s:
            print(" Length of C cannot be greater than spartsity level s,C is taken empty list")
            C = []
            
            
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
        
        P = list(range(self.n))
        if C != []:
            for i in range(len(C)):
                P.remove(C[i])
        """ Preparing the P and C that will be passed to the function """
        
        """ Another if list to find and call the right function """
        
        t3 = time.time()
        self.solve_nnls(P,C)
        t4 = time.time()
        finish = time.process_time()
        self.cpu = finish - t0
        self.real = t4-t3
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout
        print("CPU time of the algorithm",self.cpu,"seconds")
        print("Wall time of the algorithm",self.real,"seconds")
        m = mem.heap()
        print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout


    def solve_elsq(self,E = [],C = []):
        """ 
        This is wrapper for sparse non-negative least squares with extended least squares routines 
        """
        
        if self.out not in [0,1,2]:
            print("OUT parameter should be a integer >=0  and <= 2")
            return None
        elif self.n < self.s:
            print("sparsity level is higher than number of variables")
            return None
        elif self.many > math.factorial(self.n)/(math.factorial(self.s)*math.factorial(self.n-self.s)):
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        
        if not type(C) == list and C != []:
            print("C should be a list, C is taken empty list now ")
            C = []
        elif C != [] and (max(C) >= self.n or min(C) < 0):
            print("Values of C should be valid, in the range 0 <= n-1, C is taken empty list")
            C = []
        elif len(set(C)) != len(C):
            print("Values of C should be unique, C is taken empty list")
            C = []
        elif len(C) > self.s:
            print(" Length of C cannot be greater than spartsity level s,C is taken empty list")
            C = []
        elif E == []:
            print("extended least squares with empty extension set, please use solve() method instead")
        elif set(C).intersection(E) != set():
            print("There is a common index in C and E ")
            
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
        
        P = list(range(self.n))
        if C != []:
            for i in range(len(C)):
                P.remove(C[i])
        for i in range(len(E)):
            P.remove(E[i])
        """ Preparing the P and C that will be passed to the function """
        
        """ Another if list to find and call the right function """
        
        t3 = time.time()
        self.solve_nnls_elsq(P,C,E)
        t4 = time.time()
        finish = time.process_time()
        self.cpu = finish - t0
        self.real = t4-t3
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout
        print("CPU time of the algorithm",self.cpu,"seconds")
        print("Wall time of the algorithm",self.real,"seconds")
        m = mem.heap()
        print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout
        
    def solve_allsubsets(self):        
        """ 
        This is wrapper for sparse non-negative least squares routines for all subsets problem
        """
        if self.many > self.n:
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        elif self.n < self.s:
            print("sparsity level is higher than number of variables")
            return None

        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
            
        P = list(range(self.n))
        C = []
        """ Preparing the P and C that will be passed to the function """
        
        
        t3 = time.time()
        self.solve_mk0(P,C)
        """ mk0 also works, but this is in general faster """
        t4 = time.time()
        finish = time.process_time()
        self.cpu = finish - t0
        self.real = t4-t3
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout   
        print("CPU time of the algorithm",self.cpu,"seconds")
        print("Wall time of the algorithm",self.real,"seconds")
        m = mem.heap()
        print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout

    def solve_allsubsets_elsq(self,E = []):        
        """ 
        This is wrapper for sparse non-negative least squares with extended least squares routines for all subsets 
        problem
        """
        if self.many > self.n:
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        elif self.n < self.s:
            print("sparsity level is higher than number of variables")
            return None
        elif E == []:
            print("extended least squares with empty extension set, please use solve_allsubsets() method instead")
            
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
            
        P = list(range(self.n))
        C = []
        """ Preparing the P and C that will be passed to the function """
        
        
        t3 = time.time()
        self.solve_mk0_elsq(P,C,E)
        """ mk0 also works, but this is in general faster """
        t4 = time.time()
        finish = time.process_time()
        duration = finish-t0
        self.cpu = duration
        self.real = t4-t3
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout   
        print("CPU time of the algorithm",duration,"seconds")
        m = mem.heap()
        print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout
        
            