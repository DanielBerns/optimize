import numpy as np

class Goal:
    def __init__(self, r, J, m, n):
        self._iterations = 0
        self._func_eval = 0
        self._r = r
        self._J = J
        self._m = m
        self._n = n
        
    @property
    def iterations(self):
        return self._iterations
    
    @iterations.setter
    def iterations(self, value):
        self._iterations = value
        
    @property
    def r(self):
        return self._r
    
    @property
    def J(self):
        return self._J
        
    @property
    def shape(self):
        return (self._m, self._n)

    def f(x):
        """Construction of the objective function f(x)"""
        return 0.5 ∗ np.linalg.norm(self.r(x))∗∗2

    # Construction of the gradient
    def gradf(x):
        """Construction of the objective function f(x)"""
        return np.dot(np.transpose(self.J(x)), self.r(x))

    
def levenberg_marquardt (
    r, J, 
    x, 
    Delta = 100, 
    Delta_max = 10000,
    eta = 0.0001, 
    sigma = 0.1, 
    nmax = 500, 
    tol_abs = 10∗∗(−7), 
    tol_rel = 10∗∗(−7), 
    eps = 10∗∗(−3),
    Scaling = False
):
    goal = Goal(r, J, m, n)
    
    # first function evaluation at x
    fx = goal.f(x)
    Jx = goal.J(x)
    norm_Jx = np.linalg.norm(Jx)
    tolerance = min((tol_rel ∗ norm_Jx + tol_abs), eps)
    
    # for Scaling: Create two identity matrices
    D = np.eye(n)
    D_inv = np.eye(n)
    
    while norm_Jx > tolerance and goal.iterations < nmax:
        
        # Construct the Scaling matrix D (like in chapter 3.5.1)
        if Scaling:
            for i in list(range(0,n,1)):
                D_i_i = max(D[i,i], np.linalg.norm(Jx[:,i]))
                D[i, i] = D_i_i
                D_inv[i,i] = 1/D_i_i
            D_2 = np.dot(D, D)
        else: # D stays simply the identity matrix
            pass
        
        ## Calculation of the Gauss−Newton step
        # QR Decomposition of Jx
        Q, R, Pindex = scipy.linalg.qr(Jx, pivoting=True)
        # Calculate the permutation matrix P
        P = np.eye(n)[:,Pindex]
        # Gauss−Newton step depends if Jx has full rank
        rank = np.linalg.matrix_rank(Jx)
        if rank == n: # unique solution
            y = scipy.linalg.solve_triangular(
                    R[0:n,:], np.dot(Q[:,0:n].T, −r(x))
                )
        else:
            y = np.zeros((n))
            y[0:rank] = scipy.linalg.solve_triangular(
                           R[0:rank,0:rank], np.dot(Q[:,0:rank].T, −r(x))
                        )
        p = np.dot(P, y)
        Dp = np.linalg.norm(np.dot(D, p))
        # Check if p is in the trust region
        if Dp <= ((1+ sigma )∗Delta ):
            # p is good enough.
            alpha = 0
        else:
            ## Algorithm 2 is started to find a suitable alpha
            # Calculate safeguards for the iteration
            # substitute J by J_scaled
            J_scaled = np.dot(Jx, D_inv)

            # upper boundary u
            u = np.linalg.norm(np.dot(J_scaled.T, r(x)))/Delta
            # lower boundary l
            # It depends if Jx has full rank
            if rank == n:
                q = scipy.linalg.solve_triangular(
                        R[0:n ,:].T, np.dot(P.T,np.dot(D_2 ,p)), lower = True
                    )
                # Compute l = −phi(0)/phi’(0)
                l = (Dp − Delta)/(np.linalg.norm(q)∗∗2/Dp)
            else:
                l = 0
                
            alpha = 1 if u == inf else max(0.001∗u, (l∗u)∗∗(0.5))

            # Start the iteration process
            while Dp > (1 + sigma)∗Delta or Dp < (1 − sigma)∗Delta:
                if alpha == inf: 
                    print(
                        "Error: "
                        "The LM method fails to converge "
                        "(alpha gets too large). "
                        "Please try a different starting point."
                    )
                    return x, goal.iterations, 0, 0, 0
                
                # safeguarding alpha
                if alpha <= l or alpha > u:
                    alpha = max(0.001∗u, (l∗u)∗∗(0.5))
                
                D_lambda = np.dot(P.T, np.dot(D,P))
                R_I = np.concatenate((R, alpha∗∗(0.5)∗D_lambda ), axis = 0)
                
                R_lambda , Q_lambda2 = givens_qr (R_I ,n,m)

                Q_lambda = np.dot(
                                np.concatenate(
                                  (np.concatenate((Q,np.zeros((m,n))), axis = 1),
                                   np.concatenate((np.zeros((n,m)),P), axis = 1)),
                                   axis = 0
                                ), Q_lambda2
                            )
                # n additional zeros for r
                r_0 = np.append(r(x), np.zeros(n))
                # p can now be calculated, like in (29)
                p = np.dot(
                        P, 
                        scipy.linalg.solve_triangular(
                            R_lambda [0:n,:],
                            np.dot( Q_lambda [: ,0:n].T, −r_0)
                        )
                    )
                
                # often used, thus stored:
                Dp = np. linalg .norm(np.dot(D,p))
                # Calculate phi (alpha) and phi’(alpha)
                q = scipy.linalg.solve_triangular(
                        R_lambda [0:n ,:].T,
                        np.dot(P.T,np.dot(D_2 ,p)), 
                        lower=True
                    )
                phi = Dp − Delta
                phi_derivative = −np.linalg.norm(q)∗∗2/ Dp
                # update u
                if phi < 0:
                    u = alpha
                # update l
                l = max(l, alpha − phi/ phi_derivative )
                # Compute the new alpha
                alpha -= ((phi+Delta )/Delta) ∗ (phi/ phi_derivative)
                
        ## Calculation of rho
        fxp = goal.f(x+p) 
        # Note : fxp might be nan or inf
        if fxp > fx or fxp == inf or np.isnan(fxp ):
            rho = 0
        else:
            ared = 1 − (fxp/fx)
            pred = (0.5∗np.linalg.norm(np.dot(Jx, p))∗∗2)/fx+(alpha∗Dp∗∗2)/fx
            rho = ared/pred
        # Update Delta accordingly
        if rho < 0.25:
            Delta *= 0.25
        else:
            if (rho > 0.75 and (Dp >= (1−sigma)∗Delta)):
                Delta = min(2∗Delta, Delta_max)
            else:
                Delta = Delta
        
        if rho > eta:
            # take the step
            x = x + p 
            # Update info
            fx = fxp
            Jx = goal.J(x)
            norm_Jx = np.linalg.norm(Jx)
            tolerance = min((tol_rel ∗ norm_Jx + tol_abs), eps)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

"""         
levenberg_marquardt
Description
-----------
This function performs the Levenberg-Marquardt method, an iterative algorithm to find a minimizer of a given nonlinear least-squares problem. See Chapter 3 for more details.

Parameters
----------
r : function
    Residual vector r(x).
J : function
    Jacobian matrix of the residual vector r(x).
x : float
    Starting point of the iteration process.
    The description of the optional parameters can be found in Table 1.

Returns
-------
x : ndarray
    The last iterate of the iteration process.
Information : list
    List which contains information about the iteration process. It contains for every taken step its norm, the new iterate x, and the norm of the gradient at x.
"""
