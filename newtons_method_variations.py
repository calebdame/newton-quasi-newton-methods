# newtons_method_variations.py

import sympy as sy
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from autograd import jacobian

def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    if np.isscalar(x0): # method for scalars
        for i in range(maxiter):
            x1 = x0 - alpha * f(x0) / Df(x0) # newton's method algoritm for scalars
            if abs(x1 - x0) < tol: # check convergeance
                return x1, True, i+1
            x0 = x1
        return x1, False, maxiter
    else:
        for i in range(maxiter):
            x1 = np.copy(x0 - alpha*np.linalg.solve(Df(x0),f(x0))) # newton's method algoritm for vectors
            k = np.linalg.norm(x1-x0)
            if k < tol: # check convergeance
                return x1, True, i +1 
            x0 = np.copy(x1)
        return x1, False, maxiter


def example(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    x = sy.symbols('x') # x as a symbol
    f = P1*(1+x)**N1 + P2*(1+x)**(-N2) - P1 - P2 # function
    f = sy.lambdify(x, f) # turn into function
    Df = sy.lambdify(x, sy.diff(P1*(1+x)**N1 + P2*(1-x)**(-N2) - P1 - P2)) # derivative
    x1 = 0.1 - f(0.1)/Df(0.1)
    while 1: # newtons method implementation
        x2 = x1 - f(x1)/Df(x1)
        if round(x1,7) == round(x2,7): 
            return x2
        x1 = x2
        

def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    alpha_space = np.linspace(0.0078125,1, 128, endpoint=True) #alpha values to try
    zeros = [] 
    for a in alpha_space:
        z, _, _ = newton(f,x0,Df,tol, maxiter,a) # run newton's method on each alpha
        zeros += [z]
    plt.scatter(alpha_space, zeros) # plot
    plt.yscale('symlog')
    plt.xlabel('Alpha values')
    plt.ylabel('Newton\'s Method Values')
    plt.title("Convergeance for different Alpha values")
    plt.show()

def example2():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    def f(x): # array of functions set to zreo
        return np.array([4*x[0]*x[1] - x[0], -x[1]**2+1-x[1]*x[0]])
    def Df(x): # Df
        return np.array([[4*x[1]-1],[4*x[0]],[-x[1]],[-2*x[1]-x[0]]]).reshape((2,2))
    while 1:
        xa = 0.5*(np.random.random(2)-0.5) # random value to test newtons method
        z1, converg1, _ = newton(f, xa, Df, maxiter=20,alpha=1)
        z2, converg2, _ = newton(f, xa, Df, maxiter=20,alpha=0.55)
        if converg1 == False or converg2 == False: # check if both converge
            continue
        if (np.allclose(z1, np.array([0,1])) or np.allclose(z1, np.array([0,-1]))) and np.allclose(z2, np.array([3.75,0.25])):
            return xa # return if converges to different values under different alphas


def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    x_real = np.linspace(domain[0],domain[1],res)
    x_imag = np.linspace(domain[2],domain[3],res)
    X_real, X_imag = np.meshgrid(x_real, x_imag) # initialize matrices
    X_0 = X_real + 1j*X_imag
    for i in range(iters): # Newton's method 
        Y = X_0 - f(X_0)/Df(X_0)
        X_0 = np.copy(Y)
    for i in range(len(Y[:,0])): # iterate through to assign zero vals
        for j in range(len(Y[0,:])):
            Y[i,j] = np.argmin(np.abs(zeros - Y[i,j]))
    plt.pcolormesh(x_real, x_imag, np.real(Y), cmap="brg") # plot mesh
    plt.xlabel("X Values")
    plt.ylabel("Imaginary Values")
    plt.title("Basins of Attraction")
    plt.show()    
