import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter

def f(x): return 1/(25*(x**2) + 1)

def main(n:int):
    d = np.linspace(start=-1 , stop=1 , num=1000 , dtype=float) # domain
    knots = np.linspace(start=-1 , stop=1 , num=n , dtype=float) 
    f_interpolating = inter.KroghInterpolator(xi=knots , yi=f(knots))
    plt.plot(d , f(d) , label='f(x)')
    plt.plot(d , f_interpolating(d) , label='f_interpolating(x)')
    plt.scatter(knots , f(knots) , color='red')
    plt.legend()
    plt.title(f'Interpolation with {n} knots')
    plt.show()


# for n in range(4, 10+1):
#     main(n)

main(11)