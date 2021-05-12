#Recherche du zéro, dichotomie
"""
def f(z):
    f = z**5+z-1
    return f

def dichotomie(f, inf, sup, precision):
    n = 0
    while sup-inf>precision:
        c = (sup+inf)/2
        n = n+1
        if f(c)*f(inf) < 0:
            sup=c
        else:
            inf=c
    return(c, n)

inf = -10
sup = 10
precision = 10**-5
print(dichotomie(f, inf, sup, precision))
"""

#Recherche du zéro, Newton
"""
import math as m 

def f(z):
    f = z**5+z-1
    return f

def f_prime(z):
    f_prime = 5*z**4 + 1
    return f_prime

def newton(f, f_prime, x0, precision):
    xN = x0
    xNPLUS1 = xN-(f(xN)/f_prime(xN+1))
    n = 1
    while m.abs(xNPLUS1-xN)>precision:
        xN = xNPLUS1
        xNPLUS1 = xNPLUS1 = xN-(f(xN)/f_prime(xN+1))
        n = n+1
    return xNPLUS1,n

x0 = 0
precision = 10**-5
print(newton(f, f_prime, x0, precision))
"""

#Intégral, méthode des rectangles
"""
def f(x):
    f=x**2
    return f

def rectangle(f, inf, sup, n):
    h = (sup-inf)/n
    res = 0
    for i in range(n):
        res += h*f(inf + i*h)
    return res

inf = -2
sup = 2
print(rectangle(f, inf, sup, 20))
"""

#Euler u' + u = 1 1er ordre

"""
import pylab as pl

def f_prime(f):
    return 1 - f

inf = 0
sup = 3
N = 30
h = (sup-inf)/N
fn=0
tn= 0
F=[0]
T=[0]

for i in range(N):
    tn += h
    fn += h*f_prime(fn) #penser à multiplier par h
    T.append(tn)
    F.append(fn)

pl.plot(T, F)
pl.show()
"""

#Euler, second  z'' + 2*z' + z = 0 (pour simplifier, j'ai enlevé le lambda)

#Question 4:
#Soit z2=u', et z1'=z2 (2) on a:
#z2' + 2*z2 + z1 = 0 (1)

#Question 6:
def f(z, t):
    z_prime1 = z[1][t] #d'après (2)
    z_prime2 = -2*z[1][t] - z[0][t] #d'après (1)
    return z_prime1, z_prime2

#Question 7
def euleur(z, f, t, delta_t):
    return z+delta_t*f(z, t)

import numpy as np
import pylab as pl

#Peut-on transformer le tuplet (z1, z2) en liste, ce qui est plus simple à manier ?

z1_0 = 0
z2_0 = 0
inf = 0
sup = 25
n = 250
h = (sup-inf)/n
Z = np.zeros((n, 2))
T = np.linspace(inf, sup)
Z[0, :] = [z1_0, z2_0] #Que signifie Z[0, :] ?
t = inf

for i in (1, n+1):
    Z[i, :] = [euleur(Z[i-1,:], f, t, h)]

pl.plot(T, Z)
pl.show()

#On a les valeurs de z1 et z2, mais qu'est ce qu'il en est de l'EDL d'origine ?
