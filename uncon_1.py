import numpy as np
import matplotlib.pyplot as plt
from sympy import * 

class Function(object):
    def __init__(self, data):
        self.x1, self.x2 = symbols("x1, x2")
        self.data = data
        self.f = self.f()
        self.g1, self.g2 = self.g()
        self.G11, self.G12, self.G22 = self.G()
        self.f_ = lambdify(('x1', 'x2'), self.f, "numpy")
        self.g1_ = lambdify(('x1', 'x2'), self.g1, "numpy")
        self.g2_ = lambdify(('x1', 'x2'), self.g2, "numpy")
        self.G11_ = lambdify(('x1', 'x2'), self.G11, "numpy")
        self.G12_ = lambdify(('x1', 'x2'), self.G12, "numpy")
        self.G22_ = lambdify(('x1', 'x2'), self.G22, "numpy")
    
    def a(self, t):
        return exp(-(self.x1+self.x2)*t)
    
    def b(self, t):
        return self.x1/(self.x1+self.x2)*(1-exp(-(self.x1+self.x2)*t))

    def c(self, t):
        return self.x2/(self.x1+self.x2)*(1-exp(-(self.x1+self.x2)*t))

    def f(self):
        result = 0
        for meta in self.data:
            t = meta[0]
            result = result + (meta[1]-self.a(t))**2 + (meta[2]-self.b(t))**2 + (meta[3]-self.c(t))**2
        return result
    
    def g(self):
        g1 = diff(self.f, self.x1)
        g2 = diff(self.f, self.x2)
        return g1, g2

    def G(self):
        G11 = diff(self.g1, self.x1)
        G12 = diff(self.g1, self.x2)
        G22 = diff(self.g2, self.x2)
        return G11, G12, G22

    def p_line(self, alpha, x, p, dir):
        return self.f_value(x) + p * np.dot(self.g_value(x).T, dir) * alpha

    def f_value(self, x):
        return self.f_(x[0], x[1])
    
    def g_value(self, x):
        g1 = self.g1_(x[0], x[1])
        g2 = self.g2_(x[0], x[1])
        return np.array([g1, g2]).reshape(2, 1)

    def G_value(self, x):
        G11 = self.G11_(x[0], x[1])
        G12 = self.G12_(x[0], x[1])
        G22 = self.G22_(x[0], x[1])
        return np.array([[G11[0], G12[0]],[G12[0], G22[0]]])
    
    def q_value(self, x, s):
        g = self.g_value(x)
        G = self.G_value(x)
        return 0.5*np.dot(np.dot(s.T, G), s) + np.dot(g.T, s) + self.f_value(x)

    def Steihaug(self, x, delta, e):
        g = self.g_value(x)
        G = self.G_value(x)
        xj = np.array([0, 0]).reshape(2, 1)
        rj = g
        pj = -g
        if np.linalg.norm(rj) < e:
            return xj
        while True:
            if np.dot(np.dot(pj.T, G), pj) < 0:
                a = np.dot(pj.T, pj)
                b = 2 * np.dot(xj.T, pj)
                c = np.dot(xj.T, xj) - delta**2
                t1 = (-b+(b**2-4*a*c)**0.5) / (2*a)
                t2 = (-b-(b**2-4*a*c)**0.5) / (2*a)
                re1 = self.q_value(x, xj + t1 * pj)
                re2 = self.q_value(x, xj + t2 * pj)
                if re1 < re2:
                    return xj + t1 * pj
                else:
                    return xj + t2 * pj
            aj = np.dot(rj.T, rj) / np.dot(np.dot(pj.T, G), pj)
            xj_1 = xj + aj * pj
            if np.linalg.norm(xj_1) >= delta:
                a = np.dot(pj.T, pj)
                b = 2 * np.dot(xj.T, pj)
                c = np.dot(xj.T, xj) - delta**2
                t = (-b+(b**2-4*a*c)**0.5) / (2*a)
                return xj + t * pj
            rj_1 = rj + aj * np.dot(G, pj)
            if np.linalg.norm(rj_1) < e:
                return xj_1
            beatj_1 = np.dot(rj_1.T, rj_1) / np.dot(rj.T, rj)
            pj_1 = -rj_1 + beatj_1 * pj

            pj = pj_1
            xj = xj_1
            rj = rj_1
        
    def Base_Newton(self, x0, e):
        points = list()
        points.append(x0)
        xk = x0
        while np.linalg.norm(self.g_value(xk)) > e:
            G = self.G_value(xk)
            dir = -np.dot(np.linalg.inv(G), self.g_value(xk))
            xk = xk + dir
            points.append(xk)
        points = np.array(points)
        return points

    def Linear_Search_Newton(self, x0, alpha, p, gamma, e):
        points = list()
        points.append(x0)
        xk = x0
        while np.linalg.norm(self.g_value(xk)) > 1e-8:
            I = np.array([[1, 0],[0, 1]])
            l = 0
            G = self.G_value(xk)
            while True:
                G_ = G + l*I
                if G_[0, 0] > 0 and np.linalg.det(G_) > 0:
                    break
                l = l + 1
            pk = -np.dot(np.linalg.inv(G_), self.g_value(xk))
            alpha_ = alpha
            while True:
                if self.f_value(xk + alpha_ * pk) <= self.p_line(alpha_, xk, p, pk):
                    break
                else:
                    alpha_ = alpha_ * gamma
            xk = xk + alpha_ * pk
            points.append(xk)
        points = np.array(points)
        return points

    def BFGS(self, x0, alpha, p, gamma, e):
        points = list()
        points.append(x0)
        xk = x0
        gk = self.g_value(xk)
        Hk = np.array([[1, 0],[0, 1]])
        I = np.array([[1, 0],[0, 1]])
        while np.linalg.norm(self.g_value(xk)) > e:
            pk = -np.dot(Hk, gk)
            alpha_ = alpha
            while True:
                if self.f_value(xk + alpha_ * pk) <= self.p_line(alpha_, xk, p, pk):
                    break
                else:
                    alpha_ = alpha_ * gamma
            xk_1 = xk + alpha_ * pk
            gk_1 = self.g_value(xk_1)
            sk = xk_1 - xk
            yk = gk_1 - gk
            t = 1 / np.dot(yk.T, sk)
            Hk = np.dot(np.dot(I-t*np.dot(sk, yk.T), Hk), I-t*np.dot(yk, sk.T)) + t*np.dot(sk, sk.T)
            xk = xk_1
            gk = gk_1
            points.append(xk)
        points = np.array(points)
        return points

    def Trust_Region_Newton(self, x0, delta, e):
        points = list()
        points.append(x0)
        xk = x0
        while np.linalg.norm(self.g_value(xk)) > e:
            sk = self.Steihaug(xk, delta, e)
            if self.f_value(xk) == self.q_value(xk,sk):
                pk = 1
            else:
                pk = (self.f_value(xk)-self.f_value(xk+sk)) / (self.f_value(xk)-self.q_value(xk, sk))
            if pk < 0.25:
                delta = np.linalg.norm(sk) / 4
            if pk > 0.75 and np.linalg.norm(sk) == delta:
                delta = 2 * delta
            if pk > 0:
                xk = xk + sk
            points.append(xk)
        points = np.array(points)
        return points


if __name__ == "__main__":
    data = np.array([[0.1, 0.913, 0.0478, 0.0382],
                     [0.2, 0.835, 0.0915, 0.0732],
                     [0.3, 0.763, 0.1314, 0.1051],
                     [0.4, 0.697, 0.1679, 0.1343],
                     [0.5, 0.637, 0.2013, 0.1610],
                     [0.6, 0.582, 0.2318, 0.1854],
                     [0.7, 0.532, 0.2596, 0.2077],
                     [0.8, 0.486, 0.2851, 0.2281],
                     [0.9, 0.444, 0.3084, 0.2467],
                     [1.0, 0.406, 0.3296, 0.2637]])
    # initialize parameter
    func = Function(data)
    x0 = np.array([2.0, 2.0]).reshape(2, 1)
    e = 1e-8
    # Armijo line search parameter
    alpha = 1.0
    p = 0.2
    gamma = 0.9
    # Trust region newton parameter
    delta = 1.0

    # choose method
    # points = func.Base_Newton(x0, e)
    # points = func.Linear_Search_Newton(x0, alpha, p, gamma, e)
    # points = func.BFGS(x0, alpha, p, gamma, e)
    points = func.Trust_Region_Newton(x0, delta, e)

    print(points[-1])
    print(func.f_value(points[-1]))
    print(len(points))

    x1 = np.arange(0, 2.5, 0.001)
    x2 = np.arange(-0.5, 2.5, 0.001)
    X1, X2 = np.meshgrid(x1, x2)
    Z = func.f_value(np.array([X1, X2]))
    N = np.array([i*i*i*0.0001 for i in range(100)])
    CS = plt.contour(X1, X2, Z, N, colors='black')
    plt.clabel(CS, inline=True, fmt='%1.3f', fontsize=5)
    plt.plot(points[:, 0, 0], points[:, 1, 0], color='red', 
            label='Iterative trajectory', ls='-', marker='o', ms=3, mec='blue', mfc='blue')
    plt.legend()
    plt.show()