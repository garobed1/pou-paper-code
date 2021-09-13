
import numpy as np

class rk4_pend_solver():

    def __init__(self):
        self.m = 1
        self.g = 1
        self.c = 1
        self.L = 1

    def set_param(self, m, g, c, L):
        """Sets ODE constants, call to evaluate for a sample
        """
        self.m = m
        self.g = g
        self.c = c
        self.L = L


    def runge_kutta_4th(self, func, tspan, u0, num_steps):
        """Solves a system of ODEs using classical RK4 method.
    ​
        Solves the ODE du/dt = F(u,t) using the RK4 method.  `func` defines the
        function F(y,t), `tspan[0]` is the initial time, `tspan[1]` the final time,
        `u0` the initial condition, and `num_steps` the number of steps.  Returns
        NumPy arrays `t` and `u`, the time points and solution.
        """
        if tspan[0] > tspan[1]:
            raise ValueError("tspan[0] must be less than or equal to tspan[1].")
        if num_steps < 1:
            raise ValueError("num_steps number of steps must be positive.")
        t = np.linspace(tspan[0], tspan[1], num_steps+1)
        u = np.zeros((u0.size, num_steps+1))
        dt = (tspan[1] - tspan[0])/num_steps
        u[:,0] = u0
        for n in range(num_steps):
            k1 = dt*func(u[:,n], t[n])
            k2 = dt*func(u[:,n] + 0.5*k1, t[n] + 0.5*dt)
            k3 = dt*func(u[:,n] + 0.5*k2, t[n] + 0.5*dt)
            k4 = dt*func(u[:,n] + k3, t[n] + dt)
            u[:,n+1] = u[:,n] + (k1 + 2.0*(k2 + k3) + k4)/6.0
        return t, u

    def solve_for_uf(self, m, g, c, L, tspan, u0, num_steps):
        '''Set parameters and solve for the final position'''
        self.set_param(m, g, c, L)        
        
        t, u = self.runge_kutta_4th(self.func, tspan, u0, num_steps)

        return u[0, len(u[0])-1]

    def func(self, u, t):
        '''Damped Pendulum System
        '''
        F = np.zeros(u.size)
        F[0] = u[1]
        F[1] = -(self.c/self.m)*u[1] - (self.g/self.L)*np.sin(u[0])

        return F