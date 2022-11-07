# Optimizers for use in Lagaris5.ipynb and problem1.ipynb
# Also author or Gauss-Newton and Levenberg-Marquardt optimizers in problem2.ipynb

import torch
import numpy as np

from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, lossfun, gradfun, w, maxit, params=None,  gamma=0.9, eta=0.5, tol=1e-3, stochastic=False, batchsize=None, data=None) -> None:
        self.gradfun = gradfun
        self.lossfun = lossfun
        self.w = w
        self.params = params
        self.maxit = maxit
        self.gamma = gamma
        self.eta = eta
        self.tol = tol
        self.batchsize = batchsize
        self.data = data
        self.stochastic = stochastic

    def _genBatch(self) -> torch.tensor:
        """
        Randomly splits data into batches of size batch_size.
        """
        idx = np.random.choice(len(self.data), self.batchsize, replace=False)
        return self.data[idx, :]

    def _updateGrad(self, batch=None) -> None:
        if (self.stochastic):
            self.grad = self.gradfun(self.w.squeeze(), batch)
            self.gradnorm = torch.linalg.norm(self.grad)
            self.gradnormvals[self.it] = self.gradnorm
            self.lossvals[self.it] = self.lossfun(self.w.squeeze(), batch)
        else:
            self.res, self.Jac = self.gradfun(self.params, self.w)
            Jtrans = torch.transpose(self.Jac, 0, 1)
            self.grad = torch.matmul(Jtrans, self.res)

            self.gradnorm = torch.linalg.norm(self.grad)
            self.gradnormvals[self.it] = self.gradnorm
            self.lossvals[self.it] = self.lossfun(self.res)

    @abstractmethod
    def _linesearch(self, a, d) -> float:
        f0 = self.lossfun(self.res)
        f1 = f0+a*self.aux+1
        kmax = np.ceil(np.log(1e-14)/np.log(self.gamma))
        k = 0
        while (f1 > f0+a*self.aux and k < kmax):
            wtry = self._step(a, d)
            rnew, Jnew = self.gradfun(self.params, wtry)
            f1 = self.lossfun(rnew)
            a = a*self.gamma
            k += 1

        return a

    @abstractmethod
    def minimize(self) -> tuple:
        self.it = 0
        self._initSolver()

        while (self.gradnorm > self.tol and self.it < self.maxit):
            if (self.stochastic):

                batch = self._genBatch()
                self._updateGrad(batch)
                upd = self._findDirection()
                self.w = upd
                self.it += 1
                if (self.it % 10 == 0):
                    print("Iter ", self.it, ": loss = ",
                          self.lossvals[self.it-1], " gradnorm = ", self.gradnorm)
            else:
                a = 0.5

                upd = self._findDirection(a)

                self.w = upd

                self._updateGrad()

                print("Iter ", self.it, ": loss = ",
                      self.lossvals[self.it], " gradnorm = ", self.gradnorm)

                self.it += 1

        return self.w, self.it, self.lossvals[0:self.it], self.gradnormvals[0:self.it]

    @abstractmethod
    def _initSolver(self) -> torch.tensor:
        if (self.stochastic):

            self.lossvals = torch.zeros(self.maxit)
            self.gradnormvals = torch.zeros(self.maxit)
            batch = self._genBatch()
            self.grad = self.gradfun(self.w, batch)
            self.res = self.lossfun(self.w, batch)
            self.gradnorm = torch.linalg.norm(self.grad)
            self.gradnormvals[0] = self.gradnorm

            self.lossvals[0] = self.res  # self.lossfun(self.w, batch)

            return batch
        else:
            self.res, self.Jac = self.gradfun(self.params, self.w)

            self.lossvals = torch.zeros(self.maxit)
            self.gradnormvals = torch.zeros(self.maxit)
            self.lossvals[0] = self.lossfun(self.res)
            Jtrans = torch.transpose(self.Jac, 0, 1)
            self.grad = torch.matmul(Jtrans, self.res)  # grad = J^\top r
            self.gradnorm = torch.linalg.norm(self.grad)
            self.gradnormvals[0] = self.gradnorm

            return None

    @abstractmethod
    def _step(self, a, d) -> torch.tensor:
        pass

    @abstractmethod
    def _findDirection(self, a) -> torch.tensor:
        pass


class Adam(Optimizer):

    def __init__(self, lossfun, gradfun, w, maxit, params, gamma=0.9, eta=0.5, beta1=0.9, tol=0.001, beta2=0.999, eps=1e-8, alpha=None, stochastic=False, batchsize=None, data=None) -> None:
        super().__init__(lossfun, gradfun, w, maxit, params, gamma,
                         eta, tol, stochastic, batchsize, data)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.alpha = alpha

    def _initSolver(self) -> None:
        #
        super()._initSolver()

        self.m = torch.zeros(self.w.size())
        self.v = torch.zeros(self.w.size())

    def _step(self, a, d) -> torch.tensor:
        return self.w + a*d

    def _findDirection(self, a=None) -> torch.tensor:

        self.m = (self.beta1*self.m+(1-self.beta1)*self.grad)
        self.v = (self.beta2*self.v+(1-self.beta2)*self.grad**2)

        d = -self.m/(np.sqrt(self.v)+self.eps)

        if (self.alpha):
            return self.alpha*d
        else:
            self.aux = self.eta*torch.tensordot(self.grad, d)
            a = self._linesearch(a, d)
            return self._step(a, d)

    def minimize(self) -> tuple:
        return super().minimize()


class Nesterov(Optimizer):
    def __init__(self, lossfun, gradfun, w, maxit, params, gamma=0.9, eta=0.5, tol=0.001, alpha=None, stochastic=False, batchsize=None, data=None) -> None:
        super().__init__(lossfun, gradfun, w, maxit, params, gamma,
                         eta, tol, stochastic, batchsize, data)

        self.alpha = alpha

    def _linesearch(self, a, d) -> float:
        return super()._linesearch(a, d)

    def _step(self, a, d) -> torch.tensor:
        x1 = self.w - a*d
        return (1+self.mu)*x1 - self.mu*self.w

    def _initSolver(self) -> None:
        return super()._initSolver()

    def _findDirection(self, a=None) -> torch.tensor:
        self.mu = 1-3/(5+self.it)

        if (self.alpha):
            return self._step(self.alpha, self.grad)
        else:
            self.aux = -self.eta*torch.tensordot(self.grad, self.grad)
            a = self._linesearch(a, self.grad)
            return self._step(a, self.grad)

    def minimize(self) -> tuple:
        return super().minimize()


class GradientDescent(Optimizer):
    def __init__(self, lossfun, gradfun, w, maxit, params, gamma=0.9, eta=0.5, tol=0.001, alpha=None, stochastic=False, batchsize=None, data=None) -> None:
        super().__init__(lossfun, gradfun, w, maxit, params, gamma,
                         eta, tol, stochastic, batchsize, data)

        self.alpha = alpha

    def _linesearch(self, a, d) -> float:
        return super()._linesearch(a, d)

    def _initSolver(self) -> None:
        return super()._initSolver()

    def _step(self, a, d) -> torch.tensor:
        return self.w-a*d

    def _findDirection(self, a=None) -> torch.tensor:

        if (self.alpha):
            return self._step(self.alpha, self.grad)
        else:
            self.aux = -self.eta*torch.tensordot(self.grad, self.grad)
            a = self._linesearch(a, self.grad)
            return self._step(a, self.grad)

    def minimize(self) -> tuple:
        return super().minimize()


class LBFGS(Optimizer):
    def __init__(self, lossfun, gradfun, w, maxit, params=None, gamma=0.9, eta=0.5, tol=0.02, stochastic=False, batchsize=None, data=None, m=5, alpha=None) -> None:
        super().__init__(lossfun, gradfun, w, maxit, params,
                         gamma, eta, tol, stochastic, batchsize, data)

        self.m = m
        self.alpha = alpha
        self.kmax = np.ceil(np.log(1e-14)/np.log(self.gamma))

    def _linesearch(self, a, d, batch) -> tuple:

        f0 = torch.linalg.norm(self.lossfun(self.w.squeeze(), batch))
        f1 = f0+a*self.aux+1

        k = 0
        while (torch.linalg.norm(f1) > torch.linalg.norm(f0+a*self.aux) and k < self.kmax):
            wtry = self.w + a*d
            f1 = torch.linalg.norm(self.lossfun(wtry.squeeze(), batch))
            a = a*self.gamma
            k += 1

        return a, k

    def _initSolver(self) -> None:
        super()._initSolver()
        self.batch = self._genBatch()
        self.s = torch.zeros(self.w.size()[0], self.m, dtype=torch.float64)
        self.y = torch.zeros(self.w.size()[0], self.m, dtype=torch.float64)
        self.rho = torch.zeros(self.m, dtype=torch.float64)
        if (self.alpha):
            a = self.alpha
        else:
            self.aux = self.eta*torch.dot(self.grad, -self.grad)
            a, _ = self._linesearch(1, self.grad, self.batch)
        xnew = - a*self.grad
        gnew = self.gradfun(xnew.squeeze(), self.batch)
        self.s[:, 0] = xnew - self.w
        self.y[:, 0] = gnew - self.grad

        self.rho[0] = 1/torch.dot(self.s[:, 0], self.y[:, 0])

        self.w = xnew
        self.grad = gnew
        self.gradnorm = torch.linalg.norm(self.grad)

    def _findDirection(self, a, s=None, y=None) -> torch.tensor:
        s = s if torch.is_tensor(s) else self.s
        y = y if torch.is_tensor(y) else self.y

        m = s.size()[1]
        g = self.grad
        aa = torch.zeros(m)
        for i in range(m):

            aa[i] = self.rho[i]*torch.dot(s[:, i], self.grad)
            g = g - aa[i]*y[:, i]

        gam = torch.dot(s[:, 0], y[:, 0])/torch.dot(y[:, 0], y[:, 0])

        g = g*gam
        for i in range(m-1, 0, -1):
            aux = self.rho[i]*torch.dot(y[:, i], g)
            g = g + (aa[i] - aux)*s[:, i]

        if (self.alpha):
            return -self.alpha*g
        else:
            self.aux = self.eta*torch.dot(g, -g)
            a, k = self._linesearch(1, -g, self.batch)
            if k == self.kmax:
                self.aux = self.eta*torch.dot(self.grad, -self.grad)
                a, _ = self._linesearch(1, -self.grad, self.batch)
            return -a*g

    def _step(self, a, d) -> torch.tensor:
        return a*d

    def minimize(self) -> tuple:
        self.it = 1
        self._initSolver()

        while (self.gradnorm > self.tol and self.it < self.maxit):
            self._updateGrad(self.batch)
            a = 1
            if self.it < self.m:
                s = self.s[:, :self.it]
                y = self.y[:, :self.it]

                step = self._findDirection(a, s, y)
            else:
                step = self._findDirection(a, s, y)

            xnew = self.w + step
            gnew = self.gradfun(xnew.squeeze(), self.batch)

            if ((self.it % self.m == 0) or (self.it < self.m)):
                self.batch = self._genBatch()

                d = self.s.size()[0]
                self.s = torch.roll(self.s, d, 0)
                self.y = torch.roll(self.y, d, 0)
                self.rho = torch.roll(self.rho, 1)

                self.s[:, 0] = step
                self.y[:, 0] = gnew - self.grad
                self.rho[0] = 1/torch.dot(step, self.y[:, 0])

            self.w = xnew
            self.grad = gnew

            self.it += 1

            print("Iter ", self.it, ": loss = ",
                  self.lossvals[self.it-1], " gradnorm = ", self.gradnorm)

        return self.w, self.it, self.lossvals[0:self.it], self.gradnormvals[0:self.it]


class GaussNewton(Optimizer):
    def __init__(self, lossfun, gradfun, w, maxit, params, gamma=0.9, eta=0.5, tol=0.001, alpha=None) -> None:
        super().__init__(lossfun, gradfun, w, maxit, params, gamma, eta, tol)

        self.alpha = alpha

    def _linesearch(self, a, d) -> float:
        return super()._linesearch(a, d)

    def _initSolver(self) -> None:
        return super()._initSolver()

    def _step(self, a, d) -> torch.tensor:
        return self.w+a*d

    def _findDirection(self, a) -> torch.tensor:
        a = 1
        Jtr = torch.transpose(self.Jac, 0, 1)
        B = torch.matmul(Jtr, self.Jac)
        b = torch.matmul(Jtr, self.res)
        p = torch.linalg.lstsq(B, -b)[0]
        self.aux = self.eta * torch.tensordot(self.grad, p)
        if (self.alpha):
            return self._step(self.alpha, p)
        else:
            a = self._linesearch(a, p)

            return self._step(a, p)

    def minimize(self) -> tuple:
        return super().minimize()


#
