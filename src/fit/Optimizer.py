import numpy as np

class Optimizer:

    def __init__(self, dictInput):
        self.dictIP = dictInput

    def projectBounds(self, params, LB, UB):
        params[params < LB] = LB
        params[params > UB] = UB

    def ComputeWorkingSet(self, params, grad, LB, UB):
        mask = np.ones_like(grad, dtype=int)
        mask[(params < LB + self.optTol * 2) & (grad >= 0)] = 0
        mask[(params > UB - self.optTol * 2) & (grad <= 0)] = 0
        working = np.where(mask == 1)[0]
        return working

    def isLegal(self, x):
        return not np.isnan(x).any()

    def lbfgsUpdate(self, y, s, corrections, old_dirs, old_stps, Hdiag):
        ys = np.dot(y, s)
        if ys > 1e-10:
            numCorrections = old_dirs.shape[1]

            if numCorrections < corrections:
                old_dirs = np.hstack([old_dirs, np.expand_dims(s, axis=1)])
                old_stps = np.hstack([old_stps, np.expand_dims(y, axis=1)])
            else:
                old_dirs[:, :-1] = old_dirs[:, 1:]
                old_stps[:, :-1] = old_stps[:, 1:]
                old_dirs[:, -1] = s
                old_stps[:, -1] = y

            Hdiag = ys / np.dot(y, y)

    def lbfgs(self, g, s, y, Hdiag):
        k = s.shape[1]

        ro = np.zeros(k)
        for i in range(k):
            ro[i] = 1 / np.dot(y[:, i], s[:, i])

        q = np.zeros((len(g), k + 1))
        r = np.zeros((len(g), k + 1))
        al = np.zeros(k)
        be = np.zeros(k)

        q[:, -1] = g

        for i in range(k - 1, -1, -1):
            al[i] = ro[i] * np.dot(s[:, i], q[:, i + 1])
            q[:, i] = q[:, i + 1] - al[i] * y[:, i]

        r[:, 0] = Hdiag * q[:, 0]

        for i in range(k):
            be[i] = ro[i] * np.dot(y[:, i], r[:, i])
            r[:, i + 1] = r[:, i] + s[:, i] * (al[i] - be[i])

        return r[:, -1]

    def PLBFGS(self, LB, UB):
        self.maxIter_ = 10000

        print("{:10} {:10} {:10} {:10} {:10}".format("Iteration", "FunEvals", "Step Length", "Function Val", "Opt Cond"))
        nVars = len(self.process_.GetParameters())

        x = (np.random.randn(nVars) + 1) * 0.5
        self.projectBounds(x, LB, UB)

        f = 0
        g = np.zeros_like(x)
        self.process_.SetParameters(x)
        self.process_.NegLoglikelihood(f, g)

        working = self.ComputeWorkingSet(x, g, LB, UB)

        if len(working) == 0:
            print("All variables are at their bound and no further progress is possible at initial point")
            return
        elif np.linalg.norm(g[working]) <= self.optTol:
            print("All working variables satisfy optimality condition at initial point")
            return

        i = 1
        funEvals = 1
        maxIter = self.maxIter_

        corrections = 100
        old_dirs = np.zeros((nVars, 0))
        old_stps = np.zeros((nVars, 0))
        Hdiag = 0
        suffDec = 1e-4

        g_old = g.copy()
        x_old = x.copy()

        while funEvals < maxIter:
            d = np.zeros_like(x)

            if i == 1:
                d[working] = -g[working]
                Hdiag = 1
            else:
                self.lbfgsUpdate(g - g_old, x - x_old, corrections, old_dirs, old_stps, Hdiag)
                d[working] = self.lbfgs(-g[working], old_dirs[:, working], old_stps[:, working], Hdiag)

            g_old = g.copy()
            x_old = x.copy()

            f_old = f
            gtd = np.dot(g, d)
            if gtd > -self.optTol:
                print("Directional Derivative below optTol")
                break

            if i == 1:
                t = min(1 / np.sum(np.abs(g[working])), 1.0)
            else:
                t = 1.0

            x_new = x + t * d
            self.projectBounds(x_new, LB, UB)
            self.process_.SetParameters(x_new)
            self.process_.NegLoglikelihood(f_new, g_new)
            funEvals += 1

            lineSearchIters = 1
            while f_new > f + suffDec * np.dot(g, x_new - x) or np.isnan(f_new):
                temp = t
                t = 0.1 * t

                if t < temp * 1e-3:
                    t = temp * 1e-3
                elif t > temp * 0.6:
                    t = temp * 0.6

                if np.sum(np.abs(t * d)) < self.optTol:
                    print("Line Search failed")
                    t = 0
                    f_new = f
                    g_new = g
                    break

                x_new = x + t * d
                self.projectBounds(x_new, LB, UB)
                self.process_.SetParameters(x_new)
                self.process_.NegLoglikelihood(f_new, g_new)
                funEvals += 1
                lineSearchIters += 1

            x = x_new
            f = f_new
            g = g_new

            working = self.ComputeWorkingSet(x, g, LB, UB)

            if len(working) == 0:
                print("{:10} {:10} {:10.2f} {:10.2f} {:10}".format(i, funEvals, t, f, 0))
                print("All variables are at their bound and no further progress is possible")
                break
            else:
                print("{:10} {:10} {:10.2f} {:10.2f} {:10.2f}".format(i, funEvals, t, f, np.sum(np.abs(g[working]))))

                if np.linalg.norm(g[working]) <= self.optTol:
                    print("All working variables satisfy optimality condition")
                    break

            if np.sum(np.abs(t * d)) < self.optTol:
                print("Step size below optTol")
                break

            if np.abs(f - f_old) < self.optTol:
                print("Function value changing by less than optTol")
                break

            if funEvals > maxIter:
                print("Function Evaluations exceed maxIter")
                break

            i += 1

        print()