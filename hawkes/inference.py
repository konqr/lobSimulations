import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import gc

class ParametricFit():

    def __init__(self, data):
        self.data = data # list of 2D vectors : time, value of kernels

    def fitPowerLaw(self):
        Xs = sm.add_constant(np.hstack( [np.log(d[0]) for d in self.data] ))
        Ys = np.hstack([np.log(d[1]) for d in self.data])
        model = sm.OLS(Ys, Xs)
        res = model.fit()
        print(res.summary())
        thetas = res.params
        return thetas, res

    def fitExponential(self):
        Xs = sm.add_constant(np.hstack( [d[0] for d in self.data]))
        Ys = np.hstack([np.log(d[1]) for d in self.data])
        model = sm.OLS(Ys, Xs)
        res = model.fit()
        print(res.summary())
        thetas = res.params
        return thetas, res

    def fitBoth(self):
        thetasPowerLaw, resPowerLaw = self.fitPowerLaw()
        thetasExponential, resExponential = self.fitExponential()
        if resPowerLaw.aic > resExponential.aic:
            print("Exponential selected")
            thetas = thetasExponential
        else:
            print("Power Law selected")
            thetas = thetasPowerLaw
        return thetas