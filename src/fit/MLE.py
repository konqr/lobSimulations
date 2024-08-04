from tick.hawkes import HawkesSumExpKern

class MLE():

    def __init__(self, data, **kwargs):
        self.data = data
        self.cfg = kwargs

    def fit(self, tol=1e-6, max_iter=100000, elastic_net_ratio = 0, penalty = "none", solver = "bfgs"):
        hawkes_learner = HawkesSumExpKern(decays = [1.7e3, 0.1*1.7e3, 0.01*1.7e3, 0.001*1.7e3],solver=solver, verbose = True, penalty = penalty, tol=tol, max_iter=max_iter, elastic_net_ratio =elastic_net_ratio)
        hawkes_learner.fit(self.data)
        baseline = hawkes_learner.baseline
        kernels = hawkes_learner.coeffs
        return (baseline, kernels)