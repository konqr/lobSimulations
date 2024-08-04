import numpy as np

from fit.Optimizer import Optimizer

class PlainHawkes:

    def __init__(self):
        self.num_sequences_ = 0
        self.num_dims_ = 0
        self.all_exp_kernel_recursive_sum_ = []
        self.all_timestamp_per_dimension_ = []
        self.observation_window_T_ = np.array([])
        self.intensity_itegral_features_ = []
        self.parameters_ = np.array([])
        self.Beta_ = np.array([])
        self.options_ = None

    def InitializeDimension(self, data):
        num_sequences_ = len(data)
        self.all_timestamp_per_dimension_ = [[[[] for _ in range(self.num_dims_)] for _ in range(num_sequences_)]]

        for c in range(num_sequences_):
            seq = data[c].GetEvents()

            for event in seq:
                self.all_timestamp_per_dimension_[c][event.DimentionID].append(event.time)

    def Initialize(self, data):
        self.num_sequences_ = len(data)
        self.num_dims_ = data[0].num_dims()
        self.all_exp_kernel_recursive_sum_ = np.empty((self.num_sequences_, self.num_dims_, self.num_dims_), dtype=object)
        self.all_timestamp_per_dimension_ = []  # will be initialized in InitializeDimension function
        self.observation_window_T_ = np.zeros(self.num_sequences_)
        self.intensity_integral_features_ = np.zeros((self.num_sequences_, self.num_dims_, self.num_dims_))

        self.InitializeDimension(data)

        for k in range(self.num_sequences_):
            for m in range(self.num_dims_):
                for n in range(self.num_dims_):
                    if len(self.all_timestamp_per_dimension_[k][n]) > 0:
                        self.all_exp_kernel_recursive_sum_[k][m][n] = np.zeros(len(self.all_timestamp_per_dimension_[k][n]))

                        if m != n:
                            for j in range(len(self.all_timestamp_per_dimension_[k][m])):
                                if self.all_timestamp_per_dimension_[k][m][j] < self.all_timestamp_per_dimension_[k][n][0]:
                                    self.all_exp_kernel_recursive_sum_[k][m][n][0] += np.exp(-self.Beta_[m, n] * (self.all_timestamp_per_dimension_[k][n][0] - self.all_timestamp_per_dimension_[k][m][j]))

                            for i in range(1, len(self.all_timestamp_per_dimension_[k][n])):
                                value = np.exp(-self.Beta_[m, n] * (self.all_timestamp_per_dimension_[k][n][i] - self.all_timestamp_per_dimension_[k][n][i - 1])) * self.all_exp_kernel_recursive_sum_[k][m][n][i - 1]

                                for j in range(len(self.all_timestamp_per_dimension_[k][m])):
                                    if (self.all_timestamp_per_dimension_[k][n][i - 1] <= self.all_timestamp_per_dimension_[k][m][j] < self.all_timestamp_per_dimension_[k][n][i]):
                                        value += np.exp(-self.Beta_[m, n] * (self.all_timestamp_per_dimension_[k][n][i] - self.all_timestamp_per_dimension_[k][m][j]))

                                self.all_exp_kernel_recursive_sum_[k][m][n][i] = value
                        else:
                            for i in range(1, len(self.all_timestamp_per_dimension_[k][n])):
                                self.all_exp_kernel_recursive_sum_[k][m][n][i] = np.exp(-self.Beta_[m, n] * (self.all_timestamp_per_dimension_[k][n][i] - self.all_timestamp_per_dimension_[k][n][i - 1])) * (1 + self.all_exp_kernel_recursive_sum_[k][m][n][i - 1])

        for c in range(self.num_sequences_):
            self.observation_window_T_[c] = data[c].GetTimeWindow()

            for m in range(self.num_dims_):
                for n in range(self.num_dims_):
                    event_dim_m = np.array(self.all_timestamp_per_dimension_[c][m])
                    self.intensity_integral_features_[c, m, n] = (1 - np.exp(-self.Beta_[m, n] * (self.observation_window_T_[c] - event_dim_m))).sum()

    def Intensity(self, t, data):
        intensity_dim = np.zeros(self.num_dims_)

        Lambda0_ = self.parameters_[:self.num_dims_]
        Alpha_ = self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        intensity_dim = Lambda0_

        seq = data.GetEvents()

        for event in seq:
            if event.time < t:
                for d in range(self.num_dims_):
                    intensity_dim[d] += Alpha_[event.DimensionID, d] * np.exp(-self.Beta_[event.DimensionID, d] * (t - event.time))
            else:
                break

        return np.sum(intensity_dim)

    def IntensityUpperBound(self, t, L, data):
        intensity_upper_dim = np.zeros(self.num_dims_)

        Lambda0_ = self.parameters_[:self.num_dims_]
        Alpha_ = self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        intensity_upper_dim = Lambda0_

        seq = data.GetEvents()

        for event in seq:
            if event.time <= t:
                for d in range(self.num_dims_):
                    intensity_upper_dim[d] += Alpha_[event.DimensionID, d] * np.exp(-self.Beta_[event.DimensionID, d] * (t - event.time))
            else:
                break

        return np.sum(intensity_upper_dim)

    def NegLoglikelihood(self, objvalue, gradient):
        if not self.all_timestamp_per_dimension_:
            print("Process is uninitialized with any data.")
            return

        gradient[:] = np.zeros(self.num_dims_ * (1 + self.num_dims_))

        grad_lambda0_vector = gradient[:self.num_dims_].reshape(-1, 1)
        grad_alpha_matrix = gradient[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        Lambda0_ = self.parameters_[:self.num_dims_].reshape(-1, 1)
        Alpha_ = self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        objvalue[0] = 0

        for k in range(self.num_sequences_):
            timestamp_per_dimension = self.all_timestamp_per_dimension_[k]
            exp_kernel_recursive_sum = self.all_exp_kernel_recursive_sum_[k]

            for n in range(self.num_dims_):
                obj_n = 0

                for i in range(len(timestamp_per_dimension[n])):
                    local_sum = Lambda0_[n] + 1e-4

                    for m in range(self.num_dims_):
                        local_sum += Alpha_[m, n] * exp_kernel_recursive_sum[m][n][i]

                    obj_n += np.log(local_sum)

                    grad_lambda0_vector[n] += 1 / local_sum

                    for m in range(self.num_dims_):
                        grad_alpha_matrix[m, n] += exp_kernel_recursive_sum[m][n][i] / local_sum

                obj_n -= ((Alpha_[:, n] / self.Beta_[:, n]) * self.intensity_integral_features_[k][:, n]).sum()

                grad_alpha_matrix[:, n] -= self.intensity_integral_features_[k][:, n] / self.Beta_[:, n]

                obj_n -= self.observation_window_T_[k] * Lambda0_[n]

                grad_lambda0_vector[n] -= self.observation_window_T_[k]

                objvalue[0] += obj_n

        gradient /= -self.num_sequences_
        objvalue /= -self.num_sequences_

        # Regularization for base intensity
        if self.options_.base_intensity_regularizer == 'L22':
            grad_lambda0_vector += self.options_.coefficients["LAMBDA"] * Lambda0_
            objvalue += 0.5 * self.options_.coefficients["LAMBDA"] * np.sum(Lambda0_ ** 2)

        elif self.options_.base_intensity_regularizer == 'L1':
            grad_lambda0_vector += self.options_.coefficients["LAMBDA"]
            objvalue += self.options_.coefficients["LAMBDA"] * np.sum(np.abs(Lambda0_))

        # Regularization for excitation matrix
        grad_alpha_vector = gradient[self.num_dims_:].reshape(-1, 1)
        alpha_vector = self.parameters_[self.num_dims_:].reshape(-1, 1)

        if self.options_.excitation_regularizer == 'L22':
            grad_alpha_vector += self.options_.coefficients["BETA"] * alpha_vector
            objvalue += 0.5 * self.options_.coefficients["BETA"] * np.sum(alpha_vector ** 2)

        elif self.options_.excitation_regularizer == 'L1':
            grad_alpha_vector += self.options_.coefficients["BETA"]
            objvalue += self.options_.coefficients["BETA"] * np.sum(np.abs(alpha_vector))

        return

    def Gradient(self, k, gradient):
        if not self.all_timestamp_per_dimension_:
            print("Process is uninitialized with any data.")
            return

        gradient[:] = np.zeros(self.num_dims_ * (1 + self.num_dims_))

        grad_lambda0_vector = gradient[:self.num_dims_].reshape(-1, 1)
        grad_alpha_matrix = gradient[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        Lambda0_ = self.parameters_[:self.num_dims_].reshape(-1, 1)
        Alpha_ = self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        timestamp_per_dimension = self.all_timestamp_per_dimension_[k]
        exp_kernel_recursive_sum = self.all_exp_kernel_recursive_sum_[k]

        for n in range(self.num_dims_):
            for i in range(len(timestamp_per_dimension[n])):
                local_sum = Lambda0_[n]

                for m in range(self.num_dims_):
                    local_sum += Alpha_[m, n] * exp_kernel_recursive_sum[m][n][i]

                grad_lambda0_vector[n] += 1 / local_sum

                for m in range(self.num_dims_):
                    grad_alpha_matrix[m, n] += exp_kernel_recursive_sum[m][n][i] / local_sum

            grad_alpha_matrix[:, n] -= self.intensity_integral_features_[k][:, n] / self.Beta_[:, n]

            grad_lambda0_vector[n] -= self.observation_window_T_[k]

        gradient /= -self.num_sequences_


    def fit(self, data, options):
        self.Initialize(data)

        self.options_ = options

        opt = Optimizer(self)

        opt.PLBFGS(0, 1e10)

        self.RestoreOptionToDefault()


    def PredictNextEventTime(self, data, num_simulations):
        pass  # Implementation not provided

    def IntensityIntegral(self, lower, upper, data):
        sequences = [data]
        self.InitializeDimension(sequences)

        Lambda0_ = np.array(self.parameters_[:self.num_dims_])
        Alpha_ = np.array(self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_))

        timestamp_per_dimension = self.all_timestamp_per_dimension_[0]

        integral_value = 0

        for n in range(self.num_dims_):
            integral_value += Lambda0_[n] * (upper - lower)

            for m in range(self.num_dims_):
                event_dim_m = np.array(timestamp_per_dimension[m])

                mask = (event_dim_m < lower).astype(float)
                a = (mask * (((-self.Beta_[m, n] * (lower - event_dim_m)) * mask).exp() - ((-self.Beta_[m, n] * (upper - event_dim_m)) * mask).exp())).sum()

                mask = ((event_dim_m >= lower) & (event_dim_m < upper)).astype(float)
                b = (mask * (1 - ((-self.Beta_[m, n] * (upper - event_dim_m)) * mask).exp())).sum()

                integral_value += (Alpha_[m, n] / self.Beta_[m, n]) * (a + b)

        return integral_value

    def RestoreOptionToDefault(self):
        pass  # Implementation not provided

    def AssignDim(self, intensity_dim):
        pass  # Implementation not provided

    def UpdateExpSum(self, t, last_event_per_dim, expsum):
        pass  # Implementation not provided

    def Simulate(self, vec_T, sequences):
        pass  # Implementation not provided

    def Simulate(self, n, num_sequences, sequences):
        pass  # Implementation not provided
