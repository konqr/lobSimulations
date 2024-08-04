import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

def exponential_excitation_log_likelihoodI(X, mu, alpha, beta):
    """
    Computes the log-likelihood of an M-variate Hawkes process using TensorFlow.

    Parameters:
    X (np.array): Event matrix with shape (T, M), where T is the number of time steps and M is the number of dimensions.
    mu (tf.Variable): Base intensity vector with shape (M,)
    alpha (tf.Variable): Excitation matrix with shape (M, M)
    beta (tf.Variable): Decay parameter matrix with shape (M, M)

    Returns:
    float: Negative log-likelihood of the Hawkes process
    """

    M = X.shape[1]
    T = X.shape[0]
    ts = np.arange(0, T)

    # Initialize recursive function
    R = tf.zeros((M, T, M))

    # Compute R recursively
    for m in range(M):
        for n in range(M):
            tks = np.nonzero(X[:, m])[0]
            tis = np.nonzero(X[:, n])[0]

            if len(tks) > 1:
                for t in range(1, len(tks)):
                    tk = tks[t]
                    tkm1 = tks[t - 1]
                    ti = tis[(tis > tkm1) & (tis < tk)]

                    sum_over_tis_given_n = 0
                    for s in range(len(ti)):
                        if s == 0:
                            continue
                        else:
                            sum_over_tis_given_n += tf.exp(-beta[m, n] * (ti[s] - ti[s - 1]))

                    R = tf.tensor_scatter_nd_update(R, [[m, t, n]], [tf.exp(-beta[m, n] * (tk - tkm1)) * R[m, t - 1, n] + sum_over_tis_given_n])

    log_likelihood = 0
    for m in range(M):
        integral_term_over_t = 0
        for n in range(M):
            for t in ts:
                integral_term_over_t += (alpha[m, n] / beta[m, n] * (1 - tf.exp(-beta[m, n] * (T - t))))
        integral_term_over_t = -(mu[m] * T) - integral_term_over_t

        integra_term_over_countp = 0
        for t in ts:
            for n in range(M):
                integra_term_over_countp += tf.math.log(mu[m] + (alpha[m, n] * R[m, t, n]))

        log_likelihood += integral_term_over_t + integra_term_over_countp

    return -log_likelihood  # Return negative log-likelihood for minimization

def exponential_excitation_log_likelihoodII(X, mu, alpha, beta):
    """
    Computes the log-likelihood of an M-variate Hawkes process using TensorFlow.

    Parameters:
    X (np.array): Event matrix with shape (T, M), where T is the number of time steps and M is the number of dimensions.
    mu (tf.Variable): Base intensity vector with shape (M,)
    alpha (tf.Variable): Excitation matrix with shape (M, M)
    beta (tf.Variable): Decay parameter matrix with shape (M, M)

    Returns:
    float: Negative log-likelihood of the Hawkes process
    """

    M = X.shape[1]
    T = X.shape[0]
    ts = np.arange(0, T)

    # Initialize recursive function
    R = tf.zeros((M, T, M))

    # Compute R recursively
    for m in range(M):
        for n in range(M):
            tks = np.nonzero(X[:, m])[0]
            tis = np.nonzero(X[:, n])[0]

            if len(tks) > 1:
                for t in range(1, len(tks)):
                    tk = tks[t]
                    tkm1 = tks[t - 1]
                    ti = tis[(tis > tkm1) & (tis < tk)]

                    if len(ti) > 1:
                        # Compute differences using slicing
                        ti_diff = ti[1:] - ti[:-1]
                        exp_terms = tf.exp(-beta[m, n] * tf.cast(ti_diff, dtype=tf.float32))
                        sum_over_tis_given_n = tf.reduce_sum(exp_terms)

                        R = tf.tensor_scatter_nd_update(R, [[m, t, n]], [tf.exp(-beta[m, n] * tf.cast((tk - tkm1), dtype=tf.float32)) * R[m, t - 1, n] + sum_over_tis_given_n])
                    elif len(ti) == 1:
                        sum_over_tis_given_n = tf.exp(-beta[m, n] * tf.cast((ti[0] - tkm1), dtype=tf.float32))
                        R = tf.tensor_scatter_nd_update(R, [[m, t, n]], [tf.exp(-beta[m, n] * tf.cast((tk - tkm1), dtype=tf.float32)) * R[m, t - 1, n] + sum_over_tis_given_n])

    log_likelihood = 0
    for m in range(M):
        integral_term_over_t = 0
        for n in range(M):
            for t in ts:
                integral_term_over_t += (alpha[m, n] / beta[m, n] * (1 - tf.exp(-beta[m, n] * (T - t))))
        integral_term_over_t = -(mu[m] * T) - integral_term_over_t

        integra_term_over_countp = 0
        for t in ts:
            for n in range(M):
                integra_term_over_countp += tf.math.log(mu[m] + (alpha[m, n] * R[m, t, n]))

        log_likelihood += integral_term_over_t + integra_term_over_countp

    return -log_likelihood  # Return negative log-likelihood for minimization

parser = argparse.ArgumentParser()

parser.add_argument("--n_sims", type=int, default=10, help="Number of simulations")
parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--func_name", type=str, default="expI", help="Function name")

if __name__ == "__main__":

    args = parser.parse_args()

    model_name = 'simulation-hawkes'
    ric = "fake"
    cols = [
    "lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
    "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid"
    ]
    n_sims = args.n_sims
    inputs_path = os.path.join(os.path.dirname(__file__), 'data', 'inputs')
    outputs_path = os.path.join(os.path.dirname(__file__), 'data', 'outputs')

    # read data
    dataPath = os.path.join(outputs_path, model_name)
    dictBinnedData = {}
    for d in range(n_sims):
        dictBinnedData[d] = []
    dates = list(dictBinnedData.keys())

    dfs = []
    for i in dates:
        try:
            read_path = os.path.join(dataPath, ric + "_" + str(i) + "_12D.csv")
            df = pd.read_csv(read_path)

            # drop columns
            df.drop(['Unnamed: 0'], axis=1, inplace=True)

            # change columns order
            df = df[["Date", "Time", "event"] + df.drop(["Date", "Time", "event"], axis=1).columns.to_list()]

            dfs.append(df)
        except:
            print(f"No data for {ric} on {i}")
    df = pd.concat(dfs)

    pivot_df = df.copy()
    pivot_df["count"] = 1
    pivot_df = pivot_df.pivot_table(index=["Date", "Time"], columns="event", values="count").fillna(0)
    X = pivot_df.values

    # Model initialization parameters
    M = X.shape[1]

    # Init Model estimation parameters
    mu = tf.Variable(np.random.uniform(0.1, 1.0, M), dtype=tf.float32)
    alpha = tf.Variable(np.random.uniform(0.01, 0.5, (M, M)), dtype=tf.float32)
    beta = tf.Variable(np.random.uniform(0.5, 2.0, (M, M)), dtype=tf.float32)

    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            if args.func_name == "expI":
                loss = exponential_excitation_log_likelihoodI(X, mu, alpha, beta)
            elif args.func_name == "expII":
                loss = exponential_excitation_log_likelihoodII(X, mu, alpha, beta)
            else:
                raise ValueError(f"Invalid function name {args.func_name}")
        gradients = tape.gradient(loss, [mu, alpha, beta])
        optimizer.apply_gradients(zip(gradients, [mu, alpha, beta]))
        return loss

    # Perform optimization
    print("--------------------")
    print(f"Optimizing {args.func_name} function")
    for epoch in range(args.n_epochs):
        loss = train_step()
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    print("--------------------")
    print(f"Optimized mu: {mu.numpy()}")
    print(f"Optimized alpha: {alpha.numpy()}")
    print(f"Optimized beta: {beta.numpy()}")