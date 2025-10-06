import torch
import numpy as np
import DGMTorch_old as DGM
from utils import TrainingLogger, ModelManager, get_gpu_specs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
import pandas as pd

def gumbel_softmax_sample(logits, tau=1.0, hard=True):
    """Gumbel-Softmax with straight-through trick."""
    # gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    # y = (logits + gumbel_noise) / tau
    y_soft = F.softmax(logits, dim=-1)

    if hard:
        # Forward: one-hot, Backward: soft
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


def gumbel_softmax_entropy(u_logits, tau=1.0, n_samples=1):
    """
    Estimate entropy using Gumbel-Softmax relaxation.

    Args:
        u_logits: [batch_size, n_actions] raw logits
        tau: temperature > 0 (higher = more uniform, lower = sharper)
        n_samples: number of Gumbel draws for Monte Carlo estimate

    Returns:
        scalar entropy estimate
    """
    entropies = []
    for _ in range(n_samples):
        # Sample relaxed categorical distribution
        probs = F.gumbel_softmax(u_logits, tau=tau, hard=False)
        log_probs = torch.log(probs + 1e-12)
        entropy = -(probs * log_probs).sum(dim=1)   # per sample entropy
        entropies.append(entropy)

    return torch.stack(entropies, dim=0).mean()


class MarketMaking():
    def __init__(self, num_points=100, num_epochs=1000, ric='AAPL', hawkes = False):
        '''
        state variables:
            X: Cash
            Y: Inventory,
            p_a, p_b: best prices,
            q_a, q_b: best quotes,
            qD_a, qD_b: 2nd best quotes,
            n_a, n_b: queue priority,
            P_mid: mid-price
        '''
        self.device = None
        self.SAMPLER = 'sim'
        self.TERMINATION_TIME = 300
        self.NDIMS = 12
        self.NUM_POINTS = num_points
        self.EPOCHS = num_epochs
        self.eta = 10  # inventory penalty
        self.E = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
                  "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
        self.U = ["lo_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
                  "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "lo_deep_Bid"]
        if hawkes: self.U = self.E.copy()
        self.hawkes = hawkes
        # self.lambdas_poisson = [.86, .32, .33, .48, .02, .47, .47, .02, .48, .33, .32, .86]  # [5] * 12 # AMZN
        # INTC : {'lo_deep_Ask': 1.3291742343304844, 'co_deep_Ask': 2.2448482015669513, 'lo_top_Ask': 7.89707621082621, 'co_top_Ask': 6.617852118945869, 'mo_Ask': 0.5408440170940172, 'lo_inspread_Ask': 0.1327911324786325, 'lo_inspread_Bid': 0.1327911324786325, 'mo_Bid': 0.5408440170940172, 'co_top_Bid': 6.617852118945869, 'lo_top_Bid': 7.89707621082621, 'co_deep_Bid': 2.2448482015669513, 'lo_deep_Bid': 1.3291742343304844}
        # AAPL : {'lo_deep_Ask': 1.9115059993425376, 'co_deep_Ask': 2.3503168145956606, 'lo_top_Ask': 4.392785174227481, 'co_top_Ask': 4.248318129520053, 'mo_Ask': 1.0989673734385272, 'lo_inspread_Ask': 1.0131743096646941, 'lo_inspread_Bid': 1.0131743096646941, 'mo_Bid': 1.0989673734385272, 'co_top_Bid': 4.248318129520053, 'lo_top_Bid': 4.392785174227481, 'co_deep_Bid': 2.3503168145956606, 'lo_deep_Bid': 1.9115059993425376}
        self.lambdas_poisson = [1.33, 2.24, 7.90, 6.62, 0.54, 0.13, 0.13, 0.54, 6.62, 7.90, 2.24, 1.33]
        with open("D:\\PhD\\calibrated params\\INTC.OQ_ParamsInferredWCutoffEyeMu_moOnly_new_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
            kernelparams = pickle.load(f)
        mus = torch.tensor([kernelparams[i] for i in self.E], dtype=torch.float32)
        self.mus = (torch.ones((len(self.E), len(self.E)), dtype=torch.float32)*mus/len(self.E)).T
        gammas = torch.zeros((len(self.E), len(self.E)), dtype=torch.float32)
        alphas = torch.zeros((len(self.E), len(self.E)), dtype=torch.float32)
        for i, e_i in enumerate(self.E):
            for j, e_j in enumerate(self.E):
                if kernelparams.get(e_j+'->'+e_i,None) is not None:
                    gammas[i, j] = kernelparams[e_j+'->'+e_i][1][1]*kernelparams[e_j+'->'+e_i][1][2]
                    alphas[i,j] = kernelparams[e_j+'->'+e_i][0]*kernelparams[e_j+'->'+e_i][1][0]
        self.gammas = gammas
        self.alphas = alphas
        if ric == 'AAPL':
            self.lambdas_poisson = [1.91, 2.35, 4.39, 4.23, 1.09, 1.01, 1.01, 1.09, 4.23, 4.39, 2.35, 1.91]

    def ansatz_output(self, states, ts, model_phi):
        states_lob = self.means + states[:,:11].clone()*(self.stds + 1e-8)
        return (states_lob[:,0] + states_lob[:,1]*(states_lob[:,10] - 1e-2*states_lob[:,1])).unsqueeze(-1) + (1 - ts)*model_phi(ts, states)*1000

    def sampler(self, num_points=1000, seed=None, boundary=False, hawkes=False):
        '''
        Sample points from the stationary distributions for the DGM learning
        :param num_points: number of points
        :return: samples of [0,T] x {state space}
        '''
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Generate sample data

        Ys = np.round(6 * np.random.randn(num_points, 1), 0)
        Xs = 2500 - 100*Ys + np.round(0.1*np.random.randn(num_points, 1), 2)
        P_mids = np.round(200 + 10 * np.random.randn(num_points, 1), 2) / 2
        spreads = 0.01 * np.random.geometric(.8, [num_points, 1])
        p_as = np.round(P_mids + spreads / 2, 2)
        p_bs = p_as - spreads
        P_mids = (p_as + p_bs)/2.
        p = 0.1
        p2 = p/2
        q_as = np.random.geometric(p, [num_points, 1])
        qD_as = np.random.geometric(p2, [num_points, 1])
        q_bs = np.random.geometric(p, [num_points, 1])
        qD_bs = np.random.geometric(p2, [num_points, 1])
        n_as = np.array([np.random.randint(0, b) for b in q_as + qD_as + 1])/q_as
        n_bs = np.array([np.random.randint(0, b) for b in q_bs + qD_bs + 1])/q_bs
        if hawkes:
            frozen_mask = (self.alphas != 0)
            lambdas = torch.log((torch.tensor(np.random.pareto(3.0, [num_points,len(self.E), len(self.E)]), dtype=torch.float32)*frozen_mask+ 1)*self.mus)

        t = np.random.uniform(0, self.TERMINATION_TIME, [num_points, 1]) / self.TERMINATION_TIME
        t_boundary = np.ones([num_points, 1])

        # Stack features
        features = np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids])

        # --- Standardize using known distributions ---
        eps = 1e-8
        means = np.array([
            2500.0,
            0.0,   # Y ~ N(0, 2^2)
            100.0, # ask price ~ Pmid + spread/2
            100.0, # bid price ~ Pmid - spread/2
            (1-p)/p,   # mean of Geom(0.002)
            (1-p)/p,   # mean of Geom(0.002)
            (1-p2)/p2, # mean of Geom(0.0015)
            (1-p2)/p2, # mean of Geom(0.0015)
            0,  # n_as ~ U(0, q_as) (data dependent, center later)
            0,  # n_bs ~ U(0, q_bs)
            100.0  # Pmid
        ])

        stds = np.array([
            6.0,
            6.0,
            5.0,
            5.0,
            np.sqrt((1-p)/(p**2)),
            np.sqrt((1-p)/(p**2)),
            np.sqrt((1-p2)/(p2**2)),
            np.sqrt((1-p2)/(p2**2)),
            1,
            1,
            5.0
        ])

        # Fill n_as, n_bs normalization from sample (since uniform depends on q)
        means[8] = n_as.mean()
        means[9] = n_bs.mean()
        stds[8] = n_as.std() + eps
        stds[9] = n_bs.std() + eps

        features = (features - means) / (stds + eps)
        features = features.astype('float')
        self.means = torch.tensor(means.astype('float'), dtype=torch.float32)
        self.stds = torch.tensor(stds.astype('float'), dtype=torch.float32)
        if hawkes:
            features_lamb = lambdas #(lambdas - mean_lambda.numpy())/(std_lambda.numpy() + 1e-8)
            self.means_lamb = torch.mean(torch.log(torch.exp(lambdas).sum(dim=2)), axis=0)
            self.std_lamb = torch.std(torch.log(torch.exp(lambdas).sum(dim=2)), axis=0)
            if boundary:
                return torch.tensor(t_boundary, dtype=torch.float32), torch.tensor(features, dtype=torch.float32), torch.tensor(features_lamb, dtype=torch.float32)

            return torch.tensor(t, dtype=torch.float32), torch.tensor(features, dtype=torch.float32), torch.tensor(features_lamb, dtype=torch.float32)
        if boundary:
            return torch.tensor(t_boundary, dtype=torch.float32), torch.tensor(features, dtype=torch.float32)

        return torch.tensor(t, dtype=torch.float32), torch.tensor(features, dtype=torch.float32)


    def sampler_sim(self, ric='AAPL', num_points=1000, seed=None, boundary=False):
        path = '/SAN/fca/Konark_PhD_Experiments/simulated/poisson'
        files = os.listdir(path)
        files = [f for f in files if ('poisson' in f)&(ric in f)]
        files = np.random.choice(files, 10)
        Xs = np.round(1e3 * np.random.randn(num_points, 1), 2)
        Ys = np.round(2 * np.random.randn(num_points, 1), 0)
        P_mids = []
        p_as = []
        p_bs = []
        q_as = []
        qD_as = []
        q_bs = []
        qD_bs = []
        t = []
        t_boundary = self.TERMINATION_TIME * np.ones([num_points, 1])
        for fname in files:
            with open(path+'/'+fname, "rb") as f:
                results = pickle.load(f)
            df = pd.DataFrame(results[1])
            idxs = np.random.randint(0, len(df), num_points//10)
            data = df.iloc[idxs]
            p_a = data['Ask_touch'].apply(lambda x: x[0]).values
            q_a = data['Ask_touch'].apply(lambda x: x[1]).values
            p_b = data['Bid_touch'].apply(lambda x: x[0]).values
            q_b = data['Bid_touch'].apply(lambda x: x[1]).values
            qD_a = data['Ask_deep'].apply(lambda x: x[1]).values
            qD_b = data['Bid_deep'].apply(lambda x: x[1]).values
            P_mid = (p_a + p_b)/2
            p_as += [p_a]
            q_as += [q_a]
            p_bs += [p_b]
            q_bs += [q_b]
            qD_as += [qD_a]
            qD_bs += [qD_b]
            P_mids += [P_mid]
            t += [pd.DataFrame(results[0][1:]).iloc[idxs-1][1].values]
        t = np.hstack(t).reshape(-1,1)
        P_mids = np.hstack(P_mids).reshape(-1,1)
        p_as = np.hstack(p_as).reshape(-1,1)
        p_bs = np.hstack(p_bs).reshape(-1,1)
        q_as = np.hstack(q_as).reshape(-1,1)
        qD_as = np.hstack(qD_as).reshape(-1,1)
        q_bs = np.hstack(q_bs).reshape(-1,1)
        qD_bs = np.hstack(qD_bs).reshape(-1,1)
        #TODO: maybe better to use exp distri here?
        n_as = np.array([np.random.randint(0, b) for b in q_as + qD_as])
        n_bs = np.array([np.random.randint(0, b) for b in q_bs + qD_bs])
        # Convert to TensorFlow tensors immediately
        if boundary:
            return torch.tensor(t_boundary, dtype=torch.float32), torch.tensor(
                np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]), dtype=torch.float32)

        return torch.tensor(t, dtype=torch.float32), torch.tensor(
            np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]), dtype=torch.float32)

    def sample_qd(self):
        # Return a PyTorch tensor directly
        return torch.tensor(np.random.geometric(.0015, 1), dtype=torch.float32)

    def tr_lo_deep(self, qD):
        # PyTorch operation
        return qD + 1.0

    def tr_co_deep(self, qD, n, q):
        # PyTorch operations
        qD_updated = qD - 1.0

        # Replace conditional operations with smooth PyTorch operations
        qD_updated = torch.clamp(qD_updated, min=0.0)  # Ensure qD doesn't go negative

        # Handle the agent's orders case
        condition = (qD_updated == 1.0) & (n == q)
        qD_updated = torch.where(condition, qD_updated + 1.0, qD_updated)

        return qD_updated

    def tr_lo_top(self, q_as, n_as):
        # PyTorch operations
        q_as_updated = q_as + 1.0

        # Adjust priority for deep orders
        condition = n_as >= q_as_updated
        n_as_updated = torch.where(condition, n_as + 1.0, n_as)

        return q_as_updated, n_as_updated

    def tr_co_top(self, z, q_as, n_as, qD_as, p_as, P_mids, intervention=False):
        if intervention:
            # For intervention, directly update values
            q_as_updated = q_as - 1.0
            # This is a simplification; in real code, you might need a differentiable alternative
            n_as_updated = torch.rand(n_as.shape, device=n_as.device) * (q_as_updated + qD_as + 1)
            qD_as_updated =qD_as.clone()
        else:
            # For normal operation
            idxCO = torch.rand(q_as.shape, device=q_as.device) * q_as
            q_as_updated = q_as - 1.0

            # Handle cases where we are cancelling agent's orders
            condition = idxCO == n_as
            qD_as_updated = torch.where(condition, qD_as + 1.0, qD_as)

            # Handle cases where n_as > idxCO
            condition = n_as > idxCO
            n_as_updated = torch.where(condition, n_as - 1.0, n_as)

        # Handle queue depletion
        condition = q_as_updated == 0.0
        q_as_final = torch.where(condition, qD_as_updated, q_as_updated)
        qD_as_final = torch.where(condition, self.sample_qd(), qD_as_updated)
        p_as_final = torch.where(condition, p_as + z * 0.01, p_as)
        P_mids_final = torch.where(condition, P_mids + z * 0.005, P_mids)

        return q_as_final, n_as_updated, qD_as_final, p_as_final, P_mids_final

    def tr_mo(self, z, q_as, n_as, qD_as, p_as, P_mids, Xs, Ys, intervention=False):
        # Agent fill condition
        agent_fill_condition = n_as == 0.0

        if not intervention:
            # Update X and Y when agent's order is filled
            Xs_updated = torch.where(agent_fill_condition, Xs + z * p_as, Xs)
            Ys_updated = torch.where(agent_fill_condition, Ys - z * 1.0, Ys)

            # Update agent position
            # For a differentiable version, we could use a smoothed random update
            n_as_random = torch.rand(n_as.shape, device=n_as.device) * (q_as + qD_as - 1.0) + 1.0
            n_as_updated = torch.where(agent_fill_condition, n_as_random, n_as)
        else:
            # No effect during intervention for agent fills
            q_as_intervention = torch.where(agent_fill_condition, q_as + 1.0, q_as)
            n_as_intervention = torch.where(agent_fill_condition, n_as + 1.0, n_as)
            q_as = q_as_intervention
            n_as = n_as_intervention
            Xs_updated = Xs
            Ys_updated = Ys

        # Standard updates
        q_as_updated = q_as - 1.0
        n_as_final = n_as - 1.0

        # Handle queue depletion
        depletion_condition = q_as_updated == 0.0
        q_as_final = torch.where(depletion_condition, qD_as, q_as_updated)
        qD_as_final = torch.where(depletion_condition, self.sample_qd(), qD_as)
        p_as_final = torch.where(depletion_condition, p_as + z * 0.01, p_as)
        P_mids_final = torch.where(depletion_condition, P_mids + z * 0.005, P_mids)

        if intervention:
            # Update X and Y for intervention
            Xs_updated = Xs + z * p_as
            Ys_updated = Ys - z * 1.0

        return q_as_final, n_as_final, qD_as_final, p_as_final, P_mids_final, Xs_updated, Ys_updated

    def tr_is(self, z, q_as, qD_as, n_as, P_mids, p_as, intervention=False):
        # PyTorch operations
        qD_as_updated = q_as  # Copy q_as to qD_as
        q_as_updated = torch.ones_like(q_as)  # Set q_as to ones

        if intervention:
            n_as_updated = torch.zeros_like(n_as)
        else:
            n_as_updated = n_as + 1.0

        P_mids_updated = P_mids - z * 0.005
        p_as_updated = p_as - z * 0.01

        return q_as_updated, qD_as_updated, n_as_updated, P_mids_updated, p_as_updated

    def transition(self, Ss, eventID):
        Ss = self.means + Ss.clone()*(self.stds + 1e-8)
        # Unpack state variables
        Xs = Ss[:, 0]
        Ys = Ss[:, 1]
        p_as = Ss[:, 2]
        p_bs = Ss[:, 3]
        q_as = Ss[:, 4]
        q_bs = Ss[:, 5]
        qD_as = Ss[:, 6]
        qD_bs = Ss[:, 7]
        n_as = Ss[:, 8]
        n_bs = Ss[:, 9]
        P_mids = Ss[:, 10]

        # Initialize output tensor - same shape as input
        batch_size = Ss.shape[0]
        Ss_out = Ss.clone()

        # Event 0: lo_deep_Ask
        mask = (eventID == 0)
        if mask.any():
            qD_as_updated = self.tr_lo_deep(qD_as)
            Ss_out[:, 6] = qD_as_updated

        # Event 1: co_deep_Ask
        mask = (eventID == 1)
        if mask.any():
            qD_as_updated = self.tr_co_deep(qD_as, n_as, q_as)
            Ss_out[:, 6] = qD_as_updated

        # Event 2: lo_top_Ask
        mask = (eventID == 2)
        if mask.any():
            q_as_updated, n_as_updated = self.tr_lo_top(q_as, n_as)
            Ss_out[:, 4] = q_as_updated
            Ss_out[:, 8] = n_as_updated

        # Event 3: co_top_Ask
        mask = (eventID == 3)
        if mask.any():
            q_as_updated, n_as_updated, qD_as_updated, p_as_updated, P_mids_updated = self.tr_co_top(
                1.0, q_as, n_as, qD_as, p_as, P_mids)
            Ss_out[:, 2] = p_as_updated
            Ss_out[:, 4] = q_as_updated
            Ss_out[:, 6] = qD_as_updated
            Ss_out[:, 8] = n_as_updated
            Ss_out[:, 10] = P_mids_updated

        # Event 4: mo_Ask
        mask = (eventID == 4)
        if mask.any():
            q_as_updated, n_as_updated, qD_as_updated, p_as_updated, P_mids_updated, Xs_updated, Ys_updated = self.tr_mo(
                1.0, q_as, n_as, qD_as, p_as, P_mids, Xs, Ys)
            Ss_out[:, 0] = Xs_updated
            Ss_out[:, 1] = Ys_updated
            Ss_out[:, 2] = p_as_updated
            Ss_out[:, 4] = q_as_updated
            Ss_out[:, 6] = qD_as_updated
            Ss_out[:, 8] = n_as_updated
            Ss_out[:, 10] = P_mids_updated
        spreads = p_as - p_bs
        mask_spread = (spreads >= 0.015).flatten()
        # Event 5: lo_inspread_Ask
        mask = (eventID == 5)
        if mask.any():
            q_as_updated, qD_as_updated, n_as_updated, P_mids_updated, p_as_updated = self.tr_is(
                1.0, q_as[mask_spread], qD_as[mask_spread], n_as[mask_spread], P_mids[mask_spread], p_as[mask_spread])
            Ss_out[mask_spread, 2] = p_as_updated
            Ss_out[mask_spread, 4] = q_as_updated
            Ss_out[mask_spread, 6] = qD_as_updated
            Ss_out[mask_spread, 8] = n_as_updated
            Ss_out[mask_spread, 10] = P_mids_updated

        # Event 6: lo_inspread_Bid
        mask = (eventID == 6)
        if mask.any():
            q_bs_updated, qD_bs_updated, n_bs_updated, P_mids_updated, p_bs_updated = self.tr_is(
                -1.0, q_bs[mask_spread], qD_bs[mask_spread], n_bs[mask_spread], P_mids[mask_spread], p_bs[mask_spread])
            Ss_out[mask_spread, 3] = p_bs_updated
            Ss_out[mask_spread, 5] = q_bs_updated
            Ss_out[mask_spread, 7] = qD_bs_updated
            Ss_out[mask_spread, 9] = n_bs_updated
            Ss_out[mask_spread, 10] = P_mids_updated

        # Event 7: mo_Bid
        mask = (eventID == 7)
        if mask.any():
            q_bs_updated, n_bs_updated, qD_bs_updated, p_bs_updated, P_mids_updated, Xs_updated, Ys_updated = self.tr_mo(
                -1.0, q_bs, n_bs, qD_bs, p_bs, P_mids, Xs, Ys)
            Ss_out[:, 0] = Xs_updated
            Ss_out[:, 1] = Ys_updated
            Ss_out[:, 3] = p_bs_updated
            Ss_out[:, 5] = q_bs_updated
            Ss_out[:, 7] = qD_bs_updated
            Ss_out[:, 9] = n_bs_updated
            Ss_out[:, 10] = P_mids_updated

        # Event 8: co_top_Bid
        mask = (eventID == 8)
        if mask.any():
            q_bs_updated, n_bs_updated, qD_bs_updated, p_bs_updated, P_mids_updated = self.tr_co_top(
                -1.0, q_bs, n_bs, qD_bs, p_bs, P_mids)
            Ss_out[:, 3] = p_bs_updated
            Ss_out[:, 5] = q_bs_updated
            Ss_out[:, 7] = qD_bs_updated
            Ss_out[:, 9] = n_bs_updated
            Ss_out[:, 10] = P_mids_updated

        # Event 9: lo_top_Bid
        mask = (eventID == 9)
        if mask.any():
            q_bs_updated, n_bs_updated = self.tr_lo_top(q_bs, n_bs)
            Ss_out[:, 5] = q_bs_updated
            Ss_out[:, 9] = n_bs_updated

        # Event 10: co_deep_Bid
        mask = (eventID == 10)
        if mask.any():
            qD_bs_updated = self.tr_co_deep(qD_bs, n_bs, q_bs)
            Ss_out[:, 7] = qD_bs_updated

        # Event 11: lo_deep_Bid
        mask = (eventID == 11)
        if mask.any():
            qD_bs_updated = self.tr_lo_deep(qD_bs)
            Ss_out[:, 7] = qD_bs_updated
        Ss_out = (Ss_out - self.means)/(self.stds+1e-8)
        return Ss_out

    def intervention(self, ts, Ss, logits, cat=False):
        """
        Differentiable intervention using Gumbel-Softmax ST trick.

        Args:
            ts: time steps [batch_size]
            Ss: state tensor [batch_size, state_dim]
            logits: NN outputs [batch_size, num_actions]
        """
        Ss = self.means + Ss.clone()*(self.stds + 1e-8)
        batch_size = ts.shape[0]
        device = ts.device

        # ----- Gumbel-Softmax with Straight-Through -----
        if cat:
            us_onehot = logits
        else:
            us_onehot = F.gumbel_softmax(logits, tau=1.0, hard=True)  # [batch_size, num_actions]

        # Unpack state variables
        Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids = [
            Ss[:, i] for i in range(Ss.shape[1])
        ]

        # Initialize updated states
        Xs_updated     = Xs.clone()
        Ys_updated     = Ys.clone()
        p_as_updated   = p_as.clone()
        p_bs_updated   = p_bs.clone()
        q_as_updated   = q_as.clone()
        q_bs_updated   = q_bs.clone()
        qD_as_updated  = qD_as.clone()
        qD_bs_updated  = qD_bs.clone()
        n_as_updated   = n_as.clone()
        n_bs_updated   = n_bs.clone()
        P_mids_updated = P_mids.clone()

        inter_profit = torch.zeros(batch_size, device=device)

        # ---------- ACTIONS ----------

        # 0: Limit order deep asks
        mask = us_onehot[:, 0]
        qD_as_updated = mask * (qD_as + 1.0) + (1 - mask) * qD_as_updated
        # 1: CO Deep ask
        mask = us_onehot[:,1]
        qD_as_new  = self.tr_co_deep(qD_as, n_as, q_as)
        qD_as_updated = mask * (qD_as_new) + (1 - mask) * qD_as_updated
        # 1: Limit order top asks
        mask = us_onehot[:, 2]
        q_as_new = q_as + 1.0
        n_as_new = torch.where(n_as >= q_as_new, q_as_new, n_as)
        q_as_updated = mask * q_as_new + (1 - mask) * q_as_updated
        n_as_updated = mask * n_as_new + (1 - mask) * n_as_updated

        # 2: Cancel order top asks
        mask = us_onehot[:, 3]
        co_q_as, co_n_as, co_qD_as, co_p_as, co_P_mids = self.tr_co_top(
            1.0, q_as, n_as, qD_as, p_as, P_mids, intervention=True
        )
        q_as_updated   = mask * co_q_as   + (1 - mask) * q_as_updated
        n_as_updated   = mask * co_n_as   + (1 - mask) * n_as_updated
        qD_as_updated  = mask * co_qD_as  + (1 - mask) * qD_as_updated
        p_as_updated   = mask * co_p_as   + (1 - mask) * p_as_updated
        P_mids_updated = mask * co_P_mids + (1 - mask) * P_mids_updated

        # 3: Market order asks
        mask = us_onehot[:, 4]
        mo_q_as, mo_n_as, mo_qD_as, mo_p_as, mo_P_mids, mo_Xs, mo_Ys = self.tr_mo(
            1.0, q_as, n_as, qD_as, p_as, P_mids, Xs, Ys, intervention=True
        )
        q_as_updated   = mask * mo_q_as   + (1 - mask) * q_as_updated
        n_as_updated   = mask * mo_n_as   + (1 - mask) * n_as_updated
        qD_as_updated  = mask * mo_qD_as  + (1 - mask) * qD_as_updated
        p_as_updated   = mask * mo_p_as   + (1 - mask) * p_as_updated
        P_mids_updated = mask * mo_P_mids + (1 - mask) * P_mids_updated
        Xs_updated     = mask * mo_Xs     + (1 - mask) * Xs_updated
        Ys_updated     = mask * mo_Ys     + (1 - mask) * Ys_updated
        inter_profit   = inter_profit + mask * mo_p_as

        # 4: Limit order in-spread asks
        mask = us_onehot[:, 5]
        spreads = p_as - p_bs
        valid = (spreads >= 0.015).float()
        mask = mask * valid  # ensure valid only when spread is large
        lo_q_as, lo_qD_as, lo_n_as, lo_P_mids, lo_p_as = self.tr_is(
            1.0, q_as, qD_as, n_as, P_mids, p_as, intervention=True
        )
        q_as_updated   = mask * lo_q_as   + (1 - mask) * q_as_updated
        qD_as_updated  = mask * lo_qD_as  + (1 - mask) * qD_as_updated
        n_as_updated   = mask * lo_n_as   + (1 - mask) * n_as_updated
        P_mids_updated = mask * lo_P_mids + (1 - mask) * P_mids_updated
        p_as_updated   = mask * lo_p_as   + (1 - mask) * p_as_updated

        # 5: Limit order in-spread bids
        mask = us_onehot[:, 6]
        mask = mask * valid
        lo_q_bs, lo_qD_bs, lo_n_bs, lo_P_mids, lo_p_bs = self.tr_is(
            -1.0, q_bs, qD_bs, n_bs, P_mids, p_bs, intervention=True
        )
        q_bs_updated   = mask * lo_q_bs   + (1 - mask) * q_bs_updated
        qD_bs_updated  = mask * lo_qD_bs  + (1 - mask) * qD_bs_updated
        n_bs_updated   = mask * lo_n_bs   + (1 - mask) * n_bs_updated
        P_mids_updated = mask * lo_P_mids + (1 - mask) * P_mids_updated
        p_bs_updated   = mask * lo_p_bs   + (1 - mask) * p_bs_updated

        # 6: Market order bids
        mask = us_onehot[:, 7]
        mo_q_bs, mo_n_bs, mo_qD_bs, mo_p_bs, mo_P_mids, mo_Xs, mo_Ys = self.tr_mo(
            -1.0, q_bs, n_bs, qD_bs, p_bs, P_mids, Xs, Ys, intervention=True
        )
        q_bs_updated   = mask * mo_q_bs   + (1 - mask) * q_bs_updated
        n_bs_updated   = mask * mo_n_bs   + (1 - mask) * n_bs_updated
        qD_bs_updated  = mask * mo_qD_bs  + (1 - mask) * qD_bs_updated
        p_bs_updated   = mask * mo_p_bs   + (1 - mask) * p_bs_updated
        P_mids_updated = mask * mo_P_mids + (1 - mask) * P_mids_updated
        Xs_updated     = mask * mo_Xs     + (1 - mask) * Xs_updated
        Ys_updated     = mask * mo_Ys     + (1 - mask) * Ys_updated
        inter_profit   = inter_profit - mask * mo_p_bs

        # 7: Cancel order top bids
        mask = us_onehot[:, 8]
        co_q_bs, co_n_bs, co_qD_bs, co_p_bs, co_P_mids = self.tr_co_top(
            -1.0, q_bs, n_bs, qD_bs, p_bs, P_mids, intervention=True
        )
        q_bs_updated   = mask * co_q_bs   + (1 - mask) * q_bs_updated
        n_bs_updated   = mask * co_n_bs   + (1 - mask) * n_bs_updated
        qD_bs_updated  = mask * co_qD_bs  + (1 - mask) * qD_bs_updated
        p_bs_updated   = mask * co_p_bs   + (1 - mask) * p_bs_updated
        P_mids_updated = mask * co_P_mids + (1 - mask) * P_mids_updated

        # 8: Limit order top bids
        mask = us_onehot[:, 9]
        q_bs_new = q_bs + 1.0
        n_bs_new = torch.where(n_bs >= q_bs_new, q_bs_new, n_bs)
        q_bs_updated = mask * q_bs_new + (1 - mask) * q_bs_updated
        n_bs_updated = mask * n_bs_new + (1 - mask) * n_bs_updated

        # 9: Limit order deep bids
        mask = us_onehot[:, 10]
        qD_bs_updated = mask * (qD_bs + 1.0) + (1 - mask) * qD_bs_updated

        # ---------- Build final state ----------
        Ss_intervened = torch.stack([
            Xs_updated, Ys_updated, p_as_updated, p_bs_updated,
            q_as_updated, q_bs_updated, qD_as_updated, qD_bs_updated,
            n_as_updated, n_bs_updated, P_mids_updated
        ], dim=1)
        Ss_intervened = (Ss_intervened - self.means)/(self.stds+1e-8)
        return Ss_intervened  # + maybe inter_profit if needed

    def transition_lambda(self, lambdas, i):
        # lambdas = self.means_lamb + lambdas.clone()*(self.std_lamb + 1e-8)
        lambdas = torch.exp(lambdas)
        # us_onehot = torch.tensor(F.gumbel_softmax(logits, tau=1e-9, hard=True), dtype=torch.float32)
        # out = torch.einsum('bi,jk->bijk', us_onehot, self.alphas)  # [100,12,12,12]
        # out = out.sum(dim=1)
        out = torch.zeros_like(lambdas)
        if type(i) == int:
            out[:,:,i] = self.alphas[:,i]
        else:
            us_onehot = torch.tensor(F.gumbel_softmax(i, tau=1, hard=True), dtype=torch.float32)
            out = us_onehot.unsqueeze(1)*self.alphas
        lambdas = torch.clamp(lambdas + out, min=1e-6)
        return torch.log(lambdas) #(lambdas - self.means_lamb)/(self.std_lamb + 1e-8)

    def oracle_u(self, model_phi, ts, Ss):
        # Use PyTorch operations to compute the best action
        batch_size = ts.shape[0]

        # Initialize tensor to store intervention values for each action
        interventions = torch.zeros((batch_size, len(self.U)), dtype=torch.float32, device=self.device)

        # Calculate value for each action
        for u in range(len(self.U)):
            # Create a tensor of action u for all batch items
            u_tensor = torch.ones((batch_size, 1), dtype=torch.long) * u
            u_tensor = F.one_hot(u_tensor, len(self.U)).squeeze(1)
            # Calculate intervention value
            intervention_value = model_phi(ts, self.intervention(ts, Ss, u_tensor, cat = True))

            # Store in interventions tensor
            indices = torch.stack([torch.arange(batch_size, dtype=torch.int64),
                                   torch.ones(batch_size, dtype=torch.int64) * u], dim=1)
            interventions[indices[:, 0], indices[:, 1]] = intervention_value.reshape(-1).to(self.device)

        # Return the action with highest value
        return torch.argmax(interventions, dim=1).to(self.device)

    def sup_intervention(self, model_phi, ts, Ss):
        # Use PyTorch operations to compute the best action
        batch_size = ts.shape[0]

        # Initialize tensor to store intervention values for each action
        interventions = torch.zeros((batch_size, len(self.U)), dtype=torch.float32, device=self.device)

        # Calculate value for each action
        for u in range(len(self.U)):
            # Create a tensor of action u for all batch items
            u_tensor = torch.ones((batch_size, 1), dtype=torch.float32) * u

            # Calculate intervention value
            intervention_value = model_phi(ts, self.intervention(ts, Ss, u_tensor))

            # Store in interventions tensor
            indices = torch.stack([torch.arange(batch_size, dtype=torch.int64),
                                   torch.ones(batch_size, dtype=torch.int64) * u], dim=1)
            interventions[indices[:, 0], indices[:, 1]] = intervention_value.reshape(-1).to(self.device)

        # Return the highest value
        return torch.max(interventions, 1)[0].reshape((batch_size, 1)).to(self.device)


    def oracle_d(self, model_phi, model_u, ts, Ss):
        batch_size = ts.shape[0]

        # Get the control from model_u
        us, logits_u = model_u(ts, Ss)

        # Intervention values
        with torch.no_grad():
            # d = 0: do nothing
            phi_no_intervention = model_phi(ts, Ss)  # [batch_size, 1]
            # d = 1: apply control
            phi_with_intervention = model_phi(ts, self.intervention(ts, Ss, logits_u))  # [batch_size, 1]

            # Stack along decision dimension: 0 = no intervention, 1 = intervene
            phi_stack = torch.cat([phi_no_intervention, phi_with_intervention], dim=1)  # [batch_size, 2]

            # Optimal d = argmax intervention value
            optimal_d = torch.argmax(phi_stack, dim=1)

        return optimal_d.to(self.device)


    def loss_phi_poisson(self, model_phi, model_d, model_u,
                         ts, Ss, Ts, S_boundarys, lambdas,
                         train_phi=False, train_d=False, train_u=False, tau=0.5):

        # Autograd for time derivative
        ts.requires_grad_(True)
        output = model_phi(ts, Ss)
        output = output.to(self.device)

        phi_t = torch.autograd.grad(output.sum(), ts, create_graph=True)[0].to(self.device)

        # Integral term
        I_phi = torch.zeros_like(output).to(self.device)
        for i in range(self.NDIMS):
            Ss_transitioned = self.transition(Ss, torch.tensor(i, dtype=torch.float32, device=self.device))
            I_phi += lambdas[i] * (model_phi(ts, Ss_transitioned) - output)

        # Generator term
        L_phi = phi_t + I_phi

        # Running cost
        f = -self.eta * torch.square(Ss[:, 1])
        f = f.reshape(-1, 1).to(self.device)

        # Decision network
        _, logits_d = model_d(ts, Ss)
        if train_d:
            ds_onehot = gumbel_softmax_sample(logits_d, tau=tau, hard=True)
            ds = ds_onehot[:, 1:2]   # assuming binary choice: intervene vs not
        else:
            ds = torch.argmax(logits_d, dim=-1).reshape(-1, 1).float().detach()

        # Control network
        _, logits_u = model_u(ts, Ss)
        if not train_u:
            logits_u = logits_u.detach()

        # Intervention value
        M_phi = model_phi(ts, self.intervention(ts, Ss, logits_u))

        # HJB evaluation
        evaluation = (1 - ds) * (L_phi + f) + ds * (M_phi - output)

        # Interior loss
        interior_loss = nn.MSELoss()(output + evaluation, output)

        # Boundary loss
        output_boundary = model_phi(Ts, S_boundarys)
        g = S_boundarys[:, 0] + S_boundarys[:, 1] * S_boundarys[:, -1] - 2*self.eta*S_boundarys[:, 1]**2 # terminal condition
        g = g.reshape(-1, 1).to(self.device)
        boundary_loss = nn.MSELoss()(output_boundary, g)

        # Combine losses

        lamb_bc = 1e-1
        loss = interior_loss + boundary_loss*lamb_bc
        print(interior_loss, boundary_loss)
        # Optionally freeze φ
        if not train_phi:
            loss = loss.detach()

        return loss, evaluation

    def loss_phi_hawkes(self, model_phi, model_d, model_u,
                        ts, Ss, lambdas, Ts, S_boundarys, lambdas_boundary,
                        train_phi=False, train_d=False, train_u=False, tau=0.5):
        # Use autograd to calculate time derivative
        ts.requires_grad_(True)
        lambdas.requires_grad_(True)

        # Forward pass
        lambdas_E = torch.log(torch.clamp(torch.exp(lambdas).sum(dim=2), min=1e-6))
        lambdas_E = (lambdas_E - self.means_lamb) / (self.std_lamb + 1e-8)
        lambdas_E[:, self.std_lamb==0] = 0
        states = torch.hstack([Ss, lambdas_E])
        output = self.ansatz_output(states, ts, model_phi).to(self.device) # ansatz

        # Derivative wrt ts
        phi_t = torch.autograd.grad(output.sum(), ts, create_graph=True)[0].to(self.device)

        # Derivative wrt lambdas
        grad_lambda = torch.autograd.grad(output.sum(), lambdas, create_graph=True)[0]

        # Mask frozen entries
        frozen_mask = (self.alphas == 0).unsqueeze(0).expand_as(lambdas)
        grad_lambda = grad_lambda.masked_fill(frozen_mask, 0.0)

        # Derivative wrt exp(lambdas)
        phi_lamb = (
                torch.exp(lambdas).reshape(lambdas.shape[0], -1)
                * grad_lambda.reshape(lambdas.shape[0], -1)
        ).to(self.device)


        # Calculate integral term
        I_phi = torch.zeros_like(output).to(self.device)
        for i in range(self.NDIMS):
            Ss_transitioned = self.transition(Ss, torch.tensor(i, dtype=torch.float32, device=self.device))
            lambdas_transitioned = self.transition_lambda(lambdas, i)
            lambdas_transitioned = torch.log(torch.clamp(torch.exp(lambdas_transitioned).sum(dim=2), min=1e-6))
            lambdas_transitioned = (lambdas_transitioned - self.means_lamb)/(self.std_lamb + 1e-8)
            lambdas_transitioned[:, self.std_lamb==0] = 0
            states_transitioned = torch.hstack([Ss_transitioned, lambdas_transitioned])
            I_phi += torch.exp(lambdas[:,i,:]).sum(dim=1).reshape(lambdas.shape[0],1) * (self.ansatz_output(states_transitioned, ts, model_phi) - output)

        # Generator term
        L_phi = phi_t + I_phi + ((self.gammas *(self.mus -  torch.exp(lambdas))).reshape(lambdas_transitioned.shape[0],-1) * phi_lamb).sum(dim=1).reshape(lambdas.shape[0],1)

        # Running cost
        states_lob = self.means + Ss[:,:11].clone()*(self.stds + 1e-8)
        f = -self.eta * torch.square(states_lob[:, 1])
        f = f.reshape(-1, 1).to(self.device)

        # Decision network
        logits_d = model_d(ts, states)[1]
        if train_d:
            ds_onehot = gumbel_softmax_sample(logits_d, tau=tau, hard=True)  # categorical + differentiable
            ds = ds_onehot[:, 1:2]   # assuming binary: [no-intervene, intervene]
        else:
            ds = torch.argmax(logits_d, dim=-1).reshape(-1, 1).float().detach()

        # Control network
        logits_u = model_u(ts, states)[1]
        if not train_u:
            logits_u = logits_u.detach()

        # Intervention value
        transition_lambdas_E = torch.log(torch.clamp(torch.exp(self.transition_lambda(lambdas, logits_u)).sum(dim=2), min=1e-6))
        transition_lambdas_E = (transition_lambdas_E - self.means_lamb)/(self.std_lamb + 1e-8)
        transition_lambdas_E[:, self.std_lamb==0] = 0
        states_intervened = torch.hstack([self.intervention(ts, Ss, logits_u), transition_lambdas_E])
        M_phi = self.ansatz_output(states_intervened, ts, model_phi)

        # HJB evaluation
        print(ds.unique(return_counts=True))
        evaluation = (1 - ds) * (L_phi + f) + ds * (M_phi - output)
        evaluation /= 1000.0
        # Interior loss
        interior_loss = nn.MSELoss()(evaluation, torch.zeros_like(evaluation))

        # Boundary loss
        lambdas_boundary = torch.log(torch.clamp(torch.exp(lambdas_boundary).sum(dim=2), min=1e-6))
        lambdas_boundary = (lambdas_boundary - self.means_lamb)/(self.std_lamb + 1e-8)
        lambdas_boundary[:, self.std_lamb==0] = 0
        states_boundary = torch.hstack([S_boundarys, lambdas_boundary])
        output_boundary = self.ansatz_output(states_boundary, Ts, model_phi)
        g = S_boundarys[:, 0] + S_boundarys[:, 1] * S_boundarys[:, -1]
        g = g.reshape(-1, 1).to(self.device)
        boundary_loss = nn.MSELoss()(output_boundary, g)

        # Combine losses
        lamb_bc = 0
        loss = interior_loss + boundary_loss*lamb_bc
        print(interior_loss, boundary_loss)

        # Optionally freeze φ by detaching its contribution
        if not train_phi:
            loss = loss.detach()

        return loss, evaluation

    def update_alpha(self, log_alpha, entropy, target_entropy, optimizer_alpha):
        alpha_loss = -(log_alpha * (entropy.detach() - target_entropy)).mean()
        optimizer_alpha.zero_grad()
        alpha_loss.backward()
        optimizer_alpha.step()
        return alpha_loss.item(), log_alpha.exp().item()

    def train_step_hawkes(self, model_phi, optimizer_phi, scheduler_phi, model_d, optimizer_d, scheduler_d, model_u, optimizer_u, scheduler_u, phi_epochs=10, phi_optim='ADAM', freeze_d = False):
        # Generate sample data
        if self.SAMPLER=='iid':
            ts, Ss, lambdas = self.sampler(self.NUM_POINTS, hawkes=True)
            Ts, S_boundarys, lambdas_boundary = self.sampler(self.NUM_POINTS, boundary=True, hawkes=True)
        else:
            raise Exception('invalid sampler')
        ts.to(self.device)
        Ss.to(self.device)
        lambdas.to(self.device)
        Ts.to(self.device)
        S_boundarys.to(self.device)
        lambdas_boundary.to(self.device)
        # Train value function
        train_loss_phi = 0.0

        # Use gradient clipping to prevent exploding gradients
        for j in range(phi_epochs):
            loss_phi, _ = self.loss_phi_hawkes(model_phi, model_d, model_u, ts, Ss, lambdas, Ts, S_boundarys, lambdas_boundary,  train_phi=True)
            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_phi.parameters(), 1.0)
            if phi_optim == 'ADAM':
                optimizer_phi.zero_grad()
                loss_phi.backward()
                optimizer_phi.step()
            train_loss_phi = loss_phi.item()
            if scheduler_phi is not None:
                scheduler_phi.step()


            # Check for NaN or Inf
            if torch.isnan(loss_phi) or torch.isinf(loss_phi):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model Phi loss: {train_loss_phi:0.4f}')
            for param_group in optimizer_phi.param_groups:
                lr = param_group['lr']
                print('Model Phi LR: '+str(lr))
        lambdas_E = torch.log(torch.clamp(torch.exp(lambdas).sum(dim=2), min=1e-6))
        lambdas_E = (lambdas_E - self.means_lamb)/(self.std_lamb + 1e-8)
        lambdas_E[:, self.std_lamb==0] = 0
        states = torch.hstack([Ss, lambdas_E]).detach()
        # # Train control function
        for j in range(phi_epochs):
            _, eval = self.loss_phi_hawkes(model_phi, model_d, model_u, ts, Ss, lambdas, Ts, S_boundarys, lambdas_boundary,  train_u = True)
            loss_u = -eval.mean()

            u_logits = model_u(ts, states)[1]
            u_entropy_loss = -(torch.softmax(u_logits + 1e-3, dim=1) * F.log_softmax(u_logits + 1e-20, dim=1)).sum(dim=1).mean()
            loss_u -= 1*u_entropy_loss
            optimizer_u.zero_grad()
            loss_u.backward()
            # --- Check for vanishing gradients ---
            grad_norm = 0.0
            for p in model_u.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            print(grad_norm)
            if grad_norm < 1e-6:
                print(f"Warning: vanishing gradients detected (norm={grad_norm:.2e})")
            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_u.parameters(), 1.0)
            optimizer_u.step()
            train_loss_u = loss_u.item()
            scheduler_u.step()
            # Check for NaN or Inf
            if torch.isnan(loss_u) or torch.isinf(loss_u):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model u loss: {train_loss_u:0.4f}')
        train_loss_d = 0
        for j in range(phi_epochs*int(freeze_d)):
            _, eval = self.loss_phi_hawkes(model_phi, model_d, model_u, ts, Ss, lambdas, Ts, S_boundarys, lambdas_boundary,  train_d = True)
            loss_d = -eval.mean()
            d_logits = model_d(ts, states)[1]
            d_entropy_loss = -(torch.softmax(d_logits + 1e-3, dim=1) * F.log_softmax(d_logits+1e-20, dim=1)).sum(dim=1).mean()
            loss_d -= 10*d_entropy_loss
            optimizer_d.zero_grad()
            loss_d.backward()
            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), 1.0)
            optimizer_d.step()
            train_loss_d = loss_d.item()
            scheduler_d.step()
            # Check for NaN or Inf
            if torch.isnan(loss_d) or torch.isinf(loss_d):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model d loss: {train_loss_d:0.4f}')



        # train_loss_u = 0.0
        acc_u = 0.0
        acc_d = 0.0
        # gt_u = self.oracle_u(model_phi, ts, states)
        pred_u, prob_us = model_u(ts, states)
        #
        # acc_u = 100*torch.sum(pred_u.flatten() == gt_u).item() / len(pred_u)
        #
        # print(f'Model u Acc: {acc_u:0.4f}')
        #
        # # # Train decision function
        # gt_d = self.oracle_d(model_phi, model_u, ts, states)
        # #
        pred_d, prob_ds = model_d(ts, states)
        #
        # acc_d = 100*torch.sum(pred_d.flatten() == gt_d).item() / len(pred_d)

        # print(f'Model d Acc: {acc_d:0.4f}')
        print(pred_d.unique(return_counts=True))
        print(pred_u.unique(return_counts=True))
        return model_phi, model_d, model_u, train_loss_phi, train_loss_d, train_loss_u, acc_d, acc_u

    def train_step(self, model_phi, optimizer_phi, scheduler_phi, model_d, optimizer_d, scheduler_d, model_u, optimizer_u, scheduler_u, phi_epochs=10, phi_optim='ADAM', hawkes=False, freeze_d = False):
        if hawkes:
            return self.train_step_hawkes(model_phi, optimizer_phi, scheduler_phi, model_d, optimizer_d, scheduler_d, model_u, optimizer_u, scheduler_u, phi_epochs=phi_epochs, phi_optim=phi_optim, freeze_d = freeze_d)
        # Setup for training
        lambdas = self.lambdas_poisson

        # Generate sample data
        if self.SAMPLER=='iid':
            ts, Ss = self.sampler(self.NUM_POINTS)
            Ts, S_boundarys = self.sampler(self.NUM_POINTS, boundary=True)
        elif self.SAMPLER == 'sim':
            ts, Ss = self.sampler_sim(num_points=self.NUM_POINTS)
            Ts, S_boundarys = self.sampler_sim(num_points=self.NUM_POINTS, boundary=True)
        else:
            raise Exception('invalid sampler')
        ts.to(self.device)
        Ss.to(self.device)
        Ts.to(self.device)
        S_boundarys.to(self.device)
        # Train value function
        train_loss_phi = 0.0

        def closure():
            optimizer_phi.zero_grad()
            loss_phi, _ = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas, train_phi=True)
            loss_phi.backward()
            return loss_phi
        # Use gradient clipping to prevent exploding gradients
        for j in range(phi_epochs):
            loss_phi, _ = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas, train_phi=True)
            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_phi.parameters(), 1.0)
            if phi_optim == 'ADAM':
                optimizer_phi.zero_grad()
                loss_phi.backward()
                optimizer_phi.step()
            else:
                optimizer_phi.step(closure)
            train_loss_phi = loss_phi.item()
            if scheduler_phi is not None:
                scheduler_phi.step()


            # Check for NaN or Inf
            if torch.isnan(loss_phi) or torch.isinf(loss_phi):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model Phi loss: {train_loss_phi:0.4f}')
            for param_group in optimizer_phi.param_groups:
                lr = param_group['lr']
                print('Model Phi LR: '+str(lr))
        # # Train control function
        for j in range(phi_epochs):
            _, eval = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas, train_u = True)
            loss_u = -eval.mean()
            u_logits = model_u(ts, Ss)[1]
            u_entropy_loss = -(torch.softmax(u_logits + 1e-3, dim=1) * F.log_softmax(u_logits + 1e-20, dim=1)).sum(dim=1).mean()
            loss_u -= .1*u_entropy_loss
            optimizer_u.zero_grad()
            loss_u.backward()
            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_u.parameters(), 1.0)
            optimizer_u.step()
            train_loss_u = loss_u.item()
            scheduler_u.step()
            # Check for NaN or Inf
            if torch.isnan(loss_u) or torch.isinf(loss_u):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model u loss: {train_loss_u:0.4f}')

        for j in range(phi_epochs):
            _, eval = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas, train_d = True)
            loss_d = -eval.mean()
            d_logits = model_d(ts, Ss)[1]
            d_entropy_loss = -(torch.softmax(d_logits + 1e-3, dim=1) * F.log_softmax(d_logits+1e-20, dim=1)).sum(dim=1).mean()
            loss_d -= 10*d_entropy_loss
            optimizer_d.zero_grad()
            loss_d.backward()
            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), 1.0)
            optimizer_d.step()
            train_loss_d = loss_d.item()
            scheduler_d.step()
            # Check for NaN or Inf
            if torch.isnan(loss_d) or torch.isinf(loss_d):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model d loss: {train_loss_d:0.4f}')
        gt_u = self.oracle_u(model_phi, ts, Ss)
        # train_loss_u = 0.0
        acc_u = 0.0
        acc_d = 0.0

        pred_u, prob_us = model_u(ts, Ss)

        acc_u = 100*torch.sum(pred_u.flatten() == gt_u).item() / len(pred_u)

        print(f'Model u Acc: {acc_u:0.4f}')

        # # Train decision function
        gt_d = self.oracle_d(model_phi, model_u, ts, Ss)
        #
        pred_d, prob_ds = model_d(ts, Ss)

        acc_d = 100*torch.sum(pred_d.flatten() == gt_d).item() / len(pred_d)

        print(f'Model d Acc: {acc_d:0.4f}')

        return model_phi, model_d, model_u, train_loss_phi, train_loss_d, train_loss_u, acc_d, acc_u

    def train(self, **kwargs):
        # Default parameter values
        defaults = {
            'continue_training': False,
            'checkpoint_label' : None,
            'continue_epoch': 0,
            'checkpoint_frequency': 50,
            'layer_widths': [20, 20, 20],
            'n_layers': [5, 5, 5],
            'typeNN': 'LSTM',
            'log_dir': './logs',
            'model_dir': './models',
            'label': None,
            'refresh_epoch' : 300,
            'lr' : 1e-2,
            'sampler': 'sim',
            'phi_epochs' : 10,
            'phi_optim' : 'ADAM',
            'unified': False,
            'feature_width' : 25,
            'activation' : 'tanh',
            'hawkes' : self.hawkes
        }

        # Update defaults with provided kwargs
        for key, value in kwargs.items():
            if key in defaults:
                defaults[key] = value

        # Extract parameters for use
        hawkes = defaults['hawkes']
        continue_training = defaults['continue_training']
        checkpoint_label = defaults['checkpoint_label']
        continue_epoch = defaults['continue_epoch']
        checkpoint_frequency = defaults['checkpoint_frequency']
        layer_widths = defaults['layer_widths']
        n_layers = defaults['n_layers']
        typeNN = defaults['typeNN']
        typeNN2= typeNN
        if type(typeNN) == list:
            typeNN2 =typeNN[1]
            typeNN = typeNN[0]
        log_dir = defaults['log_dir']
        model_dir = defaults['model_dir']
        label = defaults['label']
        refresh_epoch = defaults['refresh_epoch']
        lr = defaults['lr']
        sampler = defaults['sampler']
        phi_epochs = defaults['phi_epochs']
        phi_optim = defaults['phi_optim']
        unified = defaults['unified']
        feature_width = defaults['feature_width']
        activation = defaults['activation']
        self.SAMPLER = sampler
        # Initialize logger and model manager
        logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
        model_manager = ModelManager(model_dir = model_dir, label = label)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device=device
        # Create models
        input_dim = 11
        if hawkes: input_dim += len(self.E)
        if unified:
            model0 = DGM.DGMNet(layer_widths[0], n_layers[0], input_dim, output_dim= feature_width, typeNN = typeNN, hidden_activation=activation)
            model_phi = DGM.DenseNet(feature_width, layer_widths[1], n_layers[1], 1, model0, hidden_activation=activation)
            model_u = DGM.DenseNet(feature_width, layer_widths[2], n_layers[2], 10, model0, hidden_activation=activation)
            model_d = DGM.DenseNet(feature_width, layer_widths[3], n_layers[3], 2, model0, hidden_activation=activation)
        else:
            model_phi = DGM.DGMNet(layer_widths[0], n_layers[0], input_dim, typeNN = typeNN, hidden_activation=activation)
            model_u = DGM.PIANet(layer_widths[1], n_layers[1], input_dim, len(self.U), typeNN = typeNN2, hidden_activation=activation)
            model_d = DGM.PIANet(layer_widths[2], n_layers[2], input_dim, 2, typeNN = typeNN2, hidden_activation=activation)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            if unified:
                nn.DataParallel(model0)
            model_phi = nn.DataParallel(model_phi)
            model_u = nn.DataParallel(model_u)
            model_d = nn.DataParallel(model_d)


        # Load existing models if continuing training
        if continue_training:
            loadedModels = model_manager.load_models(phi=model_phi, d=model_d, u=model_u, timestamp= checkpoint_label, epoch = continue_epoch)
            if loadedModels is not None:
                model_phi, model_d, model_u = loadedModels['phi'], loadedModels['d'], loadedModels['u']
                print("Resuming training from previously saved models")
        model_phi.to(device)
        model_u.to(device)
        model_d.to(device)
        if unified: model0.to(device)
        # Define lambda function for the scheduler
        def lr_lambda(epoch):
            # Calculate decay rate
            #decay_rate = np.log(1e-4) / (self.EPOCHS*phi_epochs - 1)
            #return np.max([1e-5,np.exp(decay_rate * epoch )])
            #return (1 - 1e-5)**epoch
            return np.max([1e-6,0.5**(epoch//5000)])

        def lr_lambda_lbfgs(epoch):
            # Calculate decay rate
            #decay_rate = np.log(1e-4) / (self.EPOCHS*phi_epochs - 1)
            #return np.max([1e-5,np.exp(decay_rate * epoch )])
            #return (1 - 1e-5)**epoch
            return np.max([1e-2,0.1**(epoch//1000)])

        if phi_optim == 'ADAM':
            optimizer_phi = optim.Adam(model_phi.parameters(), lr=lr)
            scheduler_phi = optim.lr_scheduler.LambdaLR(optimizer_phi, lr_lambda)
        else:
            optimizer_phi = optim.LBFGS(model_phi.parameters(), lr=lr*100, line_search_fn ='strong_wolfe')
            # scheduler_phi = optim.lr_scheduler.LambdaLR(optimizer_phi, lr_lambda_lbfgs)
            scheduler_phi = None
        optimizer_u = optim.Adam(model_u.parameters(), lr=lr)
        optimizer_d = optim.Adam(model_d.parameters(), lr=lr)

        # Set up schedulers

        scheduler_u = optim.lr_scheduler.LambdaLR(optimizer_u, lr_lambda)
        scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda)

        #entropy target a la SAC
        self.log_alpha_u = torch.tensor(2.0, requires_grad=True, device=self.device)
        self.log_alpha_d = torch.tensor(2.0, requires_grad=True, device=self.device)
        self.optimizer_alpha_u = torch.optim.Adam([self.log_alpha_u], lr=3e-4)
        self.optimizer_alpha_d = torch.optim.Adam([self.log_alpha_d], lr=3e-4)
        self.target_entropy_u = -np.log(1.0 / len(self.U))  # or a tuned value
        self.target_entropy_d = -np.log(1.0 / 2)
        # Training loop
        freeze_d = False
        for epoch in range(continue_epoch, self.EPOCHS):
            if epoch > 50: freeze_d = True
            print(f"\nEpoch {epoch+1}/{self.EPOCHS}")
            #KJ: PoC needed for a simple LOB model - check the solution wrt real soln
            #KJ: also useful would be Mguni et al. comparison
            #KJ: 1. 2D poisson flow + impulse control using ansatz ala AS, using DGM and compare soln
            #    2. 2D hawkes flow + impulse control using ansatz ala AS, using DGM and compare soln
            #    3. 12D poisson + IC using DGM
            #    4. 12D Hawkes + IC using DGM
            #    5. Compare 4 w DRL approach of Mguni
            model_phi, model_d, model_u, loss_phi, loss_d, loss_u, acc_u, acc_d = self.train_step(
                model_phi, optimizer_phi, scheduler_phi, model_d, optimizer_d, scheduler_d, model_u, optimizer_u, scheduler_u, phi_epochs = phi_epochs, phi_optim = phi_optim, hawkes=hawkes, freeze_d = freeze_d
            )

            # Log losses
            logger.log_losses(phi_loss=loss_phi, d_loss=loss_d, u_loss=loss_u, acc_d=acc_d, acc_u=acc_u)

            # Save checkpoint at specified frequency
            if (epoch + 1) % checkpoint_frequency == 0:
                print(f"Saving checkpoint at epoch {epoch+1}")
                model_manager.save_models(phi=model_phi, d=model_d, u=model_u, epoch=epoch+1)
                # Save and plot losses
                logger.save_logs()
                logger.plot_losses(show=True, save=True)
            # if (epoch+1) == refresh_epoch:
            #     model_u = DGM.PIANet(layer_widths[1], n_layers[1], 11, 10, typeNN = typeNN)
            #     model_d = DGM.PIANet(layer_widths[2], n_layers[2], 11, 2, typeNN = typeNN)
            # Step learning rate schedulers

            # Print epoch summary
            print(f"Epoch {epoch+1} summary - Phi Loss: {loss_phi:.4f}, D Loss: {loss_d:.4f}, U Loss: {loss_u:.4f}")

            # if (epoch+1) == continue_epoch + 1000:
            #     print('Switching to LBFGS for Phi')
            #     optimizer_phi = optim.LBFGS(model_phi.parameters())
            #     # Set up schedulers
            #     scheduler_phi = optim.lr_scheduler.LambdaLR(optimizer_phi, lr_lambda)
            #     phi_optim = 'LBFGS'
        # Save final model
        model_manager.save_models(phi=model_phi, d=model_d, u=model_u)

        print("Training completed. Models saved and losses plotted.")

        return model_phi, model_d, model_u

class MarketMakingUnifiedControl(MarketMaking):
    def __init__(self, num_points=100, num_epochs=1000, ric='AAPL'):
        super(MarketMakingUnifiedControl, self).__init__(num_points=num_points, num_epochs=num_epochs, ric=ric)

    def oracle_u(self, model_phi, ts, Ss):
        # Use PyTorch operations to compute the best action
        batch_size = ts.shape[0]

        # Initialize tensor to store intervention values for each action
        interventions = torch.zeros((batch_size, len(self.U) + 1), dtype=torch.float32, device=self.device)

        # Calculate value for each action
        for u in range(len(self.U)):
            # Create a tensor of action u for all batch items
            u_tensor = torch.ones((batch_size, 1), dtype=torch.float32) * u

            # Calculate intervention value
            intervention_value = self.intervention(model_phi, ts, Ss, u_tensor)

            # Store in interventions tensor
            indices = torch.stack([torch.arange(batch_size, dtype=torch.int64),
                                   torch.ones(batch_size, dtype=torch.int64) * u], dim=1)
            interventions[indices[:, 0], indices[:, 1]] = intervention_value.reshape(-1).to(self.device)

        u_tensor = torch.ones((batch_size, 1), dtype=torch.float32) * len(self.U)

        # Calculate value
        ts.requires_grad_(True)
        output = model_phi(ts, Ss)
        output.to(self.device)

        phi_t = torch.autograd.grad(output.sum(), ts, create_graph=True)[0].to(self.device)
        #ts.requires_grad_(False)

        # Calculate integral term
        I_phi = torch.zeros_like(output)
        I_phi.to(self.device)
        for i in range(self.NDIMS):
            # Calculate transition for event i
            Ss_transitioned = self.transition(Ss, torch.tensor(i, dtype=torch.float32))
            Ss_transitioned.to(self.device)
            I_phi += self.lambdas_poisson[i] * (model_phi(ts, Ss_transitioned) - output)

        # Calculate generator term
        L_phi = phi_t + I_phi

        # Calculate running cost
        f = -self.eta * torch.square(Ss[:, 1])  # Inventory penalty
        f = f.reshape(-1, 1).to(self.device)
        intervention_value = L_phi + f

        # Store in interventions tensor
        indices = torch.stack([torch.arange(batch_size, dtype=torch.int64),
                               torch.ones(batch_size, dtype=torch.int64) * len(self.U)], dim=1)
        interventions[indices[:, 0], indices[:, 1]] = intervention_value.reshape(-1).to(self.device)

        # Return the action with highest value
        return torch.argmax(interventions, dim=1).to(self.device)

    def loss_phi_poisson(self, model_phi, model_u, ts, Ss, Ts, S_boundarys, lambdas):
        # Use autograd to calculate time derivative
        ts.requires_grad_(True)
        output = model_phi(ts, Ss)
        output.to(self.device)

        phi_t = torch.autograd.grad(output.sum(), ts, create_graph=True)[0].to(self.device)
        #ts.requires_grad_(False)

        # Calculate integral term
        I_phi = torch.zeros_like(output)
        I_phi.to(self.device)
        for i in range(self.NDIMS):
            # Calculate transition for event i
            Ss_transitioned = self.transition(Ss, torch.tensor(i, dtype=torch.float32))
            Ss_transitioned.to(self.device)
            I_phi += lambdas[i] * (model_phi(ts, Ss_transitioned) - output)

        # Calculate generator term
        L_phi = phi_t + I_phi

        # Calculate running cost
        f = -self.eta * torch.square(Ss[:, 1])  # Inventory penalty
        f = f.reshape(-1, 1).to(self.device)

        # Get optimal decision and control

        us, _ = model_u(ts, Ss)
        ds = torch.ones_like(us).to(self.device)
        ds[torch.where(us == len(self.U))] = 0
        us[torch.where(us == len(self.U))] = 0 # some default value
        # Calculate intervention value
        M_phi = self.intervention(model_phi, ts, Ss, us)

        # Calculate HJB evaluation
        evaluation = (1 - ds) * (L_phi + f) + ds * (M_phi - output)

        # Interior loss
        interior_loss = nn.MSELoss()(evaluation, torch.zeros_like(evaluation))

        # Boundary loss
        output_boundary = model_phi(Ts, S_boundarys)
        g = S_boundarys[:, 0] + S_boundarys[:, 1] * S_boundarys[:, -1]  # Terminal condition
        g = g.reshape(-1, 1).to(self.device)
        boundary_loss = nn.MSELoss()(output_boundary, g)

        # Combine losses with potential weighting
        loss = interior_loss + boundary_loss

        return loss

    def train_step(self, model_phi, optimizer_phi, scheduler_phi, model_u, optimizer_u, scheduler_u, phi_epochs=10, phi_optim='ADAM'):
        # Setup for training
        lambdas = self.lambdas_poisson

        # Generate sample data
        if self.SAMPLER=='iid':
            ts, Ss = self.sampler(self.NUM_POINTS)
            Ts, S_boundarys = self.sampler(self.NUM_POINTS, boundary=True)
        elif self.SAMPLER == 'sim':
            ts, Ss = self.sampler_sim(num_points=self.NUM_POINTS)
            Ts, S_boundarys = self.sampler_sim(num_points=self.NUM_POINTS, boundary=True)
        else:
            raise Exception('invalid sampler')
        ts.to(self.device)
        Ss.to(self.device)
        Ts.to(self.device)
        S_boundarys.to(self.device)
        # Train value function
        train_loss_phi = 0.0

        def closure():
            optimizer_phi.zero_grad()
            loss_phi = torch.log(self.loss_phi_poisson(model_phi, model_u, ts, Ss, Ts, S_boundarys, lambdas))
            loss_phi.backward()
            return loss_phi
        # Use gradient clipping to prevent exploding gradients
        for j in range(phi_epochs):
            loss_phi = self.loss_phi_poisson(model_phi, model_u, ts, Ss, Ts, S_boundarys, lambdas)
            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_phi.parameters(), 1.0)
            if phi_optim == 'ADAM':
                optimizer_phi.zero_grad()
                loss_phi.backward()
                optimizer_phi.step()
            else:
                optimizer_phi.step(closure)
            train_loss_phi = loss_phi.item()
            scheduler_phi.step()


            # Check for NaN or Inf
            if torch.isnan(loss_phi) or torch.isinf(loss_phi):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model Phi loss: {train_loss_phi:0.4f}')
            for param_group in optimizer_phi.param_groups:
                lr = param_group['lr']
                print('Model Phi LR: '+str(lr))
        # Train control function
        gt_u = self.oracle_u(model_phi, ts, Ss)
        train_loss_u = 0.0
        acc_u = 0.0
        loss_object_u = nn.CrossEntropyLoss()

        for j in range(phi_epochs):
            optimizer_u.zero_grad()
            pred_u, prob_us = model_u(ts, Ss)
            print(np.unique(gt_u.cpu(), return_counts=True))
            print(np.unique(pred_u.cpu(), return_counts=True))
            acc_u = 100*torch.sum(pred_u.flatten() == gt_u).item() / len(pred_u)
            loss_u = loss_object_u(prob_us, gt_u)
            loss_u.backward()

            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_u.parameters(), 1.0)
            optimizer_u.step()
            train_loss_u = loss_u.item()
            scheduler_u.step()

            # Check for NaN or Inf
            if torch.isnan(loss_u) or torch.isinf(loss_u):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model u loss: {train_loss_u:0.4f}')
            print(f'Model u Acc: {acc_u:0.4f}')
            for param_group in optimizer_u.param_groups:
                lr = param_group['lr']
                print('Model U LR: '+str(lr))

        return model_phi, model_u, train_loss_phi, train_loss_u, acc_u

    def train(self, **kwargs):
        # Default parameter values
        defaults = {
            'continue_training': False,
            'checkpoint_label' : None,
            'continue_epoch': 0,
            'checkpoint_frequency': 50,
            'layer_widths': [20, 20],
            'n_layers': [5, 5],
            'typeNN': 'LSTM',
            'log_dir': './logs',
            'model_dir': './models',
            'label': None,
            'refresh_epoch' : 300,
            'lr' : 1e-3,
            'sampler': 'sim',
            'phi_epochs' : 10,
            'phi_optim' : 'ADAM',
            'unified': False,
            'feature_width' : 25
        }

        # Update defaults with provided kwargs
        for key, value in kwargs.items():
            if key in defaults:
                defaults[key] = value

        # Extract parameters for use
        continue_training = defaults['continue_training']
        checkpoint_label = defaults['checkpoint_label']
        continue_epoch = defaults['continue_epoch']
        checkpoint_frequency = defaults['checkpoint_frequency']
        layer_widths = defaults['layer_widths']
        n_layers = defaults['n_layers']
        typeNN = defaults['typeNN']
        typeNN2= typeNN
        if type(typeNN) == list:
            typeNN2 =typeNN[1]
            typeNN = typeNN[0]
        log_dir = defaults['log_dir']
        model_dir = defaults['model_dir']
        label = defaults['label']
        refresh_epoch = defaults['refresh_epoch']
        lr = defaults['lr']
        sampler = defaults['sampler']
        phi_epochs = defaults['phi_epochs']
        phi_optim = defaults['phi_optim']
        unified = defaults['unified']
        feature_width = defaults['feature_width']
        self.SAMPLER = sampler
        # Initialize logger and model manager
        logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
        model_manager = ModelManager(model_dir = model_dir, label = label)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device=device
        # Create models
        if unified:
            model0 = DGM.DGMNet(layer_widths[0], n_layers[0], 11, output_dim= feature_width, typeNN = typeNN)
            model_phi = DGM.DenseNet(feature_width, layer_widths[1], n_layers[1], 1, model0)
            model_u = DGM.DenseNet(feature_width, layer_widths[2], n_layers[2], 10+1, model0)
        else:
            model_phi = DGM.DGMNet(layer_widths[0], n_layers[0], 11, typeNN = typeNN)
            model_u = DGM.PIANet(layer_widths[1], n_layers[1], 11, 10 + 1, typeNN = typeNN2)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            if unified:
                nn.DataParallel(model0)
            model_phi = nn.DataParallel(model_phi)
            model_u = nn.DataParallel(model_u)

        # Load existing models if continuing training
        if continue_training:
            loadedModels = model_manager.load_models(phi=model_phi, u=model_u, timestamp= checkpoint_label, epoch = continue_epoch)
            if loadedModels is not None:
                model_phi, model_u = loadedModels['phi'], loadedModels['u']
                print("Resuming training from previously saved models")
        model_phi.to(device)
        model_u.to(device)
        if unified: model0.to(device)
        # Define lambda function for the scheduler
        def lr_lambda(epoch):
            # Calculate decay rate
            #decay_rate = np.log(1e-4) / (self.EPOCHS*phi_epochs - 1)
            #return np.max([1e-5,np.exp(decay_rate * epoch )])
            return (1 - 1e-5)**epoch

        if phi_optim == 'ADAM':
            optimizer_phi = optim.Adam(model_phi.parameters(), lr=lr)
        else:
            optimizer_phi = optim.LBFGS(model_phi.parameters(), lr=lr*10000)
        optimizer_u = optim.Adam(model_u.parameters(), lr=lr)

        # Set up schedulers
        scheduler_phi = optim.lr_scheduler.LambdaLR(optimizer_phi, lr_lambda)
        scheduler_u = optim.lr_scheduler.LambdaLR(optimizer_u, lr_lambda)

        # Training loop
        for epoch in range(continue_epoch, self.EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.EPOCHS}")
            #KJ: PoC needed for a simple LOB model - check the solution wrt real soln
            #KJ: also useful would be Mguni et al. comparison
            #KJ: 1. 2D poisson flow + impulse control using ansatz ala AS, using DGM and compare soln
            #    2. 2D hawkes flow + impulse control using ansatz ala AS, using DGM and compare soln
            #    3. 12D poisson + IC using DGM
            #    4. 12D Hawkes + IC using DGM
            #    5. Compare 4 w DRL approach of Mguni
            model_phi, model_u, loss_phi, loss_u, acc_u = self.train_step(
                model_phi, optimizer_phi, scheduler_phi, model_u, optimizer_u, scheduler_u, phi_epochs = phi_epochs, phi_optim = phi_optim
            )

            # Log losses
            logger.log_losses(phi_loss=loss_phi, u_loss=loss_u, acc_u=acc_u)

            # Save checkpoint at specified frequency
            if (epoch + 1) % checkpoint_frequency == 0:
                print(f"Saving checkpoint at epoch {epoch+1}")
                model_manager.save_models(phi=model_phi, u=model_u, epoch=epoch+1)
                # Save and plot losses
                logger.save_logs()
                logger.plot_losses(show=True, save=True)
            # if (epoch+1) == refresh_epoch:
            #     model_u = DGM.PIANet(layer_widths[1], n_layers[1], 11, 10, typeNN = typeNN)
            #     model_d = DGM.PIANet(layer_widths[2], n_layers[2], 11, 2, typeNN = typeNN)
            # Step learning rate schedulers

            # Print epoch summary
            print(f"Epoch {epoch+1} summary - Phi Loss: {loss_phi:.4f}, U Loss: {loss_u:.4f}")

            if (epoch+1) == continue_epoch + 1000:
                print('Switching to LBFGS for Phi')
                optimizer_phi = optim.LBFGS(model_phi.parameters())
                # Set up schedulers
                scheduler_phi = optim.lr_scheduler.LambdaLR(optimizer_phi, lr_lambda)
                phi_optim = 'LBFGS'
        # Save final model
        model_manager.save_models(phi=model_phi, u=model_u)

        print("Training completed. Models saved and losses plotted.")

        return model_phi, model_u

# get_gpu_specs()
MM = MarketMaking(num_epochs=2000, num_points=1000, hawkes=True)
MM.train(lr =1e-3, ric='INTC', phi_epochs = 5, sampler='iid',log_dir = 'logs', model_dir = 'models', typeNN='LSTM', layer_widths = [50, 50, 50], n_layers= [5,5,5], unified=False, label = 'LSTM_INTC_hawkes_tc1', activation='relu')