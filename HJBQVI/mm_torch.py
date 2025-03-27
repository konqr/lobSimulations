import torch
import numpy as np
import DGMTorch as DGM
from DGMTorch import TrainingLogger, ModelManager, get_gpu_specs
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import pandas as pd

class MarketMaking():
    def __init__(self, num_points=100, num_epochs=1000, ric='AAPL'):
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
        self.TERMINATION_TIME = 23400
        self.NDIMS = 12
        self.NUM_POINTS = num_points
        self.EPOCHS = num_epochs
        self.eta = 125  # inventory penalty
        self.E = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
                  "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
        self.U = ["lo_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
                  "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "lo_deep_Bid"]
        # self.lambdas_poisson = [.86, .32, .33, .48, .02, .47, .47, .02, .48, .33, .32, .86]  # [5] * 12 # AMZN
        # INTC : {'lo_deep_Ask': 1.3291742343304844, 'co_deep_Ask': 2.2448482015669513, 'lo_top_Ask': 7.89707621082621, 'co_top_Ask': 6.617852118945869, 'mo_Ask': 0.5408440170940172, 'lo_inspread_Ask': 0.1327911324786325, 'lo_inspread_Bid': 0.1327911324786325, 'mo_Bid': 0.5408440170940172, 'co_top_Bid': 6.617852118945869, 'lo_top_Bid': 7.89707621082621, 'co_deep_Bid': 2.2448482015669513, 'lo_deep_Bid': 1.3291742343304844}
        # AAPL : {'lo_deep_Ask': 1.9115059993425376, 'co_deep_Ask': 2.3503168145956606, 'lo_top_Ask': 4.392785174227481, 'co_top_Ask': 4.248318129520053, 'mo_Ask': 1.0989673734385272, 'lo_inspread_Ask': 1.0131743096646941, 'lo_inspread_Bid': 1.0131743096646941, 'mo_Bid': 1.0989673734385272, 'co_top_Bid': 4.248318129520053, 'lo_top_Bid': 4.392785174227481, 'co_deep_Bid': 2.3503168145956606, 'lo_deep_Bid': 1.9115059993425376}
        self.lambdas_poisson = [1.33, 2.24, 7.90, 6.62, 0.54, 0.13, 0.13, 0.54, 6.62, 7.90, 2.24, 1.33]
        if ric == 'AAPL':
            self.lambdas_poisson = [1.91, 2.35, 4.39, 4.23, 1.09, 1.01, 1.01, 1.09, 4.23, 4.39, 2.35, 1.91]
    def sampler(self, num_points=1000, seed=None, boundary=False):
        '''
        Sample points from the stationary distributions for the DGM learning
        :param num_points: number of points
        :return: samples of [0,T] x {state space}
        '''
        if seed:
            np.random.seed(seed)
            torch.random.set_seed(seed)

        # Generate sample data using NumPy
        Xs = np.round(1e3 * np.random.randn(num_points, 1), 2)
        Ys = np.round(10 * np.random.randn(num_points, 1), 2)
        P_mids = np.round(200 + 10 * np.random.randn(num_points, 1), 2) / 2
        spreads = 0.01 *  np.random.geometric(.8, [num_points, 1])
        p_as = np.round(P_mids + spreads / 2, 2)
        p_bs = p_as - spreads
        q_as = np.random.geometric(.002, [num_points, 1])
        qD_as = np.random.geometric(.0015, [num_points, 1])
        q_bs = np.random.geometric(.002, [num_points, 1])
        qD_bs = np.random.geometric(.0015, [num_points, 1])
        n_as = np.array([np.random.randint(0, b) for b in q_as + qD_as])
        n_bs = np.array([np.random.randint(0, b) for b in q_bs + qD_bs])

        t = np.random.uniform(0, self.TERMINATION_TIME, [num_points, 1])
        t_boundary = self.TERMINATION_TIME * np.ones([num_points, 1])

        # Convert to TensorFlow tensors immediately
        if boundary:
            return torch.tensor(t_boundary, dtype=torch.float32), torch.tensor(
                np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]), dtype=torch.float32)

        return torch.tensor(t, dtype=torch.float32), torch.tensor(
            np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]), dtype=torch.float32)

    def sampler_sim(self, ric='AAPL', num_points=1000, seed=None, boundary=False):
        path = '/SAN/fca/Konark_PhD_Experiments/simulated/poisson'
        files = os.listdir(path)
        files = [f for f in files if ('poisson' in f)&(ric in f)]
        files = np.random.choice(files, 10)
        Xs = np.round(1e3 * np.random.randn(num_points, 1), 2)
        Ys = np.round(10 * np.random.randn(num_points, 1), 2)
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

        return Ss_out

    def intervention(self, model_phi, ts, Ss, us):
        # Get batch size from inputs
        batch_size = ts.shape[0]
        device = ts.device
        Ss = Ss.cpu()
        us = us.cpu()
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

        # Initialize state variables to be updated
        Xs_updated = Xs.clone()
        Ys_updated = Ys.clone()
        p_as_updated = p_as.clone()
        p_bs_updated = p_bs.clone()
        q_as_updated = q_as.clone()
        q_bs_updated = q_bs.clone()
        qD_as_updated = qD_as.clone()
        qD_bs_updated = qD_bs.clone()
        n_as_updated = n_as.clone()
        n_bs_updated = n_bs.clone()
        P_mids_updated = P_mids.clone()

        # Initialize profit tensor
        inter_profit = torch.zeros(batch_size, device=device)

        # Handle different intervention types using PyTorch operations

        # Market order asks
        mo_asks_mask = (us == 3).flatten()
        if mo_asks_mask.any():
            # Apply market order ask intervention
            mo_q_as = q_as[mo_asks_mask]
            mo_n_as = n_as[mo_asks_mask]
            mo_qD_as = qD_as[mo_asks_mask]
            mo_p_as = p_as[mo_asks_mask]
            mo_P_mids = P_mids[mo_asks_mask]
            mo_Xs = Xs[mo_asks_mask]
            mo_Ys = Ys[mo_asks_mask]

            # Apply market order intervention
            mo_q_as_updated, mo_n_as_updated, mo_qD_as_updated, mo_p_as_updated, mo_P_mids_updated, mo_Xs_updated, mo_Ys_updated = self.tr_mo(
                1.0, mo_q_as, mo_n_as, mo_qD_as, mo_p_as, mo_P_mids, mo_Xs, mo_Ys, intervention=True
            )

            # Update state variables
            q_as_updated[mo_asks_mask] = mo_q_as_updated
            n_as_updated[mo_asks_mask] = mo_n_as_updated
            qD_as_updated[mo_asks_mask] = mo_qD_as_updated
            p_as_updated[mo_asks_mask] = mo_p_as_updated
            P_mids_updated[mo_asks_mask] = mo_P_mids_updated
            Xs_updated[mo_asks_mask] = mo_Xs_updated
            Ys_updated[mo_asks_mask] = mo_Ys_updated

            # Update profit
            inter_profit[mo_asks_mask] = mo_p_as

        # Market order bids
        mo_bids_mask = (us == 6).flatten()
        if mo_bids_mask.any():
            # Apply market order bid intervention
            mo_q_bs = q_bs[mo_bids_mask]
            mo_n_bs = n_bs[mo_bids_mask]
            mo_qD_bs = qD_bs[mo_bids_mask]
            mo_p_bs = p_bs[mo_bids_mask]
            mo_P_mids = P_mids[mo_bids_mask]
            mo_Xs = Xs[mo_bids_mask]
            mo_Ys = Ys[mo_bids_mask]

            # Apply market order intervention
            mo_q_bs_updated, mo_n_bs_updated, mo_qD_bs_updated, mo_p_bs_updated, mo_P_mids_updated, mo_Xs_updated, mo_Ys_updated = self.tr_mo(
                -1.0, mo_q_bs, mo_n_bs, mo_qD_bs, mo_p_bs, mo_P_mids, mo_Xs, mo_Ys, intervention=True
            )

            # Update state variables
            q_bs_updated[mo_bids_mask] = mo_q_bs_updated
            n_bs_updated[mo_bids_mask] = mo_n_bs_updated
            qD_bs_updated[mo_bids_mask] = mo_qD_bs_updated
            p_bs_updated[mo_bids_mask] = mo_p_bs_updated
            P_mids_updated[mo_bids_mask] = mo_P_mids_updated
            Xs_updated[mo_bids_mask] = mo_Xs_updated
            Ys_updated[mo_bids_mask] = mo_Ys_updated

            # Update profit (negative because we're paying to buy)
            inter_profit[mo_bids_mask] = -1.0 * mo_p_bs

        # Limit order deep asks
        lo_deep_asks_mask = (us == 0).flatten()
        if lo_deep_asks_mask.any():
            qD_as_updated[lo_deep_asks_mask] = qD_as[lo_deep_asks_mask] + 1.0

        # Limit order deep bids
        lo_deep_bids_mask = (us == 9).flatten()
        if lo_deep_bids_mask.any():
            qD_bs_updated[lo_deep_bids_mask] = qD_bs[lo_deep_bids_mask] + 1.0

        # Limit order top asks
        lo_top_asks_mask = (us == 1).flatten()
        if lo_top_asks_mask.any():
            # Update q_as
            q_as_updated[lo_top_asks_mask] = q_as[lo_top_asks_mask] + 1.0

            # Update n_as if necessary
            condition = n_as[lo_top_asks_mask] >= q_as_updated[lo_top_asks_mask]
            indices = lo_top_asks_mask.nonzero(as_tuple=True)[0][condition]
            n_as_updated[indices] = q_as_updated[indices]

        # Limit order top bids
        lo_top_bids_mask = (us == 8).flatten()
        if lo_top_bids_mask.any():
            # Update q_bs
            q_bs_updated[lo_top_bids_mask] = q_bs[lo_top_bids_mask] + 1.0

            # Update n_bs if necessary
            condition = n_bs[lo_top_bids_mask] >= q_bs_updated[lo_top_bids_mask]
            indices = lo_top_bids_mask.nonzero(as_tuple=True)[0][condition]
            n_bs_updated[indices] = q_bs_updated[indices]

        # Cancel order top asks
        co_top_asks_mask = (us == 2).flatten()
        if co_top_asks_mask.any():
            co_q_as = q_as[co_top_asks_mask]
            co_n_as = n_as[co_top_asks_mask]
            co_qD_as = qD_as[co_top_asks_mask]
            co_p_as = p_as[co_top_asks_mask]
            co_P_mids = P_mids[co_top_asks_mask]

            # Apply cancel order intervention
            co_q_as_updated, co_n_as_updated, co_qD_as_updated, co_p_as_updated, co_P_mids_updated = self.tr_co_top(
                1.0, co_q_as, co_n_as, co_qD_as, co_p_as, co_P_mids, intervention=True
            )

            # Update state variables
            q_as_updated[co_top_asks_mask] = co_q_as_updated
            n_as_updated[co_top_asks_mask] = co_n_as_updated
            qD_as_updated[co_top_asks_mask] = co_qD_as_updated
            p_as_updated[co_top_asks_mask] = co_p_as_updated
            P_mids_updated[co_top_asks_mask] = co_P_mids_updated

        # Cancel order top bids
        co_top_bids_mask = (us == 7).flatten()
        if co_top_bids_mask.any():
            co_q_bs = q_bs[co_top_bids_mask]
            co_n_bs = n_bs[co_top_bids_mask]
            co_qD_bs = qD_bs[co_top_bids_mask]
            co_p_bs = p_bs[co_top_bids_mask]
            co_P_mids = P_mids[co_top_bids_mask]

            # Apply cancel order intervention
            co_q_bs_updated, co_n_bs_updated, co_qD_bs_updated, co_p_bs_updated, co_P_mids_updated = self.tr_co_top(
                -1.0, co_q_bs, co_n_bs, co_qD_bs, co_p_bs, co_P_mids, intervention=True
            )

            # Update state variables
            q_bs_updated[co_top_bids_mask] = co_q_bs_updated
            n_bs_updated[co_top_bids_mask] = co_n_bs_updated
            qD_bs_updated[co_top_bids_mask] = co_qD_bs_updated
            p_bs_updated[co_top_bids_mask] = co_p_bs_updated
            P_mids_updated[co_top_bids_mask] = co_P_mids_updated

        # Limit order in-spread asks
        lo_is_asks_mask = (us == 4).flatten()
        spreads = p_as - p_bs
        mask_spread = (spreads >= 0.015).flatten()
        lo_is_asks_mask = torch.logical_and(lo_is_asks_mask, mask_spread)
        if lo_is_asks_mask.any():
            lo_q_as = q_as[lo_is_asks_mask]
            lo_n_as = n_as[lo_is_asks_mask]
            lo_qD_as = qD_as[lo_is_asks_mask]
            lo_p_as = p_as[lo_is_asks_mask]
            lo_P_mids = P_mids[lo_is_asks_mask]

            # Apply limit order in-spread intervention
            lo_q_as_updated, lo_qD_as_updated, lo_n_as_updated, lo_P_mids_updated, lo_p_as_updated = self.tr_is(
                1.0, lo_q_as, lo_qD_as, lo_n_as, lo_P_mids, lo_p_as, intervention=True
            )

            # Update state variables
            q_as_updated[lo_is_asks_mask] = lo_q_as_updated
            qD_as_updated[lo_is_asks_mask] = lo_qD_as_updated
            n_as_updated[lo_is_asks_mask] = lo_n_as_updated
            P_mids_updated[lo_is_asks_mask] = lo_P_mids_updated
            p_as_updated[lo_is_asks_mask] = lo_p_as_updated

        # Limit order in-spread bids
        lo_is_bids_mask = (us == 5).flatten()
        lo_is_bids_mask = torch.logical_and(lo_is_bids_mask, mask_spread)
        if lo_is_bids_mask.any():
            lo_q_bs = q_bs[lo_is_bids_mask]
            lo_n_bs = n_bs[lo_is_bids_mask]
            lo_qD_bs = qD_bs[lo_is_bids_mask]
            lo_p_bs = p_bs[lo_is_bids_mask]
            lo_P_mids = P_mids[lo_is_bids_mask]

            # Apply limit order in-spread intervention
            lo_q_bs_updated, lo_qD_bs_updated, lo_n_bs_updated, lo_P_mids_updated, lo_p_bs_updated = self.tr_is(
                -1.0, lo_q_bs, lo_qD_bs, lo_n_bs, lo_P_mids, lo_p_bs, intervention=True
            )

            # Update state variables
            q_bs_updated[lo_is_bids_mask] = lo_q_bs_updated
            qD_bs_updated[lo_is_bids_mask] = lo_qD_bs_updated
            n_bs_updated[lo_is_bids_mask] = lo_n_bs_updated
            P_mids_updated[lo_is_bids_mask] = lo_P_mids_updated
            p_bs_updated[lo_is_bids_mask] = lo_p_bs_updated

        # Build updated state tensor
        Ss_intervened = torch.stack([
            Xs_updated, Ys_updated, p_as_updated, p_bs_updated, q_as_updated, q_bs_updated,
            qD_as_updated, qD_bs_updated, n_as_updated, n_bs_updated, P_mids_updated
        ], dim=1)
        Ss_intervened.to(self.device)
        inter_profit.to(self.device)
        # Calculate new value function and add profit
        return model_phi(ts, Ss_intervened) #+ inter_profit.unsqueeze(1).to(self.device)

    def oracle_u(self, model_phi, ts, Ss):
        # Use PyTorch operations to compute the best action
        batch_size = ts.shape[0]

        # Initialize tensor to store intervention values for each action
        interventions = torch.zeros((batch_size, len(self.U)), dtype=torch.float32, device=self.device)

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

        # Return the action with highest value
        return torch.argmax(interventions, dim=1).to(self.device)

    def oracle_d(self, model_phi, model_u, ts, Ss):
        batch_size = ts.shape[0]
        lambdas = self.lambdas_poisson
        # Initialize tensor to store HJB values for each decision
        hjb = torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device)

        for d in [0, 1]:
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
            ds = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device) * d
            us, _ = model_u(ts, Ss)

            # Calculate intervention value
            M_phi = self.intervention(model_phi, ts, Ss, us)

            # Calculate HJB evaluation
            evaluation = (1 - ds) * (L_phi + f) + ds * (M_phi - output)

            # Store in hjb tensor
            indices = torch.stack([torch.arange(batch_size, dtype=torch.int64),
                                   torch.ones(batch_size, dtype=torch.int64) * d], dim=1)
            hjb[indices[:, 0], indices[:, 1]] = evaluation.reshape(-1).to(self.device)

        # Return the decision with highest value
        return torch.argmax(hjb, dim=1).to(self.device)

    def loss_phi_poisson(self, model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas):
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
        ds, _ = model_d(ts, Ss)
        us, _ = model_u(ts, Ss)

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

    def train_step(self, model_phi, optimizer_phi, scheduler_phi, model_d, optimizer_d, scheduler_d, model_u, optimizer_u, scheduler_u, phi_epochs=10, phi_optim='ADAM'):
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
            loss_phi = torch.log(self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas))
            loss_phi.backward()
            return loss_phi
        # Use gradient clipping to prevent exploding gradients
        for j in range(phi_epochs):
            loss_phi = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas)
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

        # Train control function
        gt_u = self.oracle_u(model_phi, ts, Ss)
        train_loss_u = 0.0
        acc_u = 0.0
        loss_object_u = nn.CrossEntropyLoss()

        for j in range(100):
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
        # Train decision function
        gt_d = self.oracle_d(model_phi, model_u, ts, Ss)
        train_loss_d = 0.0
        acc_d = 0.0
        loss_object_d = nn.CrossEntropyLoss()

        for j in range(100):
            optimizer_d.zero_grad()
            pred_d, prob_ds = model_d(ts, Ss)
            print(np.unique(gt_d.cpu(), return_counts=True))
            print(np.unique(pred_d.cpu(), return_counts=True))
            acc_d = 100*torch.sum(pred_d.flatten() == gt_d).item() / len(pred_d)
            loss_d = loss_object_d(prob_ds, gt_d)
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
            print(f'Model d Acc: {acc_d:0.4f}')

        return model_phi, model_d, model_u, train_loss_phi, train_loss_d, train_loss_u, acc_u, acc_d

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
            model_u = DGM.DenseNet(feature_width, layer_widths[2], n_layers[2], 10, model0)
            model_d = DGM.DenseNet(feature_width, layer_widths[3], n_layers[3], 2, model0)
        else:
            model_phi = DGM.DGMNet(layer_widths[0], n_layers[0], 11, typeNN = typeNN)
            model_u = DGM.PIANet(layer_widths[1], n_layers[1], 11, 10, typeNN = typeNN2)
            model_d = DGM.PIANet(layer_widths[2], n_layers[2], 11, 2, typeNN = typeNN2)

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
            loaded_phi, loaded_d, loaded_u = model_manager.load_models(model_phi, model_d, model_u, timestamp= checkpoint_label, epoch = continue_epoch)
            if loaded_phi is not None:
                model_phi, model_d, model_u = loaded_phi, loaded_d, loaded_u
                print("Resuming training from previously saved models")
        model_phi.to(device)
        model_u.to(device)
        model_d.to(device)
        if unified: model0.to(device)
        # Define lambda function for the scheduler
        def lr_lambda(epoch):
            # Calculate decay rate
            decay_rate = np.log(1e-4) / (self.EPOCHS - 1)
            return np.max([1e-5,np.exp(decay_rate * epoch )])
        if phi_optim == 'ADAM':
            optimizer_phi = optim.Adam(model_phi.parameters(), lr=lr)
        else:
            optimizer_phi = optim.LBFGS(model_phi.parameters(), lr=lr*10000)
        optimizer_u = optim.Adam(model_u.parameters(), lr=lr)
        optimizer_d = optim.Adam(model_d.parameters(), lr=lr)

        # Set up schedulers
        scheduler_phi = optim.lr_scheduler.LambdaLR(optimizer_phi, lr_lambda)
        scheduler_u = optim.lr_scheduler.LambdaLR(optimizer_u, lr_lambda)
        scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda)

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
            model_phi, model_d, model_u, loss_phi, loss_d, loss_u, acc_u, acc_d = self.train_step(
                model_phi, optimizer_phi, scheduler_phi, model_d, optimizer_d, scheduler_d, model_u, optimizer_u, scheduler_u, phi_epochs = phi_epochs, phi_optim = phi_optim
            )

            # Log losses
            logger.log_losses(loss_phi, loss_d, loss_u, acc_d, acc_u)

            # Save checkpoint at specified frequency
            if (epoch + 1) % checkpoint_frequency == 0:
                print(f"Saving checkpoint at epoch {epoch+1}")
                model_manager.save_models(model_phi, model_d, model_u, epoch=epoch+1)
                # Save and plot losses
                logger.save_logs()
                logger.plot_losses(show=True, save=True)
            # if (epoch+1) == refresh_epoch:
            #     model_u = DGM.PIANet(layer_widths[1], n_layers[1], 11, 10, typeNN = typeNN)
            #     model_d = DGM.PIANet(layer_widths[2], n_layers[2], 11, 2, typeNN = typeNN)
            # Step learning rate schedulers

            # Print epoch summary
            print(f"Epoch {epoch+1} summary - Phi Loss: {loss_phi:.4f}, D Loss: {loss_d:.4f}, U Loss: {loss_u:.4f}")

            if (epoch+1) == continue_epoch + 1000:
                print('Switching to LBFGS for Phi')
                optimizer_phi = optim.LBFGS(model_phi.parameters())
                # Set up schedulers
                scheduler_phi = optim.lr_scheduler.LambdaLR(optimizer_phi, lr_lambda)
                phi_optim = 'LBFGS'
        # Save final model
        model_manager.save_models(model_phi, model_d, model_u)

        print("Training completed. Models saved and losses plotted.")

        return model_phi, model_d, model_u

# get_gpu_specs()
# MM = MarketMaking(num_epochs=2000, num_points=100)
# MM.train(sampler='iid',log_dir = 'logs', model_dir = 'models', typeNN='LSTM', layer_widths = [20]*4, n_layers= [2]*4, unified=True, label = 'LSTM')