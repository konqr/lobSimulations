import numpy as np
import tensorflow as tf
import DGM
tf.experimental.numpy.experimental_enable_numpy_behavior()

class MarketMaking():

    def __init__(self, num_points=100, num_epochs=1000):
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
        self.TERMINATION_TIME = 23400
        self.NDIMS = 12
        self.NUM_POINTS = num_points
        self.EPOCHS = num_epochs
        self.eta = 0.5  # inventory penalty
        self.E = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
                  "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
        self.U = ["lo_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
                  "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "lo_deep_Bid"]
        self.lambdas_poisson = [.86, .32, .33, .48, .02, .47, .47, .02, .48, .33, .32, .86] # [5] * 12

    def sampler(self, num_points=1000, seed=None, boundary=False):
        '''
        Sample points from the stationary distributions for the DGM learning
        :param num_points: number of points
        :return: samples of [0,T] x {state space}
        '''
        if seed:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # Generate sample data using NumPy
        Xs = np.round(1e3 * np.random.randn(num_points, 1), 2)
        Ys = np.round(10 * np.random.randn(num_points, 1), 2)
        P_mids = np.round(200 + 10 * np.random.randn(num_points, 1), 2) / 2
        spreads = 0.01 * np.random.geometric(.8, [num_points, 1])
        p_as = np.round(P_mids + spreads / 2, 2)
        p_bs = np.round(P_mids - spreads / 2, 2)
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
            return tf.convert_to_tensor(t_boundary, dtype=tf.float32), tf.convert_to_tensor(
                np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]), dtype=tf.float32)

        return tf.convert_to_tensor(t, dtype=tf.float32), tf.convert_to_tensor(
            np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]), dtype=tf.float32)

    def sample_qd(self):
        # Return a TensorFlow tensor directly
        return tf.constant(np.random.geometric(.0015, 1), dtype=tf.float32)

    def tr_lo_deep(self, qD):
        # TensorFlow operation
        return qD + 1.0

    def tr_co_deep(self, qD, n, q):
        # TensorFlow operations
        qD_updated = qD - 1.0

        # Replace conditional operations with smooth TensorFlow operations
        qD_updated = tf.maximum(qD_updated, 0.0)  # Ensure qD doesn't go negative

        # Handle the agent's orders case
        condition = tf.logical_and(tf.equal(qD_updated, 1.0), tf.equal(n, q))
        qD_updated = tf.where(condition, qD_updated + 1.0, qD_updated)

        return qD_updated

    def tr_lo_top(self, q_as, n_as):
        # TensorFlow operations
        q_as_updated = q_as + 1.0

        # Adjust priority for deep orders
        condition = tf.greater_equal(n_as, q_as_updated)
        n_as_updated = tf.where(condition, n_as + 1.0, n_as)

        return q_as_updated, n_as_updated

    def tr_co_top(self, z, q_as, n_as, qD_as, p_as, P_mids, intervention=False):
        if intervention:
            # For intervention, directly update values
            q_as_updated = q_as - 1.0
            # This is a simplification; in real code, you might need a differentiable alternative
            n_as_updated = tf.random.uniform(tf.shape(n_as), minval=0, maxval=q_as_updated + qD_as + 1, dtype=tf.float32)
        else:
            # For normal operation
            idxCO = tf.random.uniform(tf.shape(q_as), minval=0, maxval=q_as, dtype=tf.float32)
            q_as_updated = q_as - 1.0

            # Handle cases where we are cancelling agent's orders
            condition = tf.equal(idxCO, n_as)
            qD_as_updated = tf.where(condition, qD_as + 1.0, qD_as)

            # Handle cases where n_as > idxCO
            condition = tf.greater(n_as, idxCO)
            n_as_updated = tf.where(condition, n_as - 1.0, n_as)

        # Handle queue depletion
        condition = tf.equal(q_as_updated, 0.0)
        q_as_final = tf.where(condition, qD_as, q_as_updated)
        qD_as_final = tf.where(condition, self.sample_qd(), qD_as)
        p_as_final = tf.where(condition, p_as + z * 0.01, p_as)
        P_mids_final = tf.where(condition, P_mids + z * 0.005, P_mids)

        return q_as_final, n_as_updated, qD_as_final, p_as_final, P_mids_final

    def tr_mo(self, z, q_as, n_as, qD_as, p_as, P_mids, Xs, Ys, intervention=False):
        # Agent fill condition
        agent_fill_condition = tf.equal(n_as, 0.0)

        if not intervention:
            # Update X and Y when agent's order is filled
            Xs_updated = tf.where(agent_fill_condition, Xs + z * p_as, Xs)
            Ys_updated = tf.where(agent_fill_condition, Ys - z * 1.0, Ys)

            # Update agent position
            # For a differentiable version, we could use a smoothed random update
            n_as_random = tf.random.uniform(tf.shape(n_as), minval=1.0, maxval=q_as + qD_as, dtype=tf.float32)
            n_as_updated = tf.where(agent_fill_condition, n_as_random, n_as)
        else:
            # No effect during intervention for agent fills
            q_as_intervention = tf.where(agent_fill_condition, q_as + 1.0, q_as)
            n_as_intervention = tf.where(agent_fill_condition, n_as + 1.0, n_as)
            q_as = q_as_intervention
            n_as = n_as_intervention
            Xs_updated = Xs
            Ys_updated = Ys

        # Standard updates
        q_as_updated = q_as - 1.0
        n_as_final = n_as - 1.0

        # Handle queue depletion
        depletion_condition = tf.equal(q_as_updated, 0.0)
        q_as_final = tf.where(depletion_condition, qD_as, q_as_updated)
        qD_as_final = tf.where(depletion_condition, self.sample_qd(), qD_as)
        p_as_final = tf.where(depletion_condition, p_as + z * 0.01, p_as)
        P_mids_final = tf.where(depletion_condition, P_mids + z * 0.005, P_mids)

        if intervention:
            # Update X and Y for intervention
            Xs_updated = Xs + z * p_as
            Ys_updated = Ys - z * 1.0

        return q_as_final, n_as_final, qD_as_final, p_as_final, P_mids_final, Xs_updated, Ys_updated

    def tr_is(self, z, q_as, qD_as, n_as, P_mids, p_as, intervention=False):
        # TensorFlow operations
        qD_as_updated = q_as  # Copy q_as to qD_as
        q_as_updated = tf.ones_like(q_as)  # Set q_as to ones

        if intervention:
            n_as_updated = tf.zeros_like(n_as)
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

        # Use tf.case instead of if-elif chain for differentiability
        # Define the operations for each event case
        def case_0():
            qD_as_updated = self.tr_lo_deep(qD_as)
            return tf.stack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as_updated, qD_bs, n_as, n_bs, P_mids], axis=1)

        def case_1():
            qD_as_updated = self.tr_co_deep(qD_as, n_as, q_as)
            return tf.stack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as_updated, qD_bs, n_as, n_bs, P_mids], axis=1)

        def case_2():
            q_as_updated, n_as_updated = self.tr_lo_top(q_as, n_as)
            return tf.stack([Xs, Ys, p_as, p_bs, q_as_updated, q_bs, qD_as, qD_bs, n_as_updated, n_bs, P_mids], axis=1)

        def case_3():
            q_as_updated, n_as_updated, qD_as_updated, p_as_updated, P_mids_updated = self.tr_co_top(1.0, q_as, n_as, qD_as, p_as, P_mids)
            return tf.stack([Xs, Ys, p_as_updated, p_bs, q_as_updated, q_bs, qD_as_updated, qD_bs, n_as_updated, n_bs, P_mids_updated], axis=1)

        def case_4():
            q_as_updated, n_as_updated, qD_as_updated, p_as_updated, P_mids_updated, Xs_updated, Ys_updated = self.tr_mo(
                1.0, q_as, n_as, qD_as, p_as, P_mids, Xs, Ys)
            return tf.stack([Xs_updated, Ys_updated, p_as_updated, p_bs, q_as_updated, q_bs, qD_as_updated, qD_bs, n_as_updated, n_bs, P_mids_updated], axis=1)

        def case_5():
            q_as_updated, qD_as_updated, n_as_updated, P_mids_updated, p_as_updated = self.tr_is(1.0, q_as, qD_as, n_as, P_mids, p_as)
            return tf.stack([Xs, Ys, p_as_updated, p_bs, q_as_updated, q_bs, qD_as_updated, qD_bs, n_as_updated, n_bs, P_mids_updated], axis=1)

        def case_6():
            q_bs_updated, qD_bs_updated, n_bs_updated, P_mids_updated, p_bs_updated = self.tr_is(-1.0, q_bs, qD_bs, n_bs, P_mids, p_bs)
            return tf.stack([Xs, Ys, p_as, p_bs_updated, q_as, q_bs_updated, qD_as, qD_bs_updated, n_as, n_bs_updated, P_mids_updated], axis=1)

        def case_7():
            q_bs_updated, n_bs_updated, qD_bs_updated, p_bs_updated, P_mids_updated, Xs_updated, Ys_updated = self.tr_mo(
                -1.0, q_bs, n_bs, qD_bs, p_bs, P_mids, Xs, Ys)
            return tf.stack([Xs_updated, Ys_updated, p_as, p_bs_updated, q_as, q_bs_updated, qD_as, qD_bs_updated, n_as, n_bs_updated, P_mids_updated], axis=1)

        def case_8():
            q_bs_updated, n_bs_updated, qD_bs_updated, p_bs_updated, P_mids_updated = self.tr_co_top(-1.0, q_bs, n_bs, qD_bs, p_bs, P_mids)
            return tf.stack([Xs, Ys, p_as, p_bs_updated, q_as, q_bs_updated, qD_as, qD_bs_updated, n_as, n_bs_updated, P_mids_updated], axis=1)

        def case_9():
            q_bs_updated, n_bs_updated = self.tr_lo_top(q_bs, n_bs)
            return tf.stack([Xs, Ys, p_as, p_bs, q_as, q_bs_updated, qD_as, qD_bs, n_as, n_bs_updated, P_mids], axis=1)

        def case_10():
            qD_bs_updated = self.tr_co_deep(qD_bs, n_bs, q_bs)
            return tf.stack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs_updated, n_as, n_bs, P_mids], axis=1)

        def case_11():
            qD_bs_updated = self.tr_lo_deep(qD_bs)
            return tf.stack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs_updated, n_as, n_bs, P_mids], axis=1)

        def default():
            # Default case - return original state
            return Ss

        # Create a dictionary of cases for the tf.case operation
        case_dict = [
            (tf.equal(eventID, 0), case_0),
            (tf.equal(eventID, 1), case_1),
            (tf.equal(eventID, 2), case_2),
            (tf.equal(eventID, 3), case_3),
            (tf.equal(eventID, 4), case_4),
            (tf.equal(eventID, 5), case_5),
            (tf.equal(eventID, 6), case_6),
            (tf.equal(eventID, 7), case_7),
            (tf.equal(eventID, 8), case_8),
            (tf.equal(eventID, 9), case_9),
            (tf.equal(eventID, 10), case_10),
            (tf.equal(eventID, 11), case_11)
        ]

        # Use tf.case to select the appropriate operation based on eventID
        return tf.case(case_dict, default=default, exclusive=True)

    def intervention(self, model_phi, ts, Ss, us):
        batch_size = tf.shape(ts)[0]

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
        Xs_updated = Xs
        Ys_updated = Ys
        p_as_updated = p_as
        p_bs_updated = p_bs
        q_as_updated = q_as
        q_bs_updated = q_bs
        qD_as_updated = qD_as
        qD_bs_updated = qD_bs
        n_as_updated = n_as
        n_bs_updated = n_bs
        P_mids_updated = P_mids

        # Initialize profit tensor
        inter_profit = tf.zeros(len(ts))

        # Handle different intervention types using TensorFlow operations

        # Market order asks
        mo_asks_mask = tf.equal(us, 3)
        if tf.reduce_any(mo_asks_mask):
            # Apply market order ask intervention
            idxs = tf.where(mo_asks_mask)[:,0]

            mo_q_as = tf.gather(q_as, idxs)
            mo_n_as = tf.gather(n_as, idxs)
            mo_qD_as = tf.gather(qD_as, idxs)
            mo_p_as = tf.gather(p_as, idxs)
            mo_P_mids = tf.gather(P_mids, idxs)
            mo_Xs = tf.gather(Xs, idxs)
            mo_Ys = tf.gather(Ys, idxs)

            # Apply market order intervention
            mo_q_as_updated, mo_n_as_updated, mo_qD_as_updated, mo_p_as_updated, mo_P_mids_updated, mo_Xs_updated, mo_Ys_updated = self.tr_mo(
                1.0, mo_q_as, mo_n_as, mo_qD_as, mo_p_as, mo_P_mids, mo_Xs, mo_Ys, intervention=True
            )

            # Update state variables
            idxs = tf.reshape(idxs,(len(idxs),1))
            q_as_updated = tf.tensor_scatter_nd_update(q_as_updated, idxs, mo_q_as_updated)
            n_as_updated = tf.tensor_scatter_nd_update(n_as_updated, idxs, mo_n_as_updated)
            qD_as_updated = tf.tensor_scatter_nd_update(qD_as_updated, idxs, mo_qD_as_updated)
            p_as_updated = tf.tensor_scatter_nd_update(p_as_updated, idxs, mo_p_as_updated)
            P_mids_updated = tf.tensor_scatter_nd_update(P_mids_updated, idxs, mo_P_mids_updated)
            Xs_updated = tf.tensor_scatter_nd_update(Xs_updated, idxs, mo_Xs_updated)
            Ys_updated = tf.tensor_scatter_nd_update(Ys_updated, idxs, mo_Ys_updated)

            # Update profit
            profit_updates = tf.reshape(mo_p_as, [-1])
            inter_profit = tf.tensor_scatter_nd_update(inter_profit, idxs, profit_updates)

        # Market order bids
        mo_bids_mask = tf.equal(us, 6)
        if tf.reduce_any(mo_bids_mask):
            # Apply market order bid intervention
            idxs = tf.where(mo_bids_mask)[:,0]

            mo_q_bs = tf.gather(q_bs, idxs)
            mo_n_bs = tf.gather(n_bs, idxs)
            mo_qD_bs = tf.gather(qD_bs, idxs)
            mo_p_bs = tf.gather(p_bs, idxs)
            mo_P_mids = tf.gather(P_mids, idxs)
            mo_Xs = tf.gather(Xs, idxs)
            mo_Ys = tf.gather(Ys, idxs)

            # Apply market order intervention
            mo_q_bs_updated, mo_n_bs_updated, mo_qD_bs_updated, mo_p_bs_updated, mo_P_mids_updated, mo_Xs_updated, mo_Ys_updated = self.tr_mo(
                -1.0, mo_q_bs, mo_n_bs, mo_qD_bs, mo_p_bs, mo_P_mids, mo_Xs, mo_Ys, intervention=True
            )

            # Update state variables
            idxs = tf.reshape(idxs,(len(idxs),1))
            q_bs_updated = tf.tensor_scatter_nd_update(q_bs_updated, idxs, mo_q_bs_updated)
            n_bs_updated = tf.tensor_scatter_nd_update(n_bs_updated, idxs, mo_n_bs_updated)
            qD_bs_updated = tf.tensor_scatter_nd_update(qD_bs_updated, idxs, mo_qD_bs_updated)
            p_bs_updated = tf.tensor_scatter_nd_update(p_bs_updated, idxs, mo_p_bs_updated)
            P_mids_updated = tf.tensor_scatter_nd_update(P_mids_updated, idxs, mo_P_mids_updated)
            Xs_updated = tf.tensor_scatter_nd_update(Xs_updated, idxs, mo_Xs_updated)
            Ys_updated = tf.tensor_scatter_nd_update(Ys_updated, idxs, mo_Ys_updated)

            # Update profit (negative because we're paying to buy)
            profit_updates = tf.reshape(-1.0 * mo_p_bs, [-1])
            inter_profit = tf.tensor_scatter_nd_update(inter_profit, idxs, profit_updates)

        # Limit order deep asks
        lo_deep_asks_mask = tf.equal(us, 0)
        if tf.reduce_any(lo_deep_asks_mask):
            idxs = tf.where(lo_deep_asks_mask)[:,0]
            lo_qD_as = tf.gather(qD_as, idxs)
            idxs = tf.reshape(idxs,(len(idxs),1))
            qD_as_updated = tf.tensor_scatter_nd_update(qD_as_updated, idxs, lo_qD_as + 1.0)

        # Limit order deep bids
        lo_deep_bids_mask = tf.equal(us, 9)
        if tf.reduce_any(lo_deep_bids_mask):
            idxs = tf.where(lo_deep_bids_mask)[:,0]

            lo_qD_bs = tf.gather(qD_bs, idxs)
            idxs = tf.reshape(idxs,(len(idxs),1))
            qD_bs_updated = tf.tensor_scatter_nd_update(qD_bs_updated, idxs, lo_qD_bs + 1.0)

        # Limit order top asks
        lo_top_asks_mask = tf.equal(us, 1)
        if tf.reduce_any(lo_top_asks_mask):
            idxs = tf.where(lo_top_asks_mask)[:,0]

            lo_q_as = tf.gather(q_as, idxs)
            lo_n_as = tf.gather(n_as, idxs)

            # Update q_as
            lo_q_as_updated = lo_q_as + 1.0
            idxs = tf.reshape(idxs,(len(idxs),1))
            q_as_updated = tf.tensor_scatter_nd_update(q_as_updated, idxs, lo_q_as_updated)

            # Update n_as if necessary
            condition = tf.greater_equal(lo_n_as, lo_q_as_updated)
            lo_n_as_updated = tf.where(condition, lo_q_as_updated, lo_n_as)
            n_as_updated = tf.tensor_scatter_nd_update(n_as_updated, idxs, lo_n_as_updated)

        # Limit order top bids
        lo_top_bids_mask = tf.equal(us, 8)
        if tf.reduce_any(lo_top_bids_mask):
            idxs = tf.where(lo_top_bids_mask)[:,0]

            lo_q_bs = tf.gather(q_bs, idxs)
            lo_n_bs = tf.gather(n_bs, idxs)

            # Update q_bs
            lo_q_bs_updated = lo_q_bs + 1.0
            idxs = tf.reshape(idxs,(len(idxs),1))
            q_bs_updated = tf.tensor_scatter_nd_update(q_bs_updated, idxs, lo_q_bs_updated)

            # Update n_bs if necessary
            condition = tf.greater_equal(lo_n_bs, lo_q_bs_updated)
            lo_n_bs_updated = tf.where(condition, lo_q_bs_updated, lo_n_bs)
            n_bs_updated = tf.tensor_scatter_nd_update(n_bs_updated, idxs, lo_n_bs_updated)

        # Cancel order top asks
        co_top_asks_mask = tf.equal(us, 2)
        if tf.reduce_any(co_top_asks_mask):
            idxs = tf.where(co_top_asks_mask)[:,0]

            co_q_as = tf.gather(q_as, idxs)
            co_n_as = tf.gather(n_as, idxs)
            co_qD_as = tf.gather(qD_as, idxs)
            co_p_as = tf.gather(p_as, idxs)
            co_P_mids = tf.gather(P_mids, idxs)

            # Apply cancel order intervention
            co_q_as_updated, co_n_as_updated, co_qD_as_updated, co_p_as_updated, co_P_mids_updated = self.tr_co_top(
                1.0, co_q_as, co_n_as, co_qD_as, co_p_as, co_P_mids, intervention=True
            )

            # Update state variables
            idxs = tf.reshape(idxs,(len(idxs),1))
            q_as_updated = tf.tensor_scatter_nd_update(q_as_updated, idxs, co_q_as_updated)
            n_as_updated = tf.tensor_scatter_nd_update(n_as_updated, idxs, co_n_as_updated)
            qD_as_updated = tf.tensor_scatter_nd_update(qD_as_updated, idxs, co_qD_as_updated)
            p_as_updated = tf.tensor_scatter_nd_update(p_as_updated, idxs, co_p_as_updated)
            P_mids_updated = tf.tensor_scatter_nd_update(P_mids_updated, idxs, co_P_mids_updated)

        # Cancel order top bids
        co_top_bids_mask = tf.equal(us, 7)
        if tf.reduce_any(co_top_bids_mask):
            idxs = tf.where(co_top_bids_mask)[:,0]

            co_q_bs = tf.gather(q_bs, idxs)
            co_n_bs = tf.gather(n_bs, idxs)
            co_qD_bs = tf.gather(qD_bs, idxs)
            co_p_bs = tf.gather(p_bs, idxs)
            co_P_mids = tf.gather(P_mids, idxs)

            # Apply cancel order intervention
            co_q_bs_updated, co_n_bs_updated, co_qD_bs_updated, co_p_bs_updated, co_P_mids_updated = self.tr_co_top(
                -1.0, co_q_bs, co_n_bs, co_qD_bs, co_p_bs, co_P_mids, intervention=True
            )

            # Update state variables
            idxs = tf.reshape(idxs,(len(idxs),1))
            q_bs_updated = tf.tensor_scatter_nd_update(q_bs_updated, idxs, co_q_bs_updated)
            n_bs_updated = tf.tensor_scatter_nd_update(n_bs_updated, idxs, co_n_bs_updated)
            qD_bs_updated = tf.tensor_scatter_nd_update(qD_bs_updated, idxs, co_qD_bs_updated)
            p_bs_updated = tf.tensor_scatter_nd_update(p_bs_updated, idxs, co_p_bs_updated)
            P_mids_updated = tf.tensor_scatter_nd_update(P_mids_updated, idxs, co_P_mids_updated)

        # Limit order in-spread asks
        lo_is_asks_mask = tf.equal(us, 4)
        if tf.reduce_any(lo_is_asks_mask):
            idxs = tf.where(lo_is_asks_mask)[:,0]

            lo_q_as = tf.gather(q_as, idxs)
            lo_n_as = tf.gather(n_as, idxs)
            lo_qD_as = tf.gather(qD_as, idxs)
            lo_p_as = tf.gather(p_as, idxs)
            lo_P_mids = tf.gather(P_mids, idxs)

            # Apply limit order in-spread intervention
            lo_q_as_updated, lo_qD_as_updated, lo_n_as_updated, lo_P_mids_updated, lo_p_as_updated = self.tr_is(
                1.0, lo_q_as, lo_qD_as, lo_n_as, lo_P_mids, lo_p_as, intervention=True
            )

            # Update state variables
            idxs = tf.reshape(idxs,(len(idxs),1))
            q_as_updated = tf.tensor_scatter_nd_update(q_as_updated, idxs, lo_q_as_updated)
            qD_as_updated = tf.tensor_scatter_nd_update(qD_as_updated, idxs, lo_qD_as_updated)
            n_as_updated = tf.tensor_scatter_nd_update(n_as_updated, idxs, lo_n_as_updated)
            P_mids_updated = tf.tensor_scatter_nd_update(P_mids_updated, idxs, lo_P_mids_updated)
            p_as_updated = tf.tensor_scatter_nd_update(p_as_updated, idxs, lo_p_as_updated)

        # Limit order in-spread bids
        lo_is_bids_mask = tf.equal(us, 5)
        if tf.reduce_any(lo_is_bids_mask):
            idxs = tf.where(lo_is_bids_mask)[:,0]

            lo_q_bs = tf.gather(q_bs, idxs)
            lo_n_bs = tf.gather(n_bs, idxs)
            lo_qD_bs = tf.gather(qD_bs, idxs)
            lo_p_bs = tf.gather(p_bs, idxs)
            lo_P_mids = tf.gather(P_mids, idxs)

            # Apply limit order in-spread intervention
            lo_q_bs_updated, lo_qD_bs_updated, lo_n_bs_updated, lo_P_mids_updated, lo_p_bs_updated = self.tr_is(
                -1.0, lo_q_bs, lo_qD_bs, lo_n_bs, lo_P_mids, lo_p_bs, intervention=True
            )

            # Update state variables
            idxs = tf.reshape(idxs,(len(idxs),1))
            q_bs_updated = tf.tensor_scatter_nd_update(q_bs_updated, idxs, lo_q_bs_updated)
            qD_bs_updated = tf.tensor_scatter_nd_update(qD_bs_updated, idxs, lo_qD_bs_updated)
            n_bs_updated = tf.tensor_scatter_nd_update(n_bs_updated, idxs, lo_n_bs_updated)
            P_mids_updated = tf.tensor_scatter_nd_update(P_mids_updated, idxs, lo_P_mids_updated)
            p_bs_updated = tf.tensor_scatter_nd_update(p_bs_updated, idxs, lo_p_bs_updated)

        # Build updated state tensor
        Ss_intervened = tf.stack([
            Xs_updated, Ys_updated, p_as_updated, p_bs_updated, q_as_updated, q_bs_updated,
            qD_as_updated, qD_bs_updated, n_as_updated, n_bs_updated, P_mids_updated
        ], axis=1)

        # Calculate new value function and add profit
        inter_profit = tf.reshape(inter_profit, [-1, 1])
        return model_phi(ts, Ss_intervened) #+ inter_profit

    def oracle_u(self, model_phi, ts, Ss):
        # Use TensorFlow operations to compute the best action
        batch_size = tf.shape(ts)[0]

        # Initialize tensor to store intervention values for each action
        interventions = tf.zeros((batch_size, len(self.U)), dtype=tf.float32)

        # Calculate value for each action
        for u in range(len(self.U)):
            # Create a tensor of action u for all batch items
            u_tensor = tf.ones((batch_size, 1), dtype=tf.float32) * u

            # Calculate intervention value
            intervention_value = self.intervention(model_phi, ts, Ss, u_tensor)

            # Store in interventions tensor
            interventions = tf.tensor_scatter_nd_update(
                interventions,
                tf.stack([tf.range(batch_size, dtype=tf.int32), tf.ones(batch_size, dtype=tf.int32) * u], axis=1),
                tf.reshape(intervention_value, [-1])
            )

        # Return the action with highest value
        return tf.argmax(interventions, axis=1)

    def oracle_d(self, model_phi, model_u, ts, Ss):
        batch_size = tf.shape(ts)[0]

        # Initialize tensor to store HJB values for each decision
        hjb = tf.zeros((batch_size, 2), dtype=tf.float32)

        for d in [0, 1]:
            # Use GradientTape to calculate time derivative
            with tf.GradientTape() as tape:
                tape.watch(ts)
                output = model_phi(ts, Ss)

            phi_t = tape.gradient(output, ts)

            # Calculate integral term
            I_phi = tf.zeros_like(output)
            for i in range(self.NDIMS):
                # Calculate transition for event i
                Ss_transitioned = self.transition(Ss, tf.constant(i, dtype=tf.float32))
                I_phi += self.lambdas_poisson[i] * (model_phi(ts, Ss_transitioned) - output)

            # Calculate generator term
            L_phi = phi_t + I_phi

            # Calculate running cost
            f = -self.eta * tf.square(Ss[:, 1])  # Inventory penalty
            f = tf.reshape(f, [-1, 1])

            # Decision tensor
            ds = tf.ones((batch_size, 1), dtype=tf.float32) * d

            # Get optimal control
            us, _ = model_u(ts, Ss)

            # Calculate intervention value
            M_phi = self.intervention(model_phi, ts, Ss, us)

            # Calculate HJB value
            evaluation = (1 - ds) * (L_phi + f) + ds * (M_phi - output)

            # Store in hjb tensor
            hjb = tf.tensor_scatter_nd_update(
                hjb,
                tf.stack([tf.range(batch_size, dtype=tf.int32), tf.ones(batch_size, dtype=tf.int32) * d], axis=1),
                tf.reshape(evaluation, [-1])
            )

        # Return the decision with highest value
        return tf.argmax(hjb, axis=1)

    def loss_phi_poisson(self, model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas):
        # Use GradientTape to calculate time derivative
        with tf.GradientTape() as tape:
            tape.watch(ts)
            output = model_phi(ts, Ss)

        phi_t = tape.gradient(output, ts)

        # Calculate integral term
        I_phi = tf.zeros_like(output)
        for i in range(self.NDIMS):
            # Calculate transition for event i
            Ss_transitioned = self.transition(Ss, tf.constant(i, dtype=tf.float32))
            I_phi += lambdas[i] * (model_phi(ts, Ss_transitioned) - output)

        # Calculate generator term
        L_phi = phi_t + I_phi

        # Calculate running cost
        f = -self.eta * tf.square(Ss[:, 1])  # Inventory penalty
        f = tf.reshape(f, [-1, 1])

        # Get optimal decision and control
        ds, _ = model_d(ts, Ss)
        us, _ = model_u(ts, Ss)

        # Calculate intervention value
        M_phi = self.intervention(model_phi, ts, Ss, us)

        # Calculate HJB evaluation
        evaluation = (1 - ds) * (L_phi + f) + ds * (M_phi - output)

        # Interior loss
        interior_loss = tf.keras.losses.MeanSquaredError()(evaluation, tf.zeros_like(evaluation))

        # Boundary loss
        output_boundary = model_phi(Ts, S_boundarys)
        g = S_boundarys[:, 0] + S_boundarys[:, 1] * S_boundarys[:, -1]  # Terminal condition
        g = tf.reshape(g, [-1, 1])
        boundary_loss = tf.keras.losses.MeanSquaredError()(output_boundary, g)

        # Combine losses with potential weighting
        loss = interior_loss + boundary_loss

        return loss

    def train_step(self, model_phi, optimizer_phi, model_d, optimizer_d, model_u, optimizer_u):
        # Setup for training
        lambdas = self.lambdas_poisson

        # Generate sample data
        ts, Ss = self.sampler(self.NUM_POINTS)
        Ts, S_boundarys = self.sampler(self.NUM_POINTS, boundary=True)

        # Learning rate schedule

        # Train value function

        train_loss_phi = tf.keras.metrics.Mean(name='train_loss_phi')

        # Use gradient clipping to prevent exploding gradients
        for j in range(10):
            with tf.GradientTape() as tape:
                loss_phi = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas)

            gradients = tape.gradient(loss_phi, model_phi.trainable_variables)
            # Clip gradients to prevent exploding values
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer_phi.apply_gradients(zip(gradients, model_phi.trainable_variables))
            train_loss_phi(loss_phi)

            # Check for NaN or Inf
            if tf.math.is_nan(loss_phi) or tf.math.is_inf(loss_phi):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model Phi loss: {train_loss_phi.result():0.4f}')

        # Train control function
        gt_u = self.oracle_u(model_phi, ts, Ss)
        train_loss_u = tf.keras.metrics.Mean(name='train_loss_u')
        loss_object_u = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for j in range(10):
            with tf.GradientTape() as tape:
                _, prob_us = model_u(ts, Ss)
                loss_u = loss_object_u(gt_u, prob_us)

            gradients = tape.gradient(loss_u, model_u.trainable_variables)
            # Clip gradients to prevent exploding values
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer_u.apply_gradients(zip(gradients, model_u.trainable_variables))
            train_loss_u(loss_u)

            # Check for NaN or Inf
            if tf.math.is_nan(loss_u) or tf.math.is_inf(loss_u):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model u loss: {train_loss_u.result():0.4f}')

        # Train decision function
        gt_d = self.oracle_d(model_phi, model_u, ts, Ss)
        train_loss_d = tf.keras.metrics.Mean(name='train_loss_d')
        loss_object_d = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for j in range(10):
            with tf.GradientTape() as tape:
                _, prob_ds = model_d(ts, Ss)
                loss_d = loss_object_d(gt_d, prob_ds)

            gradients = tape.gradient(loss_d, model_d.trainable_variables)
            # Clip gradients to prevent exploding values
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer_d.apply_gradients(zip(gradients, model_d.trainable_variables))
            train_loss_d(loss_d)

            # Check for NaN or Inf
            if tf.math.is_nan(loss_d) or tf.math.is_inf(loss_d):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model d loss: {train_loss_d.result():0.4f}')

        return model_phi, model_d, model_u

    def train(self):
        # Create models
        model_phi = DGM.DGMNet(20, 5, 11)
        model_u = DGM.PIANet(20, 5, 11, 10)
        model_d = DGM.PIANet(20, 5, 11, 2)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,  # Lower initial rate for stability
            decay_steps=self.EPOCHS * 10,
            decay_rate=0.9)
        optimizer_phi = tf.keras.optimizers.Adam(lr_schedule)
        optimizer_u = tf.keras.optimizers.Adam(lr_schedule)
        optimizer_d = tf.keras.optimizers.Adam(lr_schedule)
        for epoch in range(self.EPOCHS):
            model_phi, model_d, model_u = self.train_step(model_phi, optimizer_phi, model_d, optimizer_d, model_u, optimizer_u)
        return model_phi, model_d, model_u

MM = MarketMaking(num_points=10000)
MM.train()