import numpy as np
import tensorflow as tf
import DGM
tf.experimental.numpy.experimental_enable_numpy_behavior()
#tf.compat.v1.disable_eager_execution()

class MarketMaking():

    def __init__(self, num_points = 100, num_epochs = 1000):
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
        self.eta = 0.5 # inventory penalty
        self.E = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                  "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        self.U = ["lo_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                  "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "lo_deep_Bid" ]
        self.lambdas_poisson = [5]*12

        return

    def sampler(self, num_points=1000, seed=None, boundary=False):
        '''
        Sample points from the stationary distributions for the DGM learning
        :param num_points: number of points
        :return: samples of [0,T] x {state space}
        '''
        if seed: np.random.seed(seed)
        Xs = np.round(1e3*np.random.randn(num_points,1), 2)
        Ys = np.round(10*np.random.randn(num_points,1), 2)
        P_mids = np.round(200+10*np.random.randn(num_points,1), 2)/2
        spreads = 0.01*np.random.geometric(.8, [num_points,1])
        p_as = np.round(P_mids + spreads/2, 2)
        p_bs = np.round(P_mids - spreads/2, 2)
        q_as = np.random.geometric(.002, [num_points,1])
        qD_as = np.random.geometric(.0015, [num_points,1])
        q_bs = np.random.geometric(.002, [num_points,1])
        qD_bs = np.random.geometric(.0015, [num_points,1])
        n_as = np.array([np.random.randint(0, b) for b in q_as + qD_as])
        n_bs = np.array([np.random.randint(0, b) for b in q_bs + qD_bs])
        # n2_as = np.random.randint(n_as, q_as + qD_as)
        # n2_bs = np.random.randint(n_bs, q_bs + qD_bs)

        t = np.random.uniform(0, self.TERMINATION_TIME, [num_points,1])
        t_boundary = self.TERMINATION_TIME*np.ones([num_points,1])
        if boundary:
            return t_boundary, np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids])
        return t, np.hstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]) #, n2_as, n2_bs)

    def sampleQD(self):
        return np.random.geometric(.0015, 1)

    def tr_lo_deep(selfs, qD):
        return qD+1

    def tr_co_deep(selfself, qD, n, q):
        qD -= 1
        negIdxs = np.where(qD < 0)[0]
        for i in negIdxs:
            qD[i] = 0 # reject cancel if no qty in queue
        agentsIdxs = np.where((qD == 1)&(n == q))[0]
        for i in agentsIdxs:
            qD[i] += 1 #reject cancel if canelling agent's orders
        return qD

    def tr_lo_top(self, q_as, n_as):
        q_as +=1
        deepOrdersIdxs = np.where(n_as >= q_as)[0]
        for i in deepOrdersIdxs:
            n_as[i] = n_as[i]+1 # deep orders get less priority
        return q_as, n_as

    def tr_co_top(self, z, q_as, n_as, qD_as, p_as, P_mids, intervention=False):

        if intervention:
            q_as -= 1
            n_as = np.random.randint(n_as, q_as+qD_as+1)
        else:
            idxCO = np.random.randint(0,q_as)
            q_as -= 1
            agentsIdxs = np.where(idxCO == n_as)[0]
            for i in agentsIdxs:
                qD_as[i] += 1 #reject cancel if canelling agent's orders
            agentsIdxs = np.where(n_as > idxCO)[0]
            for i in agentsIdxs:
                n_as[i] -= 1
        negIdxs = np.where(q_as == 0)[0] # Queue depletion
        for i in negIdxs:
            q_as[i] = qD_as[i].copy()
            qD_as[i] = self.sampleQD()
            p_as[i] += z*0.01
            P_mids[i] += z*0.005
        return q_as, n_as, qD_as, p_as, P_mids

    def tr_mo(self, z, q_as, n_as, qD_as, p_as, P_mids, Xs, Ys, intervention=False):
        agentIdxs = np.where(n_as == 0)[0] # agent fill
        for i in agentIdxs:
            if not intervention:
                Xs[i] += z*p_as[i]
                Ys[i] -= z*1
                n_as[i] = 1+ np.random.randint(n_as[i], q_as[i]+qD_as[i]) # random position afterwards... maybe better to default to -1 or q + qD
            else: #no effect
                q_as[i] +=1
                n_as[i] +=1
        q_as -= 1
        n_as -=1
        negIdxs = np.where(q_as == 0)[0] # Queue depletion
        for i in negIdxs:
            q_as[i] = qD_as[i].copy()
            qD_as[i] = self.sampleQD()
            p_as[i] += z*0.01
            P_mids[i] += z*0.005
        if intervention: # TODO: check if able to sell! if Y < 0, cant sell
            Xs += z*p_as
            Ys -= z
        return q_as, n_as, qD_as, p_as, P_mids, Xs, Ys

    def tr_is(self, z, q_as, qD_as, n_as, P_mids, p_as, intervention=False):
        qD_as = q_as.copy()
        q_as = np.ones(q_as.shape)
        if intervention:
            n_as = 0
        else:
            n_as += 1
        P_mids -= z*0.005
        p_as -= z*0.01
        return q_as, qD_as, n_as, P_mids, p_as

    def _numpy(self, x):
        res = []
        for i in x:
            res += [i.numpy()]
        return res

    def transition(self, Ss, eventID):
        Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids = Ss.transpose()
        Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids =self._numpy([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids])
        if eventID == 0 : # lo_deep_ask
            qD_as = self.tr_lo_deep(qD_as)
        elif eventID == 11: #lo_deep_bid
            qD_bs = self.tr_lo_deep(qD_as)
        elif eventID == 1: # co_deep_ask
            qD_as =self.tr_co_deep(qD_as, n_as, q_as)
        elif eventID == 10: # co_deep_bid
            qD_bs =self.tr_co_deep(qD_bs, n_bs, q_bs)
        elif eventID == 2: #lo_top_ask
            q_as, n_as = self.tr_lo_top(q_as, n_as)
        elif eventID == 9: #lo_top_bid
            q_bs, n_bs = self.tr_lo_top(q_bs, n_bs)
        elif eventID == 3: # co_top_ask
            q_as, n_as, qD_as, p_as, P_mids = self.tr_co_top(1, q_as, n_as, qD_as, p_as, P_mids)
        elif eventID == 8: # co_top_bid
            q_bs, n_bs, qD_bs, p_bs, P_mids = self.tr_co_top(-1, q_bs, n_bs, qD_bs, p_bs, P_mids)
        elif eventID == 4: #mo_ask
            q_as, n_as, qD_as, p_as, P_mids, Xs, Ys = self.tr_mo(1,q_as, n_as, qD_as, p_as, P_mids, Xs, Ys)
        elif eventID==7: #mo_bid
            q_bs, n_bs, qD_bs, p_bs, P_mids, Xs, Ys = self.tr_mo(-1,q_bs, n_bs, qD_bs, p_bs, P_mids, Xs, Ys)
        elif eventID == 5: #inspread_ask
            q_as, qD_as, n_as, P_mids, p_as = self.tr_is(1, q_as, qD_as, n_as, P_mids, p_as)
        elif eventID == 6: #inspread_bid
            q_bs, qD_bs, n_bs, P_mids, p_bs = self.tr_is(-1, q_bs, qD_bs, n_bs, P_mids, p_bs)
        else:
            raise Exception('Event ID is out of range')
        return np.vstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]).transpose()

    def intervention(self, model_phi, ts, Ss, us):
        Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids = Ss.transpose()
        Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids =self._numpy([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids])
        inter_profit = np.zeros([us.shape[0],1])
        mo_asks = np.where(us == 3)[0]
        for i in mo_asks:
            q, n,qD, p, P, x, y = np.array([q_as[i]]), np.array([n_as[i]]), np.array([qD_as[i]]), np.array([p_as[i]]), np.array([P_mids[i]]), np.array([Xs[i]]), np.array([Ys[i]])
            q_as[i], n_as[i], qD_as[i], p_as[i], P_mids[i], Xs[i], Ys[i] = self.tr_mo(1, q, n,qD, p, P, x, y, intervention=True)
            inter_profit[i] = p_as[i]
        mo_bids = np.where(us == 6)[0]
        for i in mo_bids:
            q, n,qD, p, P, x, y = np.array([q_bs[i]]), np.array([n_bs[i]]), np.array([qD_bs[i]]), np.array([p_bs[i]]), np.array([P_mids[i]]), np.array([Xs[i]]), np.array([Ys[i]])
            q_bs[i], n_bs[i], qD_bs[i], p_bs[i], P_mids[i], Xs[i], Ys[i] = self.tr_mo(-1, q, n,qD, p, P, x, y, intervention=True)
            inter_profit[i] = -1*p_bs[i]
        lo_deep_asks = np.where(us==0)[0]
        for i in lo_deep_asks:
            qD_as[i]+=1
        lo_deep_bids = np.where(us==9)[0]
        for i in lo_deep_bids:
            qD_bs[i]+=1
        lo_top_asks = np.where(us==1)[0]
        for i in lo_top_asks:
            q_as[i] +=1
            if n_as[i] >= q_as[i]:
                n_as[i] = q_as[i]
        lo_top_bids = np.where(us==8)[0]
        for i in lo_top_bids:
            q_bs[i] +=1
            if n_bs[i] >= q_bs[i]:
                n_bs[i] = q_bs[i]
        co_top_asks = np.where(us==2)[0]
        for i in co_top_asks:
            q, n,qD, p, P= np.array([q_as[i]]), np.array([n_as[i]]), np.array([qD_as[i]]), np.array([p_as[i]]), np.array([P_mids[i]])
            q_as[i], n_as[i], qD_as[i], p_as[i], P_mids[i] = self.tr_co_top(1, q, n,qD, p, P, intervention=True)
        co_top_bids = np.where(us == 7)[0]
        for i in co_top_bids:
            q, n,qD, p, P = np.array([q_bs[i]]), np.array([n_bs[i]]), np.array([qD_bs[i]]), np.array([p_bs[i]]), np.array([P_mids[i]])
            q_bs[i], n_bs[i], qD_bs[i], p_bs[i], P_mids[i] = self.tr_co_top(-1, q, n,qD, p, P, intervention=True)
        lo_is_asks = np.where(us==4)[0]
        for i in lo_is_asks:
            q, n,qD, p, P= np.array([q_as[i]]), np.array([n_as[i]]), np.array([qD_as[i]]), np.array([p_as[i]]), np.array([P_mids[i]])
            q_as[i], qD_as[i], n_as[i], P_mids[i], p_as[i] = self.tr_is(1, q, qD, n, P, p, intervention=True)
        lo_is_bids = np.where(us==5)[0]
        for i in lo_is_bids:
            q, n,qD, p, P= np.array([q_bs[i]]), np.array([n_bs[i]]), np.array([qD_bs[i]]), np.array([p_bs[i]]), np.array([P_mids[i]])
            q_bs[i], qD_bs[i], n_bs[i], P_mids[i], p_bs[i] = self.tr_is(-1, q, qD, n, P, p, intervention=True)
        Ss_intervened = np.vstack([Xs, Ys, p_as, p_bs, q_as, q_bs, qD_as, qD_bs, n_as, n_bs, P_mids]).transpose()
        return model_phi.call(ts, Ss_intervened) + tf.convert_to_tensor(inter_profit, dtype=tf.float32)

    def oracle_u(self, model_phi, ts, Ss):
        interventions = np.zeros((tf.shape(ts)[0], len(self.U)))
        for u in range(len(self.U)):
            interventions[:,u] = self.intervention(model_phi, ts, Ss, u*np.ones((tf.shape(ts)[0], 1))).flatten()
        return np.argmax(interventions, axis=1)

    def oracle_d(self, model_phi, model_u, ts, Ss):
        hjb = np.zeros((tf.shape(ts)[0], 2))
        for d in [0,1]:
            with tf.GradientTape() as tape:
                tape.watch(ts)
                output = model_phi.call(ts, Ss)
            phi_t = tape.gradient(output, ts)
            I_phi = 0
            for i in range(self.NDIMS):
                I_phi += self.lambdas_poisson[i]*(model_phi(ts, self.transition(Ss, i)) - output)
            L_phi = phi_t + I_phi
            f = -self.eta*Ss[:,1]**2 # Y_t
            ds = d*np.ones((tf.shape(ts)[0], 1))
            us, prob_us = model_u.call(ts, Ss)
            M_phi = self.intervention(model_phi, ts, Ss, us)
            evaluation = (1-ds)*(L_phi + tf.reshape(f, [tf.shape(f)[0],1])) + ds*(M_phi - output)
            hjb[:,d] = evaluation.flatten()
        return np.argmax(hjb, axis=1)

    def loss_phi_poisson(self, model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas):
        # ts = tf.convert_to_tensor(ts, dtype=tf.float32)
        # Ss = tf.convert_to_tensor(Ss, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(ts)
            output = model_phi.call(ts, Ss)
        phi_t = tape.gradient(output, ts)
        I_phi = 0
        for i in range(self.NDIMS):
            I_phi += lambdas[i]*(model_phi(ts, self.transition(Ss, i)) - output)
        L_phi = phi_t + I_phi
        f = -self.eta*Ss[:,1]**2 # Y_t
        ds, prob_ds = model_d.call(ts, Ss)
        us, prob_us = model_u.call(ts, Ss)
        M_phi = self.intervention(model_phi, ts, Ss, us)
        evaluation = (1-ds)*(L_phi + tf.reshape(f, [tf.shape(f)[0],1])) + ds*(M_phi - output)
        interior_loss = tf.keras.losses.MeanSquaredError()
        loss_int = interior_loss.call(evaluation, tf.zeros(tf.shape(evaluation)))
        output_boundary = model_phi(Ts, S_boundarys)
        g = S_boundarys[:,0] + S_boundarys[:,1]*S_boundarys[:,-1]
        boundary_loss = tf.keras.losses.MeanSquaredError()
        loss_bound = boundary_loss.call(output_boundary, tf.reshape(g, [tf.shape(g)[0],1]))
        return loss_int+loss_bound

    def train_step(self, model_phi, model_d, model_u):
        lambdas = self.lambdas_poisson
        ts, Ss = self.sampler(self.NUM_POINTS)
        ts = tf.convert_to_tensor(ts, dtype=tf.float32)
        Ss = tf.convert_to_tensor(Ss, dtype=tf.float32)
        Ts, S_boundarys = self.sampler(self.NUM_POINTS, boundary=True)
        Ts = tf.convert_to_tensor(Ts, dtype=tf.float32)
        S_boundarys = tf.convert_to_tensor(S_boundarys, dtype=tf.float32)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=self.EPOCHS*10,
            decay_rate=0.9)
        # train value fn first
        optimizer= tf.keras.optimizers.Adam(lr_schedule)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        for j in range(10):
            with tf.GradientTape() as tape:
                loss_phi  = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas)
            gradients = tape.gradient(loss_phi, model_phi.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_phi.trainable_variables))
            train_loss(loss_phi)
            print(f'Model Phi loss: {train_loss.result():0.2f}')

        # train u next
        gt_u = self.oracle_u(model_phi, ts, Ss)
        optimizer= tf.keras.optimizers.Adam(lr_schedule)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        for j in range(10):
            with tf.GradientTape() as tape:
                us, prob_us = model_u.call(ts, Ss)
                loss_u = loss_object.call(gt_u, prob_us)
            gradients = tape.gradient(loss_u, model_u.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_u.trainable_variables))
            train_loss(loss_u)
            print(f'Model u loss: {train_loss.result():0.2f}')

        # train d next
        gt_d = self.oracle_d(model_phi,model_u, ts, Ss)
        optimizer= tf.keras.optimizers.Adam(lr_schedule)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        for j in range(10):
            with tf.GradientTape() as tape:
                ds, prob_ds = model_d.call(ts, Ss)
                loss_d = loss_object.call(gt_d, prob_ds)
            gradients = tape.gradient(loss_d, model_d.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_d.trainable_variables))
            train_loss(loss_d)
            print(f'Model d loss: {train_loss.result():0.2f}')
        return model_phi, model_d, model_u

    def train(self):
        model_phi = DGM.DGMNet(200, 5, 11)
        model_u = DGM.PIANet(200,3,11, 10)
        model_d = DGM.PIANet(200, 3,11, 2)
        for epoch in range(self.EPOCHS):
            model_phi, model_d, model_u = self.train_step(model_phi, model_d, model_u)
        return model_phi, model_d, model_u

MM = MarketMaking(num_points=10000)
MM.train()

