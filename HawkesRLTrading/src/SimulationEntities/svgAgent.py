from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, Dict, Any
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import get_queue_priority
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent

Tensor = torch.Tensor

# -----------------------------
# Networks
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (256, 256),
                 layernorm: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if layernorm:
                layers += [nn.LayerNorm(h)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PolicyNet(nn.Module):
    """Two‑head categorical policy with Gumbel‑Softmax outputs for differentiable actions."""
    def __init__(self, state_dim: int, n_actions_d: int = 13, n_actions_u: int = 13,
                 hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.trunk = MLP(state_dim, hidden[-1], hidden=hidden[:-1] if len(hidden) > 1 else (hidden[0],))
        feat_dim = hidden[-1]
        self.logits_d = nn.Linear(feat_dim, n_actions_d)
        self.logits_u = nn.Linear(feat_dim, n_actions_u)

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.trunk(s)
        return self.logits_d(z), self.logits_u(z)

    @torch.no_grad()
    def act(self, s: Tensor, greedy: bool = True) -> Tuple[int, int]:
        ld, lu = self.forward(s)
        if greedy:
            a_d = torch.argmax(ld, dim=-1)
            a_u = torch.argmax(lu, dim=-1)
        else:
            a_d = Categorical(logits=ld).sample()
            a_u = Categorical(logits=lu).sample()
        return int(a_d.item()), int(a_u.item())

    def gumbel_actions(self, s: Tensor, tau: float = 1.0, hard: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return differentiable one‑hots via straight‑through Gumbel‑Softmax.
        Returns: (onehot_d, onehot_u, logits_d, logits_u)
        """
        logits_d, logits_u = self.forward(s)
        y_d = F.gumbel_softmax(logits_d, tau=tau, hard=hard)
        y_u = F.gumbel_softmax(logits_u, tau=tau, hard=hard)
        return y_d, y_u, logits_d, logits_u


class WorldModel(nn.Module):
    """Predict Δs = f(s, a_d_onehot, a_u_onehot, ε), with ε ~ N(0, I)."""
    def __init__(self, state_dim: int, n_actions_d: int = 13, n_actions_u: int = 13, hidden=(512, 512)):
        super().__init__()
        self.noise_dim = state_dim
        in_dim = state_dim + n_actions_d + n_actions_u + self.noise_dim
        self.net_mu = MLP(in_dim, state_dim, hidden=hidden)
        self.net_logstd = MLP(in_dim, state_dim, hidden=hidden)
        self.max_logstd = 2.0
        self.min_logstd = -5.0

    def forward(self, s: Tensor, a_d_oh: Tensor, a_u_oh: Tensor, eps: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        batch = s.size(0)
        if eps is None:
            eps = torch.randn(batch, self.noise_dim, device=s.device)
        x = torch.cat([s, a_d_oh, a_u_oh, eps], dim=-1)
        mu = self.net_mu(x)
        logstd = torch.clamp(self.net_logstd(x), self.min_logstd, self.max_logstd)
        std = torch.exp(logstd)
        delta = mu + std * torch.randn_like(mu)
        s_next = s + delta
        return s_next, mu, logstd


# -----------------------------
# SVG(∞) Trainer — Algorithm 1 implementation (no imagined multi‑step rollouts)
# -----------------------------

class SVGAgent(GymTradingAgent):
    """Implements Algorithm 1 (Heess et al., 2015) adapted to the given agent.

    Key choices for discrete actions & env rewards:
    - Discrete actions handled with straight‑through Gumbel‑Softmax.
    - Rewards are computed *from state deltas* with a differentiable surrogate
      `reward_from_states(s_t, s_{t+1})` that mirrors your `calculaterewards`.
      No reward network is used.
    - No imagined multi‑step rollouts: gradients use the *observed* transitions by
      inferring the world‑noise ξ that maps s_t,a_t to s_{t+1}, then holding ξ
      fixed while backpropagating through f̂(s_t, π_θ(s_t,η), ξ_t).
    """
    def __init__(self,
                 n_actions_d: int = 2,
                 n_actions_u: int = 4,
                 device: str = "cpu",
                 gamma: float = 0.999,
                 policy_hidden: Tuple[int, ...] = (256, 256),
                 model_hidden: Tuple[int, ...] = (512, 512),
                 tau_gumbel: float = 1.0,
                 entropy_coef: float = 1e-3,
                 lr_policy: float = 1e-2,
                 lr_model: float = 1e-1,
                 seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random",
                 Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5,
                 wake_on_MO: bool=True, wake_on_Spread: bool=True, cashlimit=1000000, inventorylimit=100,
                 start_trading_lag=0, truncation_enabled=True, action_space_config = 1, include_time = False,
                 alt_state=False, enhance_state=False, buffer_capacity=100000, rewardpenalty = 0.1, hidden_activation='leaky_relu',
                 transaction_cost = 0.01):
        super().__init__(seed=seed, log_events=log_events, log_to_file=log_to_file, strategy=strategy,
                         Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO,
                         wake_on_Spread=wake_on_Spread, cashlimit=cashlimit, inventorylimit=inventorylimit, start_trading_lag=start_trading_lag,
                         truncation_enabled=truncation_enabled)

        self.device = torch.device(device)
        self.gamma = gamma
        self.tau_gumbel = tau_gumbel
        self.entropy_coef = entropy_coef
        self.state_dim: Optional[int] = None
        self.policy: Optional[PolicyNet] = None
        self.world: Optional[WorldModel] = None
        self.opt_policy: Optional[torch.optim.Optimizer] = None
        self.opt_world: Optional[torch.optim.Optimizer] = None
        self.n_actions_d = n_actions_d
        self.n_actions_u = n_actions_u
        self._cfg = dict(policy_hidden=policy_hidden, model_hidden=model_hidden,
                         lr_policy=lr_policy, lr_model=lr_model)
        self.action_space_config = action_space_config
        if action_space_config == 0:
            self.allowed_actions= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                                   "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
            self.convert_dict = {}
        elif action_space_config == 1:
            self.allowed_actions= ["lo_top_Ask","co_top_Ask","co_top_Bid", "lo_top_Bid" ]
            self.convert_dict = {0:2, 1:3, 2:8, 3:9}
        elif action_space_config == 2:
            self.allowed_actions = ['lo-lo', 'bid-co-lo', 'ask-co-lo']
        self.include_time = include_time
        self.alt_state = alt_state
        self.enhance_state = enhance_state
        # Trajectory storage
        self.trajectory_buffer = []
        self.buffer_capacity = buffer_capacity
        self.rewardpenalty = rewardpenalty  # inventory penalty
        self.last_state, self.last_action = None, None
        self.transaction_cost = transaction_cost

    def getState(self, data):
        """
        Read and preprocess trading data

        :param data: Input trading data
        :return: Processed state tensor
        """
        time = data['current_time']
        p_a, q_a = data['LOB0']['Ask_L1']
        p_b, q_b = data['LOB0']['Bid_L1']
        self.mid = 0.5*(p_a + p_b)
        _, qD_a = data['LOB0']['Ask_L2']
        _, qD_b = data['LOB0']['Bid_L2']
        pos = data['Positions']
        n_as = get_queue_priority(data, pos, 'Ask_L1')
        n_as = np.append(n_as, get_queue_priority(data, pos, 'Ask_L2'))
        n_as = np.append(n_as, [q_a+qD_a])
        n_bs = get_queue_priority(data, pos, 'Bid_L1')
        n_bs = np.append(n_bs, get_queue_priority(data, pos, 'Bid_L2'))
        n_bs = np.append(n_bs, [q_b+qD_b])
        n_a, n_b = np.min(n_as), np.min(n_bs)
        lambdas = data['current_intensity']
        lambdas_norm = lambdas.flatten()/np.sum(lambdas.flatten())
        past_times = data['past_times']
        if self.Inventory['INTC'] ==0: self.init_cash = self.cash
        skew = (n_a - n_b)/(0.5*(q_a + q_b))
        avgFillPrice = 0
        if self.Inventory['INTC'] != 0 :
            avgFillPrice = (self.cash-self.init_cash)/self.Inventory['INTC']
        bool_mo_bid = np.argmax(lambdas_norm) == 4
        bool_mo_ask = np.argmax(lambdas_norm) == 7
        if self.alt_state:
            state = [[time, self.Inventory['INTC']] ]
            if self.ablation_params.get('spread', True):
                state[0] = state[0] + [p_a - p_b]
            if self.ablation_params.get('n_a', True):
                state[0] = state[0] + [n_a/q_a, n_b/q_b]
            if self.ablation_params.get('lambda', True):
                state[0] = state[0] + list(lambdas_norm)
            if self.ablation_params.get('hist', True):
                state[0] = state[0] + list(past_times.flatten())
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        else:
            state = torch.tensor([[ self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5] + list(lambdas.flatten()) + list(past_times.flatten())], dtype=torch.float32).to(self.device)
        if self.enhance_state:
            state = torch.cat([state, torch.tensor([[skew, p_a < avgFillPrice, p_b > avgFillPrice,  bool_mo_bid, bool_mo_ask]], dtype=torch.float32).to(self.device)], 1)
        if self.include_time:
            state = torch.cat([state,torch.tensor([[time]], dtype=torch.float32).to(self.device)], 1)
        return state

    def readData(self, data):
        return self.getState(data)
    # ---------------- Algorithm 1: outer loop utilities ----------------

    def train(self,
                                max_model_epochs: int = 50,
                                policy_steps: int = 1000,
                                batch_size: int = 256,
                                clear_after: bool = False,
                                logger = None) -> Dict[str, float]:
        """Run the sequence of Algorithm 1 on the current on‑policy database D.

        Steps:
         1–5: (done by env; we consume agent.trajectory_buffer)
         7:   fit world model f̂ using D
         8–9: initialize v'_s = 0, v'_θ = 0 (implicitly via zero tensors)
         10–14: backward pass over each episode computing gradients
         15: update θ using v'_θ at t=0 (we do SGD on accumulated loss)
        """
        if len(self.trajectory_buffer) == 0:
            return {"warn": 1.0}
        # Build per‑episode lists
        episodes: Dict[int, List[Tuple[Any, int, Tuple[int, ...], float, Any, int]]] = {}
        for ep, trans in self.trajectory_buffer:
            episodes.setdefault(ep, []).append(trans)

        max_len = batch_size * 2
        reservoir = []

        n_seen = 0
        for ep, trans_list in episodes.items():
            for (s, a_d, a_u_tuple, r, s2, done) in trans_list:
                # Normalize a_u
                if isinstance(a_u_tuple, (list, tuple)):
                    a_u = int(a_u_tuple[0])
                else:
                    a_u = int(a_u_tuple)

                n_seen += 1

                if len(reservoir) < max_len:
                    # Just fill the reservoir initially
                    reservoir.append((s, int(a_d), a_u, s2))
                else:
                    # Replace with decreasing probability
                    j = random.randint(0, n_seen - 1)
                    if j < max_len:
                        reservoir[j] = (s, int(a_d), a_u, s2)

        # Unpack sampled transitions
        S_list, A_d_list, A_u_list, S2_list = zip(*reservoir)
        S_list, A_d_list, A_u_list, S2_list = list(S_list), list(A_d_list), list(A_u_list), list(S2_list)

        S = torch.stack(S_list).squeeze().float().to(self.device)
        S2 = torch.stack(S2_list).squeeze().float().to(self.device)
        A_d = torch.tensor(A_d_list, dtype=torch.long, device=self.device)
        A_u = torch.tensor(A_u_list, dtype=torch.long, device=self.device)
        if self.state_dim is None:
            self._init_networks(S.size(-1))
        # ---- Step 7: fit world model on D ----
        metrics = self._fit_world(S, A_d, A_u, S2, epochs=max_model_epochs, batch_size=batch_size)
        # ---- Steps 8–15: backward pass & policy update ----
        metrics.update(self._svg_infinity_backward_update({self.trajectory_buffer[-1][0]:episodes[self.trajectory_buffer[-1][0]]}, policy_steps=policy_steps))
        if clear_after:
            self.trajectory_buffer.clear()
        logger.log_losses(**metrics)

        return metrics

    def setupNNs(self, state):
        # Get state dimensions
        data0 = self.getState(state)
        state_dim = len(data0[0])
        self.state_dim = state_dim
        self.policy = PolicyNet(state_dim, self.n_actions_d, self.n_actions_u, hidden=self._cfg['policy_hidden']).to(self.device)
        self.world = WorldModel(state_dim, self.n_actions_d, self.n_actions_u, hidden=self._cfg['model_hidden']).to(self.device)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=self._cfg['lr_policy'])
        self.opt_world = torch.optim.Adam(self.world.parameters(), lr=self._cfg['lr_model'])

    # ---- Differentiable reward built from state deltas (no reward net) ----
    def reward_from_states(self, s: Tensor, s_next: Tensor) -> Tensor:
        """Approximate env reward from state deltas based on agent.calculaterewards.
        State layout (core first 11 dims): [cash(0), inv(1), ..., mid(10)].
        r ≈ Δcash + (inv_next*mid_next - inv*mid) - λ*inv_next^2
        """
        cash_t, inv_t, mid_t = s[:, 0], s[:, 1], s[:, 10]
        cash_tp, inv_tp, mid_tp = s_next[:, 0], s_next[:, 1], s_next[:, 10]
        penalty = self.rewardpenalty
        delta_cash = cash_tp - cash_t
        delta_inv_val = inv_tp * mid_tp - inv_t * mid_t
        r = delta_cash + delta_inv_val - penalty * (inv_tp ** 2)
        return r

    # ---- World fitting (step 7) ----
    def _fit_world(self, S: Tensor, A_d: Tensor, A_u: Tensor, S2: Tensor,
                   epochs: int = 50, batch_size: int = 256) -> Dict[str, float]:
        n = S.size(0)
        idx = torch.arange(n, device=S.device)
        last = {}
        for ep in range(epochs):
            perm = idx[torch.randperm(n)]
            for start in range(0, n, batch_size):
                b = perm[start:start + batch_size]
                s, s2 = S[b], S2[b]
                a_d_oh = F.one_hot(A_d[b], num_classes=self.n_actions_d).float()
                a_u_oh = F.one_hot(A_u[b], num_classes=self.n_actions_u).float()
                s2_hat, mu, logstd = self.world(s, a_d_oh, a_u_oh)
                loss = F.mse_loss(s2_hat, s2)
                self.opt_world.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.world.parameters(), 5.0)
                self.opt_world.step()
                print('World Model Loss : ', float(loss.item()))
                last = {"world_loss": float(loss.item())}
        return last

    # ---- Backward pass & policy update (steps 8–15) ----
    def _svg_infinity_backward_update(self, episodes: Dict[int, List[Tuple[Any, int, Tuple[int, ...], float, Any, int]]],
                                      policy_steps: int = 1) -> Dict[str, float]:
        """Perform backward induction (Algorithm 1 lines 8–15) over the on‑policy episodes.

        This implements the recursive computation of v_s and v_theta by processing each
        episode from T down to 0 and using vector–Jacobian products (torch.autograd.grad).

        We assume the world model f̂ has been trained and is stochastic Gaussian, so we
        infer ξ_t = (s' - μ(s,a))/σ(s,a). For policy randomness η_t we *infer* gumbel noise
        by rejection sampling so that argmax(logits+g)=observed_action (fallback deterministic g).

        For each episode we compute v_θ^0 (gradient of total return wrt θ) and average
        across episodes before applying the parameter update (line 15).
        """
        assert self.policy and self.world
        device = self.device
        policy_params = list(self.policy.parameters())
        old_params = [p.detach().clone() for p in policy_params]
        # accumulator for gradients (sum over episodes)
        agg_param_grads = [torch.zeros_like(p, device=device) for p in policy_params]
        n_episodes = len(episodes)

        def sample_gumbel_conditional(logits: Tensor, action_idx: int, max_tries: int = 500) -> Tensor:
            # logits: [1, K]
            K = logits.size(-1)
            for _ in range(max_tries):
                u = torch.rand_like(logits)
                g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
                if int((logits + g).argmax(dim=-1).item()) == int(action_idx):
                    return g
            # fallback deterministic g that guarantees chosen index is max
            with torch.no_grad():
                g = torch.zeros_like(logits)
                k = int(action_idx)
                # set other entries low enough
                diff = (logits[0, k] - logits[0]) + 1.0
                g = -diff.unsqueeze(0)
                g[0, k] = 0.0
            return g

        # iterate episodes and accumulate v_theta^0
        for ep_idx, trans_list in episodes.items():
            # initialize backwards accumulators
            v_s_next = torch.zeros(self.state_dim, device=device)  # shape (S,)
            v_theta_next = [torch.zeros_like(p, device=device) for p in policy_params]

            # process transitions in reverse order
            for (s_raw, a_d, a_u_tuple, r_obs, s2_raw, done) in reversed(trans_list):
                # prepare tensors
                s_np = s_raw
                s_np2 = s2_raw
                s = s_np
                s.requires_grad_(True)
                s2_obs = s_np2

                # observed action indices
                a_d_idx = int(a_d)
                a_u_idx = int(a_u_tuple[0] if isinstance(a_u_tuple, (tuple, list)) else a_u_tuple)

                # get logits at s
                logits_d, logits_u = self.policy.forward(s)

                # infer eta (gumbel noise) per head so the sampled action equals the observed one
                g_d = sample_gumbel_conditional(logits_d.detach(), a_d_idx)
                g_u = sample_gumbel_conditional(logits_u.detach(), a_u_idx)
                g_d = g_d.to(device)
                g_u = g_u.to(device)

                # form differentiable one‑hot actions using the inferred gumbel noise
                def gumbel_softmax_with_g(logits: Tensor, g: Tensor, tau: float, hard: bool = True) -> Tensor:
                    z = (logits + g) / tau
                    soft = F.softmax(z, dim=-1)
                    if not hard:
                        return soft
                    # hard straight‑through
                    idx = soft.argmax(dim=-1)
                    y_hard = F.one_hot(idx, num_classes=soft.size(-1)).float()
                    return (y_hard - soft).detach() + soft

                a_d_oh = gumbel_softmax_with_g(logits_d, g_d, self.tau_gumbel, hard=True)
                a_u_oh = gumbel_softmax_with_g(logits_u, g_u, self.tau_gumbel, hard=True)

                # infer xi from observed s' using world μ, σ
                with torch.no_grad():
                    mu = self.world.net_mu(torch.cat([s, a_d_oh, a_u_oh, torch.zeros(1, self.world.noise_dim, device=device)], dim=-1))
                    logstd = torch.clamp(self.world.net_logstd(torch.cat([s, a_d_oh, a_u_oh, torch.zeros(1, self.world.noise_dim, device=device)], dim=-1)), self.world.min_logstd, self.world.max_logstd)
                    std = torch.exp(logstd)
                    xi = (s2_obs - mu) / (std + 1e-8)

                # forward through world with fixed xi
                s2_hat, _, _ = self.world(s, a_d_oh, a_u_oh, eps=xi)

                # differentiable reward from states (matches env reward as best approx)
                r_hat = self.reward_from_states(s, s2_hat)
                # sum to scalar for autograd vector–Jacobian products
                r_sum = r_hat.sum()

                # compute gradients r_s, r_a for both heads
                # r_s: d r / d s
                grad_r_s = torch.autograd.grad(r_sum, s, retain_graph=True, allow_unused=True)[0]
                if grad_r_s is None:
                    grad_r_s = torch.zeros_like(s)
                # r_a_d: d r / d a_d
                grad_r_ad = torch.autograd.grad(r_sum, a_d_oh, retain_graph=True, allow_unused=True)[0]
                if grad_r_ad is None:
                    grad_r_ad = torch.zeros_like(a_d_oh)
                grad_r_au = torch.autograd.grad(r_sum, a_u_oh, retain_graph=True, allow_unused=True)[0]
                if grad_r_au is None:
                    grad_r_au = torch.zeros_like(a_u_oh)

                # compute components needed for v_s recursion
                # term: r_a * pi_s  -> compute (d a / d s)^T * r_a
                term_r_a_pi_s_d = torch.autograd.grad(a_d_oh, s, grad_outputs=grad_r_ad, retain_graph=True, allow_unused=True)[0]
                if term_r_a_pi_s_d is None:
                    term_r_a_pi_s_d = torch.zeros_like(s)
                term_r_a_pi_s_u = torch.autograd.grad(a_u_oh, s, grad_outputs=grad_r_au, retain_graph=True, allow_unused=True)[0]
                if term_r_a_pi_s_u is None:
                    term_r_a_pi_s_u = torch.zeros_like(s)

                # f_s_term = (df/ds)^T v_s_next
                if v_s_next is None:
                    v_s_next_tensor = torch.zeros_like(s)
                else:
                    v_s_next_tensor = v_s_next.unsqueeze(0)
                f_s_term = torch.autograd.grad(s2_hat, s, grad_outputs=v_s_next_tensor, retain_graph=True, allow_unused=True)[0]
                if f_s_term is None:
                    f_s_term = torch.zeros_like(s)

                # f_a_term = (df/da)^T v_s_next  (for both heads)
                f_a_term_d = torch.autograd.grad(s2_hat, a_d_oh, grad_outputs=v_s_next_tensor, retain_graph=True, allow_unused=True)[0]
                if f_a_term_d is None:
                    f_a_term_d = torch.zeros_like(a_d_oh)
                f_a_term_u = torch.autograd.grad(s2_hat, a_u_oh, grad_outputs=v_s_next_tensor, retain_graph=True, allow_unused=True)[0]
                if f_a_term_u is None:
                    f_a_term_u = torch.zeros_like(a_u_oh)

                # now compute (df/da * da/ds)^T v_s_next  = (da/ds)^T * (df/da)^T v_s_next
                term_fa_pis_d = torch.autograd.grad(a_d_oh, s, grad_outputs=f_a_term_d, retain_graph=True, allow_unused=True)[0]
                if term_fa_pis_d is None:
                    term_fa_pis_d = torch.zeros_like(s)
                term_fa_pis_u = torch.autograd.grad(a_u_oh, s, grad_outputs=f_a_term_u, retain_graph=True, allow_unused=True)[0]
                if term_fa_pis_u is None:
                    term_fa_pis_u = torch.zeros_like(s)

                # v_s recursion: v_s = r_s + r_a pi_s + gamma * ( f_s_term + f_a pi_s )
                v_s_new = grad_r_s + term_r_a_pi_s_d + term_r_a_pi_s_u + self.gamma * (f_s_term + term_fa_pis_d + term_fa_pis_u)
                # flatten to vector
                v_s_new = v_s_new.squeeze(0)

                # Param gradients contributions
                # g_r_params = d a / d theta^T * r_a  (both heads)
                g_r_params_d = torch.autograd.grad(a_d_oh, policy_params, grad_outputs=grad_r_ad, retain_graph=True, allow_unused=True)
                g_r_params_d = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_r_params_d, policy_params)]
                g_r_params_u = torch.autograd.grad(a_u_oh, policy_params, grad_outputs=grad_r_au, retain_graph=True, allow_unused=True)
                g_r_params_u = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_r_params_u, policy_params)]
                # g_s2_params = (df/da * da/dtheta)^T v_s_next = grad(s2_hat, params, grad_outputs=v_s_next)
                g_s2_params = torch.autograd.grad(s2_hat, policy_params, grad_outputs=v_s_next_tensor, retain_graph=True, allow_unused=True)
                g_s2_params = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_s2_params, policy_params)]

                # combine to get v_theta at this step: v_theta = r_a pi_theta + gamma * g_s2_params + gamma * v_theta_next
                v_theta_new = []
                for i, p in enumerate(policy_params):
                    cur = g_r_params_d[i] + g_r_params_u[i] + self.gamma * g_s2_params[i] + self.gamma * v_theta_next[i]
                    # ensure tensor is detached from current graph (we will not backprop through v_theta)
                    v_theta_new.append(cur.detach())

                # move to next step
                v_s_next = v_s_new.detach()
                v_theta_next = v_theta_new

            # end of episode backward pass: v_theta_next now equals gradient of return wrt theta for this episode
            # accumulate
            for i, g in enumerate(v_theta_next):
                agg_param_grads[i] += g

        # average across episodes
        for i in range(len(agg_param_grads)):
            agg_param_grads[i] /= max(1, n_episodes)

        # apply gradient update (we minimize -V, so set param.grad = -agg)
        for p, g in zip(policy_params, agg_param_grads):
            if p.grad is None:
                p.grad = -g.clone()
            else:
                p.grad.copy_(-g)
        # single optimizer step
        self.opt_policy.step()
        # zero grads for safety
        self.opt_policy.zero_grad()

        # --- compute parameter change for convergence check ---
        with torch.no_grad():
            diffs = [torch.norm(p.detach() - old_p).item() for p, old_p in zip(policy_params, old_params)]
            max_diff = max(diffs) if diffs else 0.0
            mean_diff = float(torch.tensor(diffs).mean().item()) if diffs else 0.0

        return {
            "svg_backward_applied": 1.0,
            "param_max_update": max_diff,
            "param_mean_update": mean_diff
        }

    def _world_stats(self, s: Tensor, a_d_oh: Tensor, a_u_oh: Tensor) -> Tuple[Tensor, Tensor]:
        # Helper to get μ, σ of world(s,a)
        with torch.no_grad():
            x = torch.cat([s, a_d_oh, a_u_oh, torch.zeros(s.size(0), self.world.noise_dim, device=s.device)], dim=-1)
            mu = self.world.net_mu(x)
            logstd = torch.clamp(self.world.net_logstd(x), self.world.min_logstd, self.world.max_logstd)
            std = torch.exp(logstd)
        return mu, std

    def get_action(self, data, epsilon: float = 0.1):
        """
        Get action using current policy with epsilon-greedy exploration.

        :param data: Input trading data
        :param epsilon: Exploration probability
        :return: (chosen_action, (d_idx, u_idx), d_log_prob, u_log_prob, d_value, u_value)
        """
        if self.action_space_config < 2:
            if self.breach:
                mo = 4 if self.countInventory() > 0 else 7
                return mo, (None, None), 0, 0, 0, 0

            origData = data.copy()
            state = self.getState(data)  # use your SVGAgent.getState
            self.last_state = state

            # Exploration
            if (random.random() < epsilon) or (len(self.trajectory_buffer) < 10):
                d = random.randint(0, self.n_actions_d - 1)
                u = random.randint(0, self.n_actions_u - 1)
                _u = copy.deepcopy(u)
                u = self.convert_dict.get(u, u)

                # Compute logits from policy for log-probs
                with torch.no_grad():
                    d_logits, u_logits = self.policy(state)
                    d_log_prob = torch.log_softmax(d_logits, dim=1)[0, d]
                    u_log_prob = torch.log_softmax(u_logits, dim=1)[0, _u]

                # Validation checks
                if int(u) in [1, 3, 8, 10]:
                    a = self.actions[int(u)]
                    lvl = self.actionsToLevels[a]
                    if len(origData['Positions'][lvl]) == 0:
                        self.last_action = 12
                        return 12, (d, 0), d_log_prob.item(), 0, 0, 0

                if int(u) in [5, 6]:
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:
                        self.last_action = 12
                        return 12, (d, 0), d_log_prob.item(), 0, 0, 0

                if (int(u) == 4) and (self.countInventory() < 1):
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, 0, 0

                self.last_action = u
                return u, (d, _u), d_log_prob.item(), u_log_prob.item(), 0, 0

            # Exploitation
            with torch.no_grad():
                d_logits, u_logits = self.policy(state)

                d_probs = torch.softmax(d_logits, dim=1).squeeze()
                d = torch.multinomial(d_probs, 1).item()
                d_log_prob = torch.log(d_probs[d])

                # If no decision to act
                if d == 0:
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, 0, 0

                u_probs = torch.softmax(u_logits, dim=1).squeeze()
                u = torch.multinomial(u_probs, 1).item()
                _u = copy.deepcopy(u)
                u = self.convert_dict.get(u, u)

                # Inventory limit handling
                if np.abs(self.countInventory()) >= self.inventorylimit - 1:
                    if self.countInventory() < 0:
                        u = 3
                    elif self.countInventory() > 0:
                        u = 8

                u_log_prob = torch.log(u_probs[_u])

                # Validation checks
                if int(u) in [1, 3, 8, 10]:
                    a = self.actions[int(u)]
                    lvl = self.actionsToLevels[a]
                    if len(origData['Positions'][lvl]) == 0:
                        self.last_action = 12
                        return 12, (d, 0), d_log_prob.item(), 0, 0, 0

                if int(u) in [5, 6]:
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:
                        self.last_action = 12
                        return 12, (d, 0), d_log_prob.item(), 0, 0, 0

                if (int(u) == 4) and ((self.countInventory() < 1) or
                                      (self.countInventory() >= self.inventorylimit - 2)):
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, 0, 0

                if (int(u) == 7) and (self.countInventory() <= 2 - self.inventorylimit):
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, 0, 0

                self.last_action = u
                return u, (d, _u), d_log_prob.item(), u_log_prob.item(), 0, 0

    def store_transition(self, ep, state, action, reward, next_state, done):
        """
        Store transition in trajectory buffer

        :param state: Current state
        :param action: Chosen action (tuple of d and u)
        :param reward: Received reward
        :param next_state: Next state
        :param done: Episode termination flag
        """
        # Unpack action components
        d, u = action
        if d is None: return
        # Store additional trajectory information
        transition = (
            state,       # Current state
            d,           # Decision action
            u,           # Utility action
            reward,      # Reward
            next_state,  # Next state
            int(done)         # Done flag
        )
        self.trajectory_buffer.append((ep, transition))

    def calculaterewards(self, termination) -> Any:
        penalty = self.rewardpenalty * (self.countInventory()**2)
        self.profit = self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL = self.statelog[-1][2] - self.statelog[-2][2]
        deltaInv = self.statelog[-1][3]['INTC']*self.statelog[-1][-1]*(1- self.transaction_cost*np.sign(self.statelog[-1][3]['INTC'])) - self.statelog[-2][3]['INTC']*self.statelog[-2][-1]*(1-self.transaction_cost*np.sign(self.statelog[-1][3]['INTC']))
        # if self.istruncated or termination:
        #     deltaPNL += self.countInventory() * self.mid
        # reward shaping
        if self.istruncated:
            penalty += 100
        if self.last_action != 12:
            penalty -= self.rewardpenalty *10 # custom reward for incentivising actions rather than inaction for learning
        if (not self.alt_state) and (self.last_state.cpu().numpy()[0][8] < self.last_state.cpu().numpy()[0][4] + self.last_state.cpu().numpy()[0][6]) and (self.last_state.cpu().numpy()[0][9] < self.last_state.cpu().numpy()[0][5] + self.last_state.cpu().numpy()[0][7]):
            penalty -= self.rewardpenalty *20 # custom reward for double sided quoting
        if self.alt_state:
            if self.two_sided_reward:
                if (self.last_state.cpu().numpy()[0][3] <= 1) and (self.last_state.cpu().numpy()[0][4] <= 1):
                    penalty -= self.rewardpenalty *20 # custom reward for double sided quoting
            if self.exploration_bonus:
                penalty -= self.visit_counter.get_exploration_bonus(self.last_state.cpu().numpy()[0][1:5], self.last_action)
        return deltaPNL + deltaInv - penalty



# -----------------------------
# Minimal usage example (pseudo‑code)
# -----------------------------
"""
# agent = ProbabilisticAgent(...)
# trainer = SVGInfinityTrainer(agent, device="cuda" if torch.cuda.is_available() else "cpu")
#
# while training:
#     # Run your market sim, calling `agent.store_transition(ep, s, a, r, s_next, done)` as you already do.
#     ...
#     # Periodically train SVG(∞) on the fresh on‑policy data
#     metrics = trainer.train_from_agent_buffer(max_model_epochs=25, max_policy_steps=500)
#     print(metrics)
#     # To deploy the learned policy for action selection:
#     # state = agent.readData(data)
#     # action = policy_action_for_agent(trainer.policy, state)
#     # return action
"""
