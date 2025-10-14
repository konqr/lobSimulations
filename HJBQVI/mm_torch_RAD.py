import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture
from mm_torch import MarketMaking, gumbel_softmax_sample
import torch.nn as nn

class RADSampler:
    """
    Residual-based Adaptive Distribution (RAD) sampler for DGM
    Based on Wu et al. (2023) methodology for adaptive sampling in neural PDE solvers
    Enhanced with boundary loss handling
    """

    def __init__(self, initial_points=1000, buffer_size=5000,
                 adaptation_frequency=10, residual_threshold=1e-3,
                 sampling_method='probability', alpha=0.8, beta=0.2,
                 boundary_weight=0.5, boundary_buffer_ratio=0.3):
        """
        Args:
            initial_points: Initial number of sampling points
            buffer_size: Maximum number of points to maintain
            adaptation_frequency: How often to update sampling distribution
            residual_threshold: Threshold for high-residual regions
            sampling_method: 'probability' or 'importance' or 'hybrid'
            alpha: Weight for residual-based sampling
            beta: Weight for diversity-based sampling
            boundary_weight: Weight for boundary loss in total loss
            boundary_buffer_ratio: Ratio of buffer dedicated to boundary points
        """
        self.initial_points = initial_points
        self.buffer_size = buffer_size
        self.adaptation_frequency = adaptation_frequency
        self.residual_threshold = residual_threshold
        self.sampling_method = sampling_method
        self.alpha = alpha
        self.beta = beta
        self.boundary_weight = boundary_weight
        self.boundary_buffer_ratio = boundary_buffer_ratio

        # Storage for interior points and residuals
        self.point_buffer = None
        self.residual_buffer = None
        self.weights_buffer = None

        # Storage for boundary points and losses
        self.boundary_buffer = None
        self.boundary_loss_buffer = None
        self.boundary_weights_buffer = None

        self.iteration_count = 0

        # Gaussian Mixture Models for adaptive distribution
        self.gmm = None  # For interior points
        self.boundary_gmm = None  # For boundary points
        self.n_components = 5

    def compute_residuals(self, model_phi, model_d, model_u, ts, Ss, market_maker):
        """
        Compute HJB residuals at given points
        """
        device = ts.device
        batch_size = ts.shape[0]

        # Compute HJB residual
        lambdas = market_maker.lambdas_poisson

        # Time derivative
        ts.requires_grad_(True)
        output = model_phi(ts, Ss)
        phi_t = torch.autograd.grad(output.sum(), ts, create_graph=True)[0]

        # Integral term (jump part)
        I_phi = torch.zeros_like(output)
        for i in range(market_maker.NDIMS):
            Ss_transitioned = market_maker.transition(Ss, torch.tensor(i, dtype=torch.float32))
            I_phi += lambdas[i] * (model_phi(ts, Ss_transitioned) - output)

        # Generator term
        L_phi = phi_t + I_phi

        # Running cost
        f = -market_maker.eta * torch.square(Ss[:, 1])
        f = f.reshape(-1, 1)

        # Optimal controls
        ds, _ = model_d(ts, Ss)
        us, _ = model_u(ts, Ss)
        ds = torch.argmax(ds, 1).float().reshape(-1, 1)

        # Intervention value
        M_phi = model_phi(ts, market_maker.intervention(ts, Ss, us))

        # HJB residual
        hjb_residual = (1 - ds) * (L_phi + f) + ds * (M_phi - output)

        # Return absolute residual values
        return torch.abs(hjb_residual).detach()

    def compute_boundary_loss(self, model_phi, Ts, S_boundarys, market_maker):
        """
        Compute boundary condition losses at terminal points
        """
        # Terminal condition: φ(T, S) = g(S) where g is terminal payoff
        predicted_terminal = model_phi(Ts, S_boundarys)

        # Terminal payoff function - customize based on your problem
        # For market making, this could be the terminal inventory penalty
        terminal_payoff = market_maker.terminal_condition(S_boundarys)

        # Boundary loss (MSE between predicted and true terminal values)
        boundary_loss = torch.square(predicted_terminal - terminal_payoff)

        return boundary_loss.detach()

    def update_sampling_distribution(self, points, residuals, boundary_points=None, boundary_losses=None):
        """
        Update the sampling distribution based on residuals and boundary losses using RAD methodology
        """
        points_np = points.cpu().numpy()
        residuals_np = residuals.cpu().numpy().flatten()

        # Normalize residuals to create probability weights
        residuals_normalized = residuals_np / (np.sum(residuals_np) + 1e-8)

        # Update interior point sampling weights
        if self.sampling_method == 'probability':
            # Direct probability sampling based on residuals
            self.weights_buffer = residuals_normalized

        elif self.sampling_method == 'importance':
            # Importance sampling with residual-based weights
            self.weights_buffer = np.power(residuals_normalized, self.alpha)
            self.weights_buffer /= np.sum(self.weights_buffer)

        elif self.sampling_method == 'hybrid':
            # Hybrid approach: combine residual-based and diversity-based sampling

            # 1. Residual-based component
            residual_weights = np.power(residuals_normalized, self.alpha)

            # 2. Diversity-based component (anti-clustering)
            if len(points_np) > 1:
                # Compute pairwise distances
                distances = pdist(points_np)
                distance_matrix = squareform(distances)

                # Points in sparse regions get higher weights
                min_distances = np.min(distance_matrix + np.eye(len(points_np)) * 1e6, axis=1)
                diversity_weights = min_distances / (np.sum(min_distances) + 1e-8)
            else:
                diversity_weights = np.ones(len(points_np)) / len(points_np)

            # Combine weights
            self.weights_buffer = self.alpha * residual_weights + self.beta * diversity_weights
            self.weights_buffer /= np.sum(self.weights_buffer)

        # Update boundary point sampling weights
        if boundary_points is not None and boundary_losses is not None:
            boundary_points_np = boundary_points.cpu().numpy()
            boundary_losses_np = boundary_losses.cpu().numpy().flatten()

            # Normalize boundary losses
            boundary_losses_normalized = boundary_losses_np / (np.sum(boundary_losses_np) + 1e-8)
            self.boundary_weights_buffer = boundary_losses_normalized

            # Fit boundary GMM
            if len(boundary_points_np) >= self.n_components:
                n_samples = len(boundary_points_np)
                resample_indices = np.random.choice(
                    np.arange(n_samples),
                    size=n_samples,
                    replace=True,
                    p=self.boundary_weights_buffer / np.sum(self.boundary_weights_buffer)
                )
                boundary_points_weighted = boundary_points_np[resample_indices]

                self.boundary_gmm = GaussianMixture(n_components=self.n_components, random_state=42)
                self.boundary_gmm.fit(boundary_points_weighted)

        # Fit interior GMM for continuous sampling
        if len(points_np) >= self.n_components:
            # Use residuals as sample weights for GMM fitting
            n_samples = len(points_np)
            resample_indices = np.random.choice(
                np.arange(n_samples),
                size=n_samples,
                replace=True,
                p=self.weights_buffer / np.sum(self.weights_buffer)
            )
            points_weighted = points_np[resample_indices]

            self.gmm = GaussianMixture(n_components=self.n_components, random_state=42)
            self.gmm.fit(points_weighted)

    def sample_from_distribution(self, n_samples, state_bounds, boundary=False):
        """
        Generate new samples from the adaptive distribution
        """
        if boundary and self.boundary_gmm is not None:
            # Sample from boundary distribution
            new_points, _ = self.boundary_gmm.sample(n_samples)

            # Clip to state bounds (excluding time dimension for boundary)
            for i, (low, high) in enumerate(state_bounds):
                new_points[:, i] = np.clip(new_points[:, i], low, high)

            return torch.tensor(new_points, dtype=torch.float32)

        elif not boundary and self.gmm is not None and self.point_buffer is not None:
            # Sample from interior distribution
            new_points, _ = self.gmm.sample(n_samples)

            # Clip to state bounds
            for i, (low, high) in enumerate(state_bounds):
                new_points[:, i] = np.clip(new_points[:, i], low, high)

            return torch.tensor(new_points, dtype=torch.float32)
        else:
            # Fallback to uniform sampling
            n_dims = len(state_bounds)
            samples = np.random.uniform(0, 1, (n_samples, n_dims))

            # Scale to bounds
            for i, (low, high) in enumerate(state_bounds):
                samples[:, i] = low + samples[:, i] * (high - low)

            return torch.tensor(samples, dtype=torch.float32)

    def adaptive_resample(self, current_points, current_residuals, n_new_points, boundary=False):
        """
        Resample points based on residual/boundary loss distribution
        """
        weights = self.boundary_weights_buffer if boundary else self.weights_buffer

        if weights is None:
            return current_points

        # Sample indices based on weights
        indices = np.random.choice(
            len(current_points),
            size=min(n_new_points, len(current_points)),
            p=weights,
            replace=True
        )

        # Add noise to selected points for exploration
        selected_points = current_points[indices].clone()
        noise_scale = 0.1 * torch.std(current_points, dim=0)
        noise = torch.randn_like(selected_points) * noise_scale

        return selected_points + noise

class MarketMakingRAD(MarketMaking):
    """
    Market Making with Residual-based Adaptive Distribution sampling
    Enhanced with boundary loss handling
    """

    def __init__(self, num_points=100, num_epochs=1000, ric='AAPL', hawkes=True):
        super().__init__(num_points, num_epochs, ric, hawkes=hawkes)
        self.rad_sampler = RADSampler(
            initial_points=num_points,
            adaptation_frequency=10,
            sampling_method='hybrid',
            boundary_weight=0.3,
            boundary_buffer_ratio=0.3
        )

        # Define state bounds for sampling
        self.state_bounds = [
            (-5000, 5000),    # X (cash)
            (-10, 10),        # Y (inventory)
            (180, 220),       # p_a (ask price)
            (180, 220),       # p_b (bid price)
            (1, 100),         # q_a (ask queue)
            (1, 100),         # q_b (bid queue)
            (1, 100),         # qD_a (deep ask queue)
            (1, 100),         # qD_b (deep bid queue)
            (0, 200),         # n_a (ask priority)
            (0, 200),         # n_b (bid priority)
            (180, 220)        # P_mid (mid price)
        ]

        # Track boundary loss statistics
        self.boundary_loss_history = []

    def adaptive_sampler(self, model_phi, model_d, model_u, num_points, boundary=False):
        """
        RAD-enhanced sampler that adapts based on HJB residuals and boundary losses
        """
        if boundary:
            # Adaptive boundary sampling
            return self.adaptive_boundary_sampler(model_phi, model_d, model_u, num_points)

        # Check if we need to update the sampling distribution
        if (self.rad_sampler.iteration_count % self.rad_sampler.adaptation_frequency == 0 and
                self.rad_sampler.point_buffer is not None):

            # Compute residuals on current buffer
            residuals = self.rad_sampler.compute_residuals(
                model_phi, model_d, model_u,
                self.rad_sampler.point_buffer[:, :1],  # time
                self.rad_sampler.point_buffer[:, 1:],  # state
                self
            )

            # Also update boundary distribution if we have boundary data
            boundary_points = None
            boundary_losses = None
            if self.rad_sampler.boundary_buffer is not None:
                boundary_losses = self.rad_sampler.compute_boundary_loss(
                    model_phi,
                    self.rad_sampler.boundary_buffer[:, :1],  # time (should be T)
                    self.rad_sampler.boundary_buffer[:, 1:],  # state
                    self
                )
                boundary_points = self.rad_sampler.boundary_buffer[:, 1:]

            # Update sampling distribution
            self.rad_sampler.update_sampling_distribution(
                self.rad_sampler.point_buffer[:, 1:], residuals,
                boundary_points, boundary_losses
            )

        # Generate new samples
        if self.rad_sampler.point_buffer is None:
            # Initial sampling
            ts, Ss = self.sampler(num_points)
            combined_points = torch.cat([ts, Ss], dim=1)
        else:
            # Adaptive sampling strategy
            n_adaptive = int(0.7 * num_points)  # 70% adaptive
            n_uniform = num_points - n_adaptive  # 30% uniform

            # Adaptive samples based on residual distribution
            adaptive_states = self.rad_sampler.sample_from_distribution(
                n_adaptive, self.state_bounds
            )
            adaptive_times = torch.rand(n_adaptive, 1) * self.TERMINATION_TIME
            adaptive_points = torch.cat([adaptive_times, adaptive_states], dim=1)

            # Uniform samples for exploration
            uniform_ts, uniform_Ss = self.sampler(n_uniform)
            uniform_points = torch.cat([uniform_ts, uniform_Ss], dim=1)

            # Combine samples
            combined_points = torch.cat([adaptive_points, uniform_points], dim=0)

        # Update buffer with new points
        self._update_interior_buffer(combined_points)

        # Extract time and state
        ts_out = combined_points[:, :1]
        Ss_out = combined_points[:, 1:]

        self.rad_sampler.iteration_count += 1

        return ts_out, Ss_out

    def adaptive_boundary_sampler(self, model_phi, model_d, model_u, num_points):
        """
        Adaptive boundary sampling based on boundary loss
        """
        # Check if we need to update boundary sampling distribution
        if (self.rad_sampler.iteration_count % self.rad_sampler.adaptation_frequency == 0 and
                self.rad_sampler.boundary_buffer is not None):

            # Compute boundary losses on current boundary buffer
            boundary_losses = self.rad_sampler.compute_boundary_loss(
                model_phi,
                self.rad_sampler.boundary_buffer[:, :1],  # time (T)
                self.rad_sampler.boundary_buffer[:, 1:],  # state
                self
            )

            # Update boundary sampling distribution
            self.rad_sampler.update_sampling_distribution(
                None, None,  # No interior updates
                self.rad_sampler.boundary_buffer[:, 1:], boundary_losses
            )

        # Generate boundary samples
        if self.rad_sampler.boundary_buffer is None:
            # Initial boundary sampling
            Ts, S_boundarys = self.sampler(num_points, boundary=True)
            boundary_combined = torch.cat([Ts, S_boundarys], dim=1)
        else:
            # Adaptive boundary sampling strategy
            n_adaptive = int(0.8 * num_points)  # 80% adaptive for boundary
            n_uniform = num_points - n_adaptive  # 20% uniform

            # Adaptive boundary samples
            if self.rad_sampler.boundary_gmm is not None:
                adaptive_boundary_states = self.rad_sampler.sample_from_distribution(
                    n_adaptive, self.state_bounds, boundary=True
                )
                # Boundary points are at terminal time
                adaptive_boundary_times = torch.full((n_adaptive, 1), self.TERMINATION_TIME)
                adaptive_boundary_points = torch.cat([adaptive_boundary_times, adaptive_boundary_states], dim=1)
            else:
                adaptive_boundary_times, adaptive_boundary_states = self.sampler(n_adaptive, boundary=True)
                adaptive_boundary_points = torch.cat([adaptive_boundary_times, adaptive_boundary_states], dim=1)

            # Uniform boundary samples
            uniform_Ts, uniform_S_boundarys = self.sampler(n_uniform, boundary=True)
            uniform_boundary_points = torch.cat([uniform_Ts, uniform_S_boundarys], dim=1)

            # Combine boundary samples
            boundary_combined = torch.cat([adaptive_boundary_points, uniform_boundary_points], dim=0)

        # Update boundary buffer
        self._update_boundary_buffer(boundary_combined)

        # Extract time and state
        Ts_out = boundary_combined[:, :1]
        S_boundarys_out = boundary_combined[:, 1:]

        return Ts_out, S_boundarys_out

    def _update_interior_buffer(self, new_points):
        """
        Update interior point buffer with new points
        """
        if self.rad_sampler.point_buffer is None:
            self.rad_sampler.point_buffer = new_points
        else:
            # Calculate buffer sizes
            interior_buffer_size = int(self.rad_sampler.buffer_size * (1 - self.rad_sampler.boundary_buffer_ratio))

            # Check if buffer is full
            if self.rad_sampler.point_buffer.shape[0] + new_points.shape[0] > interior_buffer_size:
                # Remove oldest points
                keep_size = interior_buffer_size - new_points.shape[0]
                self.rad_sampler.point_buffer = self.rad_sampler.point_buffer[-keep_size:]
                if self.rad_sampler.residual_buffer is not None:
                    self.rad_sampler.residual_buffer = self.rad_sampler.residual_buffer[-keep_size:]

            # Add new points to buffer
            self.rad_sampler.point_buffer = torch.cat([
                self.rad_sampler.point_buffer, new_points
            ], dim=0)

    def _update_boundary_buffer(self, new_boundary_points):
        """
        Update boundary point buffer with new points
        """
        if self.rad_sampler.boundary_buffer is None:
            self.rad_sampler.boundary_buffer = new_boundary_points
        else:
            # Calculate boundary buffer size
            boundary_buffer_size = int(self.rad_sampler.buffer_size * self.rad_sampler.boundary_buffer_ratio)

            # Check if buffer is full
            if self.rad_sampler.boundary_buffer.shape[0] + new_boundary_points.shape[0] > boundary_buffer_size:
                # Remove oldest points
                keep_size = boundary_buffer_size - new_boundary_points.shape[0]
                self.rad_sampler.boundary_buffer = self.rad_sampler.boundary_buffer[-keep_size:]
                if self.rad_sampler.boundary_loss_buffer is not None:
                    self.rad_sampler.boundary_loss_buffer = self.rad_sampler.boundary_loss_buffer[-keep_size:]

            # Add new boundary points to buffer
            self.rad_sampler.boundary_buffer = torch.cat([
                self.rad_sampler.boundary_buffer, new_boundary_points
            ], dim=0)

    def train_step_rad(self, model_phi, optimizer_phi, scheduler_phi,
                       model_d, optimizer_d, scheduler_d,
                       model_u, optimizer_u, scheduler_u,
                       phi_epochs=10, phi_optim='ADAM'):
        """
        Training step with RAD sampling including boundary loss
        """
        lambdas = self.lambdas_poisson

        # Generate adaptive samples for interior points
        ts, Ss = self.adaptive_sampler(model_phi, model_d, model_u, self.NUM_POINTS)

        # Generate adaptive samples for boundary points
        Ts, S_boundarys = self.adaptive_sampler(model_phi, model_d, model_u, self.NUM_POINTS, boundary=True)

        ts = ts.to(self.device)
        Ss = Ss.to(self.device)
        Ts = Ts.to(self.device)
        S_boundarys = S_boundarys.to(self.device)

        # Compute and store residuals for monitoring
        if self.rad_sampler.iteration_count > 1:
            current_residuals = self.rad_sampler.compute_residuals(
                model_phi, model_d, model_u, ts, Ss, self
            )

            # Compute boundary losses
            current_boundary_losses = self.rad_sampler.compute_boundary_loss(
                model_phi, Ts, S_boundarys, self
            )

            # Store residuals in buffer
            if self.rad_sampler.residual_buffer is None:
                self.rad_sampler.residual_buffer = current_residuals
            else:
                interior_buffer_size = int(self.rad_sampler.buffer_size * (1 - self.rad_sampler.boundary_buffer_ratio))
                if (self.rad_sampler.residual_buffer.shape[0] + len(current_residuals) > interior_buffer_size):
                    keep_size = interior_buffer_size - len(current_residuals)
                    self.rad_sampler.residual_buffer = self.rad_sampler.residual_buffer[-keep_size:]

                self.rad_sampler.residual_buffer = torch.cat([
                    self.rad_sampler.residual_buffer, current_residuals
                ], dim=0)

            # Store boundary losses in buffer
            if self.rad_sampler.boundary_loss_buffer is None:
                self.rad_sampler.boundary_loss_buffer = current_boundary_losses
            else:
                boundary_buffer_size = int(self.rad_sampler.buffer_size * self.rad_sampler.boundary_buffer_ratio)
                if (self.rad_sampler.boundary_loss_buffer.shape[0] + len(current_boundary_losses) > boundary_buffer_size):
                    keep_size = boundary_buffer_size - len(current_boundary_losses)
                    self.rad_sampler.boundary_loss_buffer = self.rad_sampler.boundary_loss_buffer[-keep_size:]

                self.rad_sampler.boundary_loss_buffer = torch.cat([
                    self.rad_sampler.boundary_loss_buffer, current_boundary_losses
                ], dim=0)

            # Print statistics
            mean_residual = torch.mean(current_residuals).item()
            max_residual = torch.max(current_residuals).item()
            mean_boundary_loss = torch.mean(current_boundary_losses).item()
            max_boundary_loss = torch.max(current_boundary_losses).item()

            print(f'Residual stats - Mean: {mean_residual:.6f}, Max: {max_residual:.6f}')
            print(f'Boundary loss stats - Mean: {mean_boundary_loss:.6f}, Max: {max_boundary_loss:.6f}')

            # Store boundary loss history
            self.boundary_loss_history.append(mean_boundary_loss)

        # Enhanced loss computation with boundary weighting
        def closure():
            optimizer_phi.zero_grad()
            # Interior loss
            loss_phi_interior, _ = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas, train_phi=True)

            # Boundary loss
            loss_phi_boundary = self.compute_boundary_loss_for_training(model_phi, Ts, S_boundarys)

            # Combined loss
            total_loss = (1 - self.rad_sampler.boundary_weight) * loss_phi_interior + self.rad_sampler.boundary_weight * loss_phi_boundary
            total_loss.backward()
            return total_loss

        # Train phi with combined loss
        for j in range(phi_epochs):
            # Interior loss
            loss_phi_interior, _ = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas, train_phi=True)

            # Boundary loss
            loss_phi_boundary = self.compute_boundary_loss_for_training(model_phi, Ts, S_boundarys)

            # Combined loss with boundary weighting
            total_loss_phi = ((1 - self.rad_sampler.boundary_weight) * loss_phi_interior +
                              self.rad_sampler.boundary_weight * loss_phi_boundary)

            # Clip gradients to prevent exploding values
            torch.nn.utils.clip_grad_norm_(model_phi.parameters(), 1.0)

            if phi_optim == 'ADAM':
                optimizer_phi.zero_grad()
                total_loss_phi.backward()
                optimizer_phi.step()
            else:
                optimizer_phi.step(closure)

            train_loss_phi = total_loss_phi.item()
            if scheduler_phi is not None:
                scheduler_phi.step()

            # Check for NaN or Inf
            if torch.isnan(total_loss_phi) or torch.isinf(total_loss_phi):
                print("Warning: NaN or Inf detected in loss value")
                break

            print(f'Model Phi combined loss: {train_loss_phi:0.4f} (Interior: {loss_phi_interior.item():0.4f}, Boundary: {loss_phi_boundary.item():0.4f})')
            for param_group in optimizer_phi.param_groups:
                lr = param_group['lr']
                print('Model Phi LR: '+str(lr))

        # Train control function u
        for j in range(phi_epochs):
            _, eval = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas, train_u = True)
            loss_u = -eval.mean()
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

        # Train decision function d
        for j in range(phi_epochs):
            _, eval = self.loss_phi_poisson(model_phi, model_d, model_u, ts, Ss, Ts, S_boundarys, lambdas, train_d = True)
            loss_d = -eval.mean()
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

        acc_u = 0.0
        acc_d = 0.0

        return model_phi, model_d, model_u, train_loss_phi, train_loss_d, train_loss_u, acc_d, acc_u

    def compute_boundary_loss_for_training(self, model_phi, Ts, S_boundarys):
        """
        Compute boundary loss for training (returns tensor for backprop)
        """
        # Terminal condition: φ(T, S) = g(S)
        predicted_terminal = model_phi(Ts, S_boundarys)
        terminal_payoff = self.terminal_condition(S_boundarys)

        # Boundary loss (MSE)
        boundary_loss = torch.mean(torch.square(predicted_terminal - terminal_payoff))

        return boundary_loss

    def terminal_condition(self, S_boundarys):
        """
        Define terminal condition g(S) for the market making problem
        """
        # Terminal penalty based on inventory position

        g = S_boundarys[:, 0] + S_boundarys[:, 1] * S_boundarys[:, -1] - 2*self.eta*S_boundarys[:, 1]**2  # terminal condition

        return g.reshape(-1, 1)

# Usage example with RAD integration
def train_with_rad():
    """
    Example of how to use RAD with the market making DGM
    """
    # Initialize RAD-enhanced market maker
    mm_rad = MarketMakingRAD(num_epochs=2000, num_points=100)

    # Override the train_step method to use RAD
    original_train_step = mm_rad.train_step
    mm_rad.train_step = mm_rad.train_step_rad

    # Train with adaptive sampling
    model_phi, model_d, model_u = mm_rad.train(
        sampler='rad',  # Custom sampler flag
        log_dir='logs_rad',
        model_dir='models_rad',
        typeNN='LSTM',
        layer_widths=[20]*3,
        n_layers=[3]*3,
        unified=False,
        label='RAD_enhanced',
        lr=1e-4,
        phi_epochs=1
    )

    return model_phi, model_d, model_u

class RADMarketMaking(MarketMaking):
    """
    MarketMaking subclass that implements Residual-Adaptive Distribution (RAD) sampling for the interior (PINN) loss.
    - Focuses on HAWKES variant.
    - Ignores boundary loss entirely (interior only).
    """

    def __init__(self, *args, pool_multiplier=10, alpha=1.0, **kwargs):
        """
        pool_multiplier: how many candidates relative to requested NUM_POINTS (pool_size = pool_multiplier * NUM_POINTS)
        alpha: exponent for residual weighting (prob ∝ |residual|^alpha). alpha=1 linear, >1 concentrates more.
        All other args forwarded to MarketMaking.__init__
        """
        super().__init__(*args, **kwargs)
        self.pool_multiplier = pool_multiplier
        self.alpha = alpha

    ###############
    # RAD utilities
    ###############
    def _compute_hawkes_interior_evaluation(self, model_phi, model_d, model_u, ts, Ss, lambdas, train_phi=False, train_d=False, train_u=False, tau=0.5):
        """
        Compute the HJB interior evaluation (same expression used to compute interior loss in loss_phi_hawkes)
        but DO NOT compute or return any boundary loss. Returns:
            evaluation: (N,1) tensor containing HJB residual (evaluation) for each sample (before MSE)
        This function reproduces the forward / derivative / I_phi / L_phi / decision/control logic needed
        to compute the per-sample HJB residual (same as in your loss_phi_hawkes).
        """
        device = self.device if self.device is not None else torch.device('cpu')

        ts = ts.clone().detach().to(device).requires_grad_(True)
        lambdas = lambdas.clone().detach().to(device).requires_grad_(True)
        Ss = Ss.clone().detach().to(device)

        # Forward pass and transforms
        lambdas_E = torch.log(torch.clamp(torch.exp(lambdas).sum(dim=2), min=1e-6))
        lambdas_E = (lambdas_E - self.means_lamb) / (self.std_lamb + 1e-8)
        lambdas_E[:, self.std_lamb == 0] = 0
        states = torch.hstack([Ss, lambdas_E])
        output = self.ansatz_output(states, ts, model_phi).to(device)

        # dt derivative
        phi_t = torch.autograd.grad(output.sum(), ts, create_graph=True)[0].to(device)

        # derivative wrt lambdas (pre-exp chain)
        grad_lambda = torch.autograd.grad(output.sum(), lambdas, create_graph=True)[0]
        frozen_mask = (self.alphas == 0).unsqueeze(0).expand_as(lambdas)
        grad_lambda = grad_lambda.masked_fill(frozen_mask, 0.0)
        phi_lamb = (torch.exp(lambdas).reshape(lambdas.shape[0], -1) * grad_lambda.reshape(lambdas.shape[0], -1)).to(device)

        # Calculate integral term I_phi
        I_phi = torch.zeros_like(output).to(device)
        for i in range(self.NDIMS):
            Ss_transitioned = self.transition(Ss, torch.tensor(i, dtype=torch.float32, device=device))
            lambdas_transitioned = self.transition_lambda(lambdas, i)
            lambdas_transitioned = torch.log(torch.clamp(torch.exp(lambdas_transitioned).sum(dim=2), min=1e-6))
            lambdas_transitioned = (lambdas_transitioned - self.means_lamb) / (self.std_lamb + 1e-8)
            lambdas_transitioned[:, self.std_lamb == 0] = 0
            states_transitioned = torch.hstack([Ss_transitioned, lambdas_transitioned])
            I_phi += torch.exp(lambdas[:, i, :]).sum(dim=1).reshape(lambdas.shape[0], 1) * (
                    self.ansatz_output(states_transitioned, ts, model_phi) - output
            )

        # Generator term L_phi
        L_phi = phi_t + I_phi + ((self.gammas * (self.mus - torch.exp(lambdas))).reshape(lambdas.shape[0], -1) * phi_lamb).sum(dim=1).reshape(lambdas.shape[0], 1)

        # Running cost f
        states_lob = self.means + Ss[:, :11].clone() * (self.stds + 1e-8)
        f = -self.eta * torch.square(states_lob[:, 1])
        f = f.reshape(-1, 1).to(device)

        # Decision network d
        logits_d = model_d(ts, states)[1]
        if train_d:
            ds_onehot = gumbel_softmax_sample(logits_d, tau=tau, hard=True)  # requires same helper as original
            ds = ds_onehot[:, 1:2]
        else:
            ds = torch.argmax(logits_d, dim=-1).reshape(-1, 1).float().detach()

        # Control network u
        logits_u = model_u(ts, states)[1]
        if not train_u:
            logits_u = logits_u.detach()

        # Intervention value
        transition_lambdas_E = torch.log(torch.clamp(torch.exp(self.transition_lambda(lambdas, logits_u)).sum(dim=2), min=1e-6))
        transition_lambdas_E = (transition_lambdas_E - self.means_lamb) / (self.std_lamb + 1e-8)
        transition_lambdas_E[:, self.std_lamb == 0] = 0
        states_intervened = torch.hstack([self.intervention(ts, Ss, logits_u), transition_lambdas_E])
        M_phi = self.ansatz_output(states_intervened, ts, model_phi)

        # HJB evaluation (residual) -- same sign convention as original
        evaluation = (1 - ds) * (L_phi + f) + ds * (M_phi - output)
        evaluation = evaluation / 1000.0

        # Return per-sample residuals (not squared). RAD uses magnitude.
        return evaluation.detach()  # detach so RAD selection doesn't backprop through sampling

    def rad_sample_hawkes(self, model_phi, model_d, model_u, num_points=None, pool_multiplier=None, alpha=None, seed=None):
        """
        Generate a RAD-sampled interior minibatch for Hawkes case.
        - Creates a candidate pool of size pool_multiplier * num_points via self.sampler(hawkes=True)
        - Computes absolute residual |evaluation| for each candidate
        - Samples num_points from the pool with probability proportional to |eval|^alpha
        Returns: ts_selected, Ss_selected, lambdas_selected (tensors on the device)
        """
        if num_points is None:
            num_points = self.NUM_POINTS
        if pool_multiplier is None:
            pool_multiplier = self.pool_multiplier
        if alpha is None:
            alpha = self.alpha

        pool_size = int(pool_multiplier * num_points)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create pool
        ts_pool, Ss_pool, lambdas_pool = self.sampler(pool_size, hawkes=True)
        # Move to device
        ts_pool = ts_pool.to(self.device)
        Ss_pool = Ss_pool.to(self.device)
        lambdas_pool = lambdas_pool.to(self.device)

        # Compute per-sample interior residuals (absolute)
        with torch.no_grad():
            evals = self._compute_hawkes_interior_evaluation(model_phi, model_d, model_u, ts_pool, Ss_pool, lambdas_pool)
            # evals shape (pool_size, 1)
            residuals = torch.abs(evals).reshape(-1) + 1e-12  # avoid zeros

        # Create sampling probabilities proportional to residual^alpha
        weights = (residuals ** alpha).cpu().numpy()
        weights_sum = weights.sum()
        if weights_sum <= 0 or np.isnan(weights_sum):
            # fallback uniform
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / weights_sum

        # Sample indices (with replacement)
        chosen_idx = np.random.choice(len(probs), size=num_points, replace=True, p=probs)

        # Select corresponding tensors
        ts_selected = ts_pool[chosen_idx].clone().detach()
        Ss_selected = Ss_pool[chosen_idx].clone().detach()
        lambdas_selected = lambdas_pool[chosen_idx].clone().detach()

        return ts_selected, Ss_selected, lambdas_selected

    #########################
    # Override train_step_hawkes
    #########################
    def train_step_hawkes(self, model_phi, optimizer_phi, scheduler_phi, model_d, optimizer_d, scheduler_d, model_u, optimizer_u, scheduler_u, phi_epochs=10, phi_optim='ADAM', freeze_d=False):
        """
        Modified training step which uses RAD sampling for interior points and ignores boundary loss entirely.
        The training loop is similar to the original, but each mini-update uses RAD-sampled interior batches.
        """
        # We'll create batches using rad_sample_hawkes and then use the interior residual computations to form losses.
        # NOTE: This implementation keeps the same structure of optimizing phi, u, d but uses RAD minibatches.
        train_loss_phi = 0.0

        # For phi training, run phi_epochs gradient steps using RAD minibatches
        for j in range(phi_epochs):
            ts_batch, Ss_batch, lambdas_batch = self.rad_sample_hawkes(model_phi, model_d, model_u, num_points=self.NUM_POINTS)
            ts_batch = ts_batch.to(self.device).requires_grad_(True)
            Ss_batch = Ss_batch.to(self.device)
            lambdas_batch = lambdas_batch.to(self.device).requires_grad_(True)

            # compute evaluation (residual) for batch (non-detached, so grads flow into phi)
            evaluation = self._compute_hawkes_interior_evaluation(model_phi, model_d, model_u, ts_batch, Ss_batch, lambdas_batch, train_phi=True)

            # interior loss = MSE(evaluation, 0)
            interior_loss = nn.MSELoss()(evaluation, torch.zeros_like(evaluation))

            # backprop & step
            torch.nn.utils.clip_grad_norm_(model_phi.parameters(), 1.0)
            if phi_optim == 'ADAM':
                optimizer_phi.zero_grad()
                interior_loss.backward()
                optimizer_phi.step()
            else:
                # if LBFGS, define closure (rare for RAD but included)
                def _closure():
                    optimizer_phi.zero_grad()
                    ts_batch2 = ts_batch.clone().detach().requires_grad_(True)
                    eval2 = self._compute_hawkes_interior_evaluation(model_phi, model_d, model_u, ts_batch2, Ss_batch, lambdas_batch)
                    loss2 = nn.MSELoss()(eval2, torch.zeros_like(eval2))
                    loss2.backward()
                    return loss2
                optimizer_phi.step(_closure)

            train_loss_phi = interior_loss.item()
            if scheduler_phi is not None:
                scheduler_phi.step()

            # Safety checks
            if torch.isnan(interior_loss) or torch.isinf(interior_loss):
                print("Warning: NaN or Inf detected in Phi interior loss during RAD training")
                break

            print(f'RAD Model Phi interior loss: {train_loss_phi:0.6f}')
            for param_group in optimizer_phi.param_groups:
                lr = param_group['lr']
                print('Model Phi LR: ' + str(lr))

        # After phi updates, prepare a detached state for u and d training (use statistics computed from lambdas)
        # We will use the full RAD sample again (detached)
        ts_train, Ss_train, lambdas_train = self.rad_sample_hawkes(model_phi, model_d, model_u, num_points=self.NUM_POINTS)
        ts_train = ts_train.to(self.device)
        Ss_train = Ss_train.to(self.device)
        lambdas_E = torch.log(torch.clamp(torch.exp(lambdas_train.to(self.device)).sum(dim=2), min=1e-6))
        lambdas_E = (lambdas_E - self.means_lamb) / (self.std_lamb + 1e-8)
        lambdas_E[:, self.std_lamb == 0] = 0
        states_detached = torch.hstack([Ss_train, lambdas_E]).detach()

        # Train control network u (policy) using evaluation as in original (maximizing negative residual)
        for j in range(phi_epochs):
            # compute evaluation but do not train phi now; we want eval w.r.t current phi (detached)
            # reuse compute function but ensure model_phi is in eval mode to avoid stochastic layers if any
            evaluation = self._compute_hawkes_interior_evaluation(model_phi, model_d, model_u, ts_train, Ss_train, lambdas_train, train_u=True)
            loss_u = -evaluation.mean()

            # entropy regularization (same style as original)
            u_logits = model_u(ts_train, states_detached)[1]
            u_entropy_loss = -(torch.softmax(u_logits + 1e-3, dim=1) * F.log_softmax(u_logits + 1e-20, dim=1)).sum(dim=1).mean()
            loss_u -= 1.0 * u_entropy_loss

            optimizer_u.zero_grad()
            loss_u.backward()
            # gradient checks
            grad_norm = 0.0
            for p in model_u.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            if grad_norm < 1e-8:
                print(f"Warning: vanishing gradients on u (norm={grad_norm:.2e})")
            torch.nn.utils.clip_grad_norm_(model_u.parameters(), 1.0)
            optimizer_u.step()
            if scheduler_u is not None:
                scheduler_u.step()

            if torch.isnan(loss_u) or torch.isinf(loss_u):
                print("Warning: NaN or Inf detected in u loss during RAD training")
                break

            print(f'RAD Model u loss: {loss_u.item():0.6f}')

        # Train decision network d if desired
        for j in range(int(phi_epochs * (1 if freeze_d else 0))):
            evaluation = self._compute_hawkes_interior_evaluation(model_phi, model_d, model_u, ts_train, Ss_train, lambdas_train, train_d=True)
            loss_d = -evaluation.mean()
            d_logits = model_d(ts_train, states_detached)[1]
            d_entropy_loss = -(torch.softmax(d_logits + 1e-3, dim=1) * F.log_softmax(d_logits + 1e-20, dim=1)).sum(dim=1).mean()
            loss_d -= 10 * d_entropy_loss
            optimizer_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), 1.0)
            optimizer_d.step()
            if scheduler_d is not None:
                scheduler_d.step()
            if torch.isnan(loss_d) or torch.isinf(loss_d):
                print("Warning: NaN or Inf detected in d loss during RAD training")
                break
            print(f'RAD Model d loss: {loss_d.item():0.6f}')

        # Return same signature as original
        # compute some basic diagnostics
        pred_u, prob_us = model_u(ts_train, states_detached)
        pred_d, prob_ds = model_d(ts_train, states_detached)

        train_loss_u = 0.0
        train_loss_d = 0.0
        acc_u = 0.0
        acc_d = 0.0

        try:
            print(pred_d.unique(return_counts=True))
            print(pred_u.unique(return_counts=True))
        except Exception:
            pass

        return model_phi, model_d, model_u, train_loss_phi, train_loss_d, train_loss_u, acc_d, acc_u

MM = RADMarketMaking(num_epochs=2000, num_points=1000, hawkes=True)
MM.train(lr =1e-4, ric='INTC', phi_epochs = 5, sampler='iid',log_dir = 'logs', model_dir = 'models', typeNN='LSTM', layer_widths = [50, 50, 50], n_layers= [5,5,5], unified=False, label = 'LSTM_INTC_hawkes_tc100', activation='relu')