"""
Core diffusion utilities for text generation.
"""

import enum
import math
import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_text_log_likelihood

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Get pre-defined beta schedule."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Create a beta schedule that discretizes the given alpha_t_bar function."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    """Which type of output the model predicts."""
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    """What is used as the model's output variance."""
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    E2E_MSE = enum.auto()
    E2E_KL = enum.auto()

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    """
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        model_arch='transformer',
        training_mode='e2e',
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.model_arch = model_arch
        self.training_mode = training_mode

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """Get the distribution q(x_t | x_0)."""
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_x_start(self, x_start_mean, std):
        """Sample x_start using the reparameterization trick."""
        noise = th.randn_like(x_start_mean)
        return x_start_mean + std * noise

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_mean_variance(self, model, x_t, t, clip_denoised=True, model_kwargs=None):
        """Get model predictions."""
        if model_kwargs is None:
            model_kwargs = {}
            
        B, C = x_t.size(0), x_t.size(-1)
        assert t.shape == (B,)

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        # Split output if model predicts variance
        if self.model_var_type in [ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, x_t.size(1), C * 2)
            model_output, model_var_values = th.split(model_output, C, dim=-1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x_t.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x_t.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x_t.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)

        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
            )
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion posterior."""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


# TRAINING THE EMBEDDINGS IS HERE!


    def training_losses(self, model, x_start, t, model_kwargs=None):
        """Compute training losses for a single timestep."""
        if model_kwargs is None:
            model_kwargs = {}

        input_ids = model_kwargs.pop('input_ids').to(t.device)
        
        # Get embeddings and add noise
        x_start_mean = model.model.module.get_embeds(input_ids)
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                 th.tensor([0]).to(x_start_mean.device),
                                 x_start_mean.shape)
        x_start_log_var = 2 * th.log(std)
        x_start = self.get_x_start(x_start_mean, std)
        noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        if self.loss_type == LossType.E2E_KL:
            # KL loss calculation
            terms = {}
            out = self.p_mean_variance(model, x_t, t, model_kwargs=model_kwargs)
            
            # Calculate KL loss
            target = noise
            terms["loss"] = mean_flat((target - out["pred_xstart"]) ** 2)
            
            # Add token prediction loss
            get_logits = model.model.module.get_logits
            logits = get_logits(x_start)
            token_loss = th.nn.CrossEntropyLoss(reduction='none')(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1)
            ).mean()
            
            terms["loss"] = terms["loss"] + token_loss
            
        else:  # MSE loss
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            target = noise  # Predict noise
            terms = {}
            terms["loss"] = mean_flat((target - model_output) ** 2)
            
            # Add token prediction loss
            get_logits = model.model.module.get_logits
            logits = get_logits(x_start)
            token_loss = th.nn.CrossEntropyLoss(reduction='none')(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1)
            ).mean()
            
            terms["loss"] = terms["loss"] + token_loss

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices."""
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
