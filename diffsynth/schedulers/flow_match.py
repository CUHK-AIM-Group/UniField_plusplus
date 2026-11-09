import torch



class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def sigma_to_timestep(self, sigma):
        """Convert effective flow sigma to the DiT timestep coordinate.

        ``self.sigmas`` already contains the configured schedule shift.  The
        model timestep is defined from that effective sigma, so this mapping is
        linear and must not apply ``shift`` a second time.
        """
        return sigma * self.num_train_timesteps


    def timestep_to_sigma(self, timestep):
        """Convert a DiT timestep to effective flow sigma linearly."""
        return timestep / self.num_train_timesteps


    def _sigma_for_sample(self, timestep, sample):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep)
        sigma = self.timestep_to_sigma(timestep).to(
            device=sample.device, dtype=sample.dtype
        )
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigma_to_timestep(self.sigmas)
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id].to(device=sample.device, dtype=sample.dtype)
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].to(device=sample.device, dtype=sample.dtype)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        sigma = self._sigma_for_sample(timestep, sample)
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep):
        sigma = self._sigma_for_sample(timestep, original_samples)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    

    def predict_x0_from_v(self, v_pred, noisy_sample, timestep):
        """
        从速度场预测一步恢复干净 latent
        
        推导：
        加噪公式: x_t = (1 - σ_t) * x_0 + σ_t * ε
        速度场: v = ε - x_0
        因此: x_t = x_0 + σ_t * v
        所以: x_0 = x_t - σ_t * v_pred
        
        Args:
            v_pred: 模型预测的速度场
            noisy_sample: 加噪后的样本 x_t
            timestep: 时间步
        
        Returns:
            x_0: 预测的干净 latent
        """
        sigma = self._sigma_for_sample(timestep, v_pred)
        x_0 = noisy_sample - sigma * v_pred
        return x_0
    

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
