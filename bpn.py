import torch
import torchfields
import kornia
import scipy.constants

import itertools

import estimators
import kpn
import stn

class FM(torch.nn.Module):
    DEFAULT_MIN_TIME = 0
    DEFAULT_DELAY = 0
    DEFAULT_PIXEL_SIZE = 10e-6
    DEFAULT_QUANTUM_EFFICIENCY = 0.5
    DEFAULT_WAVELENGTH = 520e-9
    DEFAULT_DARK_CURRENT = 10
    DEFAULT_SHOT_N = 1500,
    DEFAULT_SHOT_THRESHOLD = 1000
    DEFAULT_READ_STD = 18
    DEFAULT_FWC = 12e3
    DEFAULT_RESPONSE_TAIL = 0.1
    DEFAULT_GAIN = 0.07
    DEFAULT_PRNU = 0.02
    DEFAULT_NUM_BITS = 10

    @staticmethod
    def _resample(samples, times):
        grid = kornia.create_meshgrid(*samples.shape[-2:]).to(times.device)
        grid = grid.unsqueeze(1).expand((samples.size(0), times.size(0), -1, -1, -1, ))

        mean = (times[-1] + times[0]) / 2
        std = times[-1] - mean
        normalized_times = (times - mean) / std
        normalized_times = normalized_times.reshape(1, -1, 1, 1).expand(grid.shape[:-1])
        new_grid = torch.cat((grid, normalized_times.unsqueeze(-1), ), dim=-1)

        new_samples = torch.nn.functional.grid_sample(
            samples.transpose(1, 2), new_grid, align_corners=True)
        return new_samples.transpose(1, 2)

    @staticmethod
    def _get_std(burst, std):
        return torch.as_tensor(std, device=burst.device).reshape(1, 1, 1, 1, 1).expand_as(burst)

    def _create_weights(self, min_time, delay):
        weights = torch.zeros(self._target_length)
        if self._uniform:
            self.register_buffer('_weights', weights)
        else:
            self._weights = torch.nn.Parameter(weights)

        min_percentage = torch.as_tensor(min_time / self._max_total_time)
        min_percentages = min_percentage.expand(self._target_length)
        if not self._fix_total_time:
            self._padding = torch.nn.Parameter(torch.zeros(1))
            zero_percent = torch.zeros(1)
            min_percentages = torch.cat((zero_percent, min_percentages, zero_percent, ))

        self.register_buffer('_min_percentages', min_percentages)

        delay_percentage = torch.as_tensor(delay / self._max_total_time)
        unlearned_percentage = (min_percentage + delay_percentage) * self._target_length
        self.register_buffer('_delay_percentages', delay_percentage.expand(self._target_length))
        self.register_buffer('_learned_percentage', 1 - unlearned_percentage)

    def _get_times(self, source_length):
        device = self._weights.device
        sample_division = self.get_time_division(include_padding=True, include_delay=True) * (source_length - 1)
        event_markers = sample_division[:-1].cumsum(0)
        num_new_indices = event_markers.numel()
        new_indices = torch.ceil(event_markers).long() + torch.arange(num_new_indices, device=device)
        new_grid_size = source_length + num_new_indices
        sample_grid = torch.zeros(new_grid_size, device=device)
        mask = torch.zeros_like(sample_grid, dtype=torch.bool, device=device)
        mask[new_indices] = True
        sample_grid[mask] = event_markers
        sample_grid[~mask] = torch.arange(source_length, dtype=torch.float, device=device)

        integration_indices = torch.cat((
             torch.zeros(1, dtype=torch.long, device=device),
             new_indices,
             torch.as_tensor((new_grid_size, ), dtype=torch.long, device=device),
        ))

        fps = (source_length - 1) / self._max_total_time
        return sample_grid / fps, integration_indices

    def _integrate(self, samples):
        source_length = samples.size(1)
        if self._uniform and self._fix_total_time and source_length == self._target_length:
            return samples

        times, indices = self._get_times(source_length)
        with torch.no_grad():
            samples = self._resample(samples, times)

        indices = indices if self._fix_total_time else indices[1:-1]
        delta = 2 if self._has_delay else 1
        burst = [
            torch.trapz(samples[:, start:end + 1], times[start:end + 1], dim=1)
            for start, end in zip(indices[:-1:delta], indices[1::delta])
        ]

        return torch.stack(burst, dim=1)

    def _inject_noise(self, burst, t=None, max_exposure=False):
        if self._uniform and self._fix_total_time:
            t = None

        time_division = torch.ones(1, device=burst.device) if max_exposure else self.get_time_division()
        integrated_dark_current = time_division.reshape(-1, 1, 1, 1) * self._max_total_time * self._dark_current
        burst = self._shot_noise(burst + integrated_dark_current, t=t)
        read_std = type(self)._get_std(burst, self._read_std)
        return burst + torch.normal(mean=0, std=read_std) - integrated_dark_current

    def _respond(self, burst):
        n, _, c, h, w = burst.shape
        diff = (burst - self._threshold) / self._gap
        non_linear_response = self._threshold + (1 - torch.exp(-diff)) * self._gap
        burst = self._gain * torch.where(burst < self._threshold, burst, non_linear_response)
        fpn = self._prnu * torch.empty(n, 1, c, h, w, device=burst.device).uniform_(-0.5, 0.5)
        return burst * (1 + fpn)

    def _quantize(self, burst):
        burst = estimators.round_ste(burst)
        return torch.clamp(burst / self._max_value, 0, 1)

    def _apply_to_burst(self, burst, t=None, max_exposure=False):
        noisy_burst = self._inject_noise(self._i2e * burst, t=t, max_exposure=max_exposure)
        return self._quantize(self._respond(noisy_burst))

    def __init__(
        self,
        target_length,
        max_total_time,
        uniform=False,
        fix_total_time=False,
        min_time=DEFAULT_MIN_TIME,
        delay=DEFAULT_DELAY,
        pixel_size=DEFAULT_PIXEL_SIZE,
        quantum_efficiency=DEFAULT_QUANTUM_EFFICIENCY,
        wavelength=DEFAULT_WAVELENGTH,
        dark_current=DEFAULT_DARK_CURRENT,
        shot_n=DEFAULT_SHOT_N,
        shot_threshold=DEFAULT_SHOT_THRESHOLD,
        read_std=DEFAULT_READ_STD,
        fwc=DEFAULT_FWC,
        response_tail=DEFAULT_RESPONSE_TAIL,
        gain=DEFAULT_GAIN,
        prnu=DEFAULT_PRNU,
        num_bits=DEFAULT_NUM_BITS
    ):
        super().__init__()
        self._target_length = target_length
        self._max_total_time = max_total_time
        self._uniform = uniform
        self._fix_total_time = fix_total_time
        self._has_delay = delay > 0
        self._i2e = pixel_size ** 2 * quantum_efficiency * wavelength / scipy.constants.h /  scipy.constants.c
        self._dark_current = dark_current
        self._shot_noise = estimators.ReparameterizedPoisson(n=shot_n, threshold=shot_threshold)
        self._read_std = read_std
        self._threshold = (1 - response_tail) * fwc
        self._gap = response_tail * fwc
        self._gain = gain
        self._prnu = prnu
        self._max_value = 2 ** num_bits - 1
        self._create_weights(min_time, delay)

    def get_time_division(self, include_padding=False, include_delay=False):
        weights = self._weights
        if not self._fix_total_time:
            side_padding = self._padding / 2
            weights = torch.cat((side_padding, weights, side_padding, ))

        time_division = torch.nn.functional.softmax(weights, dim=0)
        time_division = self._learned_percentage * time_division + self._min_percentages
        if not (self._has_delay and include_delay):
            return time_division if (self._fix_total_time or include_padding) else time_division[1:-1]

        if not self._fix_total_time:
            pre_padding = time_division[:1]
            time_division = time_division[1:-1]
            post_padding = time_division[-1:]

        time_division = torch.stack((time_division, self._delay_percentages, )).transpose(0, 1).reshape(-1)
        if self._fix_total_time or not include_padding:
            return time_division

        return torch.cat((pre_padding, time_division, post_padding, ))

    def get_exposure_times(self):
        return self._max_total_time * self.get_time_division()

    def predict_noise_std(self, image, factor=1):
        with torch.no_grad():
            burst = torch.nn.functional.relu(image)
            factor *= self._gain / self._max_value
            noise_variance = factor * (image + factor * self._read_std ** 2)
            return torch.sqrt(noise_variance)

    def make_max_exposure_image(self, samples):
        with torch.no_grad():
            burst = self._max_total_time * samples.mean(dim=1, keepdim=True)
            return self._apply_to_burst(burst, max_exposure=True).squeeze(1)

    def forward(self, samples, t=None):
        return self._apply_to_burst(self._integrate(samples), t=t)

class BPN(torch.nn.Module):
    DEFAULT_NOISE_STD = 0
    EPS = 1e-8

    def _apply_fm(self, samples, force=False, t=None, snr=None):
        if self.training or force:
            burst = self._fm(samples, t=t)
            max_exposure_image = self._fm.make_max_exposure_image(samples)
        else:
            burst = samples
            max_exposure_image = torch.zeros_like(samples[:, 0], requires_grad=False)

        if self.training:
            snr = samples.sqrt().mean(dim=(1, 2, 3, 4, ))
        else:
            snr = snr * torch.ones(samples.size(0), device=samples.device, requires_grad=False)

        return burst, max_exposure_image, snr

    def _normalize_burst(self, burst):
        if not self._normalize:
            noise_std = self._fm.predict_noise_std(burst[:, self._reference_index])
            return burst, noise_std

        weights = self.get_time_division()
        burst = burst.div(weights.reshape(-1, 1, 1, 1))
        factor = 1 / weights[self._reference_index]
        noise_std = self._fm.predict_noise_std(burst[:, self._reference_index], factor=factor)
        return burst, noise_std

    def _register_burst(self, burst):
        if not self._registration:
            return burst

        return self._stn(burst)

    def _predict_kernels(self, burst, registered_burst, noise_std=None):
        burst_to_merge = registered_burst
        if self._registration and self._use_unregistered_burst:
            unregistered_burst = self._stn.exclude_reference_frame(burst)
            burst_to_merge = torch.cat((burst_to_merge, unregistered_burst, ), dim=1)

        _, kernels = self._kpn(burst_to_merge, noise_std=noise_std)
        return burst_to_merge, kernels

    def _merge_burst(self, burst, kernels, singular=False):
        kernel_sum = kernels.sum(dim=-3, keepdim=True)
        if singular:
            burst = burst[:, :self._burst_length]
            kernels = kernels.sum(dim=-3, keepdim=True)
            total_sum = kernels.sum(dim=1, keepdim=True)
            kernels = kernels[:, :self._burst_length]
            kernels = kernels / (kernels.sum(dim=1, keepdim=True) + type(self).EPS) * total_sum

        convolved_burst = self._kernel_conv(burst, kernels)
        output = convolved_burst.mean(dim=1)
        return convolved_burst, output

    def __init__(
        self,
        burst_length,
        channel_num,
        max_total_time,
        kernel_size,
        uniform=False,
        fix_total_time=False,
        min_time=FM.DEFAULT_MIN_TIME,
        delay=FM.DEFAULT_DELAY,
        pixel_size=FM.DEFAULT_PIXEL_SIZE,
        quantum_efficiency=FM.DEFAULT_QUANTUM_EFFICIENCY,
        wavelength=FM.DEFAULT_WAVELENGTH,
        dark_current=FM.DEFAULT_DARK_CURRENT,
        shot_n=FM.DEFAULT_SHOT_N,
        shot_threshold=FM.DEFAULT_SHOT_THRESHOLD,
        read_std=FM.DEFAULT_READ_STD,
        fwc=FM.DEFAULT_FWC,
        response_tail=FM.DEFAULT_RESPONSE_TAIL,
        gain=FM.DEFAULT_GAIN,
        prnu=FM.DEFAULT_PRNU,
        num_bits=FM.DEFAULT_NUM_BITS,
        normalize=True,
        reference_index=None,
        registration=True,
        use_unregistered_burst=False,
        blind=True,
        scale_factor=1
    ):
        super().__init__()

        self._burst_length = burst_length
        self._normalize = normalize
        self._registration = registration

        self._fm = FM(
            burst_length, max_total_time, uniform=uniform, fix_total_time=fix_total_time,
            min_time=min_time, delay=delay, pixel_size=pixel_size,
            quantum_efficiency=quantum_efficiency, wavelength=wavelength, dark_current=dark_current,
            shot_n=shot_n, shot_threshold=shot_threshold, read_std=read_std, fwc=fwc,
            response_tail=response_tail, gain=gain, prnu=prnu, num_bits=num_bits
        )

        if registration and use_unregistered_burst:
            num_images_to_merge = 2 * burst_length - 1
        else:
            num_images_to_merge = burst_length

        self._kpn = kpn.KPN(num_images_to_merge, channel_num, kernel_size, blind=blind,
            scale_factor=scale_factor)

        self._reference_index = reference_index
        if registration:
            self._use_unregistered_burst = use_unregistered_burst
            self._stn = stn.STN(burst_length, channel_num, reference_index=reference_index)

        self._kernel_conv = kpn.KernelConv(scale_factor=scale_factor)

    def forward(
        self,
        samples,
        force_fm=False,
        t=None,
        snr=None
    ):
        burst, max_exposure_image, snr = self._apply_fm(samples, force=force_fm, t=t, snr=snr)
        normalized_burst, _ = self._normalize_burst(burst)
        registered_burst = self._register_burst(normalized_burst)
        burst_to_merge, kernels = self._predict_kernels(normalized_burst, registered_burst,
            noise_std=snr)
        convolved_burst1, output1 = self._merge_burst(burst_to_merge, kernels, singular=True)
        convolved_burst, output = self._merge_burst(burst_to_merge, kernels)
        return (burst, max_exposure_image, normalized_burst, registered_burst, convolved_burst1,
            output1, convolved_burst, output)

    def get_time_division(self, include_padding=False, include_delay=False):
        return self._fm.get_time_division(include_padding=include_padding, include_delay=include_delay)

    def get_exposure_times(self):
        return self._fm.get_exposure_times()

    def get_forward_model_parameters(self):
        return self._fm.parameters()

    def get_reconstruction_parameters(self):
        if not self._registration:
            return self._kpn.parameters()

        return itertools.chain(self._stn.parameters(), self._kpn.parameters())
