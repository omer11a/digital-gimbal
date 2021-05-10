import torch

import itertools

import utils

class KPN(torch.nn.Module):
    ENCODER_DIMS = (64, 128, 256, 512, 512, )
    DECODER_DIMS = (512, 256, )
    LAYER_SCALE_FACTOR = 2

    @staticmethod
    def _build_basic_layers(dims, extra_dims=None):
        if extra_dims is None:
            extra_dims = itertools.repeat(0)

        return torch.nn.ModuleList(
            utils.RepeatedConv(in_channel_num + extra_channel_num, out_channel_num)
            for in_channel_num, out_channel_num, extra_channel_num in zip(dims[:-1], dims[1:], extra_dims)
        )

    def __init__(
        self,
        burst_length,
        channel_num,
        kernel_size,
        blind=True,
        scale_factor=1
    ):
        super().__init__()

        self._blind = blind
        burst_channel_num = burst_length * channel_num
        in_channel_num = burst_channel_num + (0 if blind else 1)

        self._squared_scale_factor = scale_factor ** 2
        self._num_kernel_parameters = kernel_size ** 2
        out_channel_num = burst_channel_num * self._squared_scale_factor * self._num_kernel_parameters

        encoder_dims = (in_channel_num, ) + type(self).ENCODER_DIMS
        self._encoder_layers = type(self)._build_basic_layers(encoder_dims)

        decoder_dims = encoder_dims[-1:] + type(self).DECODER_DIMS + (out_channel_num, )
        extra_dims = encoder_dims[-2:-len(decoder_dims) - 1:-1]
        self._decoder_layers = type(self)._build_basic_layers(decoder_dims, extra_dims=extra_dims)
        self._final_layer = torch.nn.Conv2d(out_channel_num, out_channel_num, 1)

        self._pool_layer = torch.nn.AvgPool2d(type(self).LAYER_SCALE_FACTOR)
        self._upsample_layer = torch.nn.Upsample(
            scale_factor=type(self).LAYER_SCALE_FACTOR,
            mode='bilinear',
            align_corners=False
        )

    def get_hidden_size(self, height, width):
        encoder_scale_factor = type(self).LAYER_SCALE_FACTOR ** (len(self._encoder_layers) - 1)
        return height * width // encoder_scale_factor ** 2

    def forward(self, burst, noise_std=None):
        n, d, c, h, w = burst.shape
        burst = burst.reshape(n, d * c, h, w)
        if not self._blind:
            if noise_std.numel() == n:
                noise_std_channel = noise_std.reshape(n, 1, 1, 1).expand(n, 1, h, w)
            else:
                noise_std_channel = noise_std.reshape(n, 1, h, w)

            burst = torch.cat((burst, noise_std_channel, ), dim=1)

        hidden_features = [torch.empty((), device=burst.device)]
        out = self._encoder_layers[0](burst)
        for layer in self._encoder_layers[1:]:
            out = self._pool_layer(out)
            out = layer(out)
            hidden_features.append(out)

        final_hidden_features = hidden_features.pop()
        for layer in self._decoder_layers:
            out = self._upsample_layer(out)
            out = torch.cat((out, hidden_features.pop()), dim=1)
            out = layer(out)

        out = self._upsample_layer(out)
        out = self._final_layer(out)
        kernels = out.reshape(n, d, c, self._squared_scale_factor, self._num_kernel_parameters, h, w)
        return final_hidden_features, kernels

class KernelConv(torch.nn.Module):
    def  __init__(self, scale_factor=1):
        super().__init__()
        self._scale_factor = scale_factor
        self._pixel_shuffle = torch.nn.PixelShuffle(scale_factor)

    def _shuffle(self, burst):
        n, d, c, r, h, w = burst.shape
        burst = burst.reshape(n * d * c, r, h, w)
        burst = self._pixel_shuffle(burst)
        return burst.reshape(n, d, c, h * self._scale_factor, w * self._scale_factor)

    def forward(self, burst, kernels):
        convolved_burst = utils.adaptive_conv(burst.unsqueeze(3), kernels)
        return self._shuffle(convolved_burst)
