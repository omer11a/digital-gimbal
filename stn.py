import torch
import torchfields
import numpy as np
import kornia

import utils

class STN(torch.nn.Module):
    INTER_CHANNEL_NUMS = (100, 50, 25, )
    KERNEL_SIZE = 5
    DEFAULT_NUM_SCALES = 5

    def _get_flow(self, level):
        _, c, h, w = level.shape
        cnn_input = level.reshape(-1, self._burst_length * c, h, w)
        return self._cnn(cnn_input).reshape(-1, 2, h, w).field().from_pixels()

    def _apply_flow(self, level, flow):
        _, c, h, w = level.shape
        level = level.reshape(-1, self._burst_length, c, h, w)
        reference_frame = self.get_reference_frame(level)
        level = self.exclude_reference_frame(level).reshape(-1, c, h, w)
        level = flow(level).reshape(-1, self._num_frames_to_register, c, h, w)
        return self.include_reference_frame(level, reference_frame).reshape(-1, c, h, w)

    def __init__(self, burst_length, channel_num, reference_index=None):
        super().__init__()
        self._burst_length = burst_length
        self._reference_index = reference_index
        self._num_frames_to_register = burst_length if reference_index is None else burst_length - 1

        in_channel_num = burst_length * channel_num
        out_channel_num = self._num_frames_to_register * 2
        channel_nums = (in_channel_num, *type(self).INTER_CHANNEL_NUMS, out_channel_num, )
        self._cnn = utils.CNN(channel_nums, type(self).KERNEL_SIZE, end_with_relu=False)

    def exclude_reference_frame(self, burst):
        if self._reference_index is None:
            return burst

        chunks = (
            burst[:, :self._reference_index],
            burst[:, self._reference_index + 1:],
        )

        return torch.cat(chunks, dim=1)

    def get_reference_frame(self, burst):
        if self._reference_index is None:
            return None

        return burst[:, self._reference_index]

    def include_reference_frame(self, burst, reference_frame):
        if self._reference_index is None:
            return burst

        chunks = (
            burst[:, :self._reference_index],
            reference_frame.unsqueeze(1),
            burst[:, self._reference_index:],
        )

        return torch.cat(chunks, dim=1)

    def forward(self, burst, num_scales=DEFAULT_NUM_SCALES):
        n, d, c, h, w = burst.shape
        burst = burst.reshape(n * d, c, h, w)
        pyramid = kornia.build_pyramid(burst, num_scales - 1)[::-1]
        flow = self._get_flow(pyramid[0])
        for level in pyramid[1:]:
            flow = kornia.pyrup(flow).field()
            level = self._apply_flow(level, flow)
            flow += self._get_flow(level)

        return self._apply_flow(burst, flow).reshape(n, d, c, h, w)
