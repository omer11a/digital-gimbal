import torch
import kornia
import torchfields
import numpy as np
import pytorch3d.transforms

import collections
import functools

import utils

DEFAULT_THRESHOLD = 0.0031308
DEFAULT_A = 0.055
DEFAULT_MULT = 12.92
DEFAULT_GAMMA = 2.4
EPSILON = 0.001

BidirectionalFlow = collections.namedtuple('BidirectionalFlow', ('forward', 'backward', ))

def tensor_to_tuple(tensor):
    return tuple(int(element) for element in tensor)

def resize(tensor, size, blur=True, align_corners=False):
    old_shape = torch.as_tensor(tensor.shape[-2:], dtype=torch.float)
    new_shape = torch.as_tensor(size, dtype=torch.float)
    if torch.all(new_shape == old_shape):
        return tensor

    if blur:
        sigma = old_shape / new_shape / 3
        kernel_size = 2 * sigma.int() + 1
        if torch.any(kernel_size > 1):
            tensor = kornia.gaussian_blur2d(
                tensor,
                tensor_to_tuple(kernel_size),
                tensor_to_tuple(sigma)
            )

    return torch.nn.functional.interpolate(
        tensor,
        size=size,
        mode='bilinear',
        align_corners=align_corners,
        recompute_scale_factor=True
    )

def apply_gamma_correction(
    tensor,
    threshold=DEFAULT_THRESHOLD,
    a=DEFAULT_A,
    mult=DEFAULT_MULT,
    gamma=DEFAULT_GAMMA
):
    tensor = tensor.clamp(0, 1)
    result = torch.zeros_like(tensor)
    mask = tensor > threshold
    not_mask = ~mask
    result[mask] = (1 + a) * torch.pow(tensor[mask] + EPSILON, 1 / gamma) - a
    result[not_mask] = tensor[not_mask] * mult
    return result

def invert_gamma_correction(
    tensor,
    threshold=DEFAULT_THRESHOLD,
    a=DEFAULT_A,
    mult=DEFAULT_MULT,
    gamma=DEFAULT_GAMMA
):
    result = torch.zeros_like(tensor)
    mask = tensor > threshold
    not_mask = ~mask
    result[not_mask] = tensor[not_mask] / mult
    result[mask] = torch.pow((tensor[mask] + a) / (1 + a), gamma)
    return result

class BaseTransform:
    def __call__(self, images, flows=None):
        raise NotImplementedError()

class Compose(BaseTransform):
    def __init__(self, transforms):
        super().__init__()
        self._transforms = transforms

    def __call__(self, images, flows=None):
        for transform in self._transforms:
            images, flows = transform(images, flows=flows)

        return images, flows

class ToTensor(BaseTransform):
    DEFAULT_NUM_BITS = 8
    DEFAULT_FLOW_FORMAT = 'fb'
    FORMAT_LETTERS = {'forward' : 'f', 'backward': 'b'}

    def _make_tensor(self, arrays):
        if torch.is_tensor(arrays):
            tensor = arrays
        else:
            stacked_arrays = np.stack(arrays)
            if stacked_arrays.dtype == np.uint16:
                stacked_arrays = stacked_arrays.astype(np.int16)

            tensor = torch.from_numpy(stacked_arrays)
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(-1)
            tensor = tensor.permute(0, 3, 1, 2)

        return tensor.float()

    def __init__(self, num_bits=DEFAULT_NUM_BITS, generate_flow=False, flow_format=None):
        super().__init__()
        self._white_level = 2 ** num_bits
        self._generate_flow = generate_flow
        flow_format = type(self).DEFAULT_FLOW_FORMAT if flow_format is None else flow_format
        self._flow_indices = {
            key: flow_format.find(letter) if letter in flow_format else None
            for key, letter in type(self).FORMAT_LETTERS.items()
        }

    def __call__(self, images, flows=None):
        images = self._make_tensor(images) / self._white_level
        if flows is None and not self._generate_flow:
            return images, flows

        d = images.size(0)
        size = images.shape[-2:]
        if flows is None:
            flows = torch.zeros(d, 2, 2, *size, device=images.device)
        else:
            flows = resize(self._make_tensor(flows), size, align_corners=True, blur=False)
            flows = flows.reshape(d, -1, 2, *size)

        flow_init_dict = {
            key: None if index is None else flows[:, index].field()
            for key, index in self._flow_indices.items()
        }

        return images, BidirectionalFlow(**flow_init_dict)

class RGBToGrayscale(BaseTransform):
    def __call__(self, images, flows=None):
        images = kornia.rgb_to_grayscale(images)
        return images, flows

class InvertGammaCorrection(BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def __call__(self, images, flows=None):
        images = invert_gamma_correction(images, **self._kwargs)
        return images, flows

class MultiplyColor(BaseTransform):
    def __init__(self, range_min, range_max, independent=False):
        super().__init__()
        self._range_min = range_min
        self._range_max = range_max
        self._independent = independent

    def __call__(self, images, flows=None):
        n = images.size(0)
        if self._independent:
            values = torch.empty(n, device=images.device).uniform_(
                self._range_min, self._range_max)
        else:
            values = torch.empty(1, device=images.device).uniform_(
                self._range_min, self._range_max).expand(n)

        return images * values.reshape(n, 1, 1, 1), flows

class Replicate(BaseTransform):
    def _apply_flow(self, flows, backwards):
        if flows is None:
            return flows

        n, _, h, w = flows.shape
        new_flows = torch.zeros(n * self._num_replicas, 2, h, w, device=flows.device)
        if backwards:
            new_flows[self._num_replicas - 1::self._num_replicas] = flows
        else:
            new_flows[::self._num_replicas] = flows

        return new_flows.field()

    def __init__(self, num_replicas):
        super().__init__()
        self._num_replicas = num_replicas

    def __call__(self, images, flows=None):
        images = images.repeat_interleave(self._num_replicas, dim=0)
        if flows is not None:
            forward = self._apply_flow(flows.forward, False)
            backward = self._apply_flow(flows.backward, True)
            flows = BidirectionalFlow(
                forward,
                backward
            )

        return images, flows

class DualTransform(BaseTransform):
    def _get_params(self, images):
        return tuple()

    def _apply(self, tensor, *params):
        raise NotImplementedError()

    def _update_flow(self, flows, backwards, *params):
        return flows

    def _apply_flow(self, flows, backwards, *params):
        if flows is None:
            return flows

        flows = self._apply(flows, *params)
        return self._update_flow(flows, backwards, *params)

    def __init__(self):
        super().__init__()
        self._in_pixels = False

    def __call__(self, images, flows=None):
        params = self._get_params(images)
        images = self._apply(images, *params)
        if flows is not None:
            forward = flows.forward
            backward = flows.backward
            if self._in_pixels:
                if forward is not None:
                    forward = forward.pixels()
                if backward is not None:
                    backward = backward.pixels()

            forward = self._apply_flow(forward, False, *params)
            backward = self._apply_flow(backward, True, *params)
            if self._in_pixels:
                if forward is not None:
                    forward = forward.from_pixels()
                if backward is not None:
                    backward = backward.from_pixels()

            flows = BidirectionalFlow(forward, backward)

        return images, flows

class Resize(DualTransform):
    def _apply(self, tensor):
        return resize(tensor, self._size, blur=self._blur)

    def _update_flow(self, flows, backwards, *params):
        return flows.field()

    def __init__(self, size, blur=True):
        super().__init__()
        self._size = size
        self._blur = blur

class RandomCrop(DualTransform):
    def _get_random_coordinate(self, size, crop_size):
        if crop_size >= size:
            return 0

        return torch.randint(0, size - crop_size, (1, ))

    def _get_params(self, images):
        height, width = images.shape[-2:]
        crop_height, crop_width = self._size
        x = self._get_random_coordinate(width, crop_width)
        y = self._get_random_coordinate(height, crop_height)
        return (crop_height, crop_width, x, y, )

    def _apply(self, tensor, crop_height, crop_width, x, y):
        return tensor[..., y:y + crop_height, x : x + crop_width]

    def __init__(self, size):
        super().__init__()
        self._in_pixels = True
        self._size = size

class RandomFlip(DualTransform):
    DEFAULT_P = 0.5

    def _get_params(self, images):
        return (self._p < torch.rand(1), )

    def _apply(self, tensor, should_flip):
        if not should_flip:
            return tensor

        return torch.flip(tensor, dims=(self._dim_to_flip, ))

    def _update_flow(self, flows, backwards, should_flip):
        flows = flows.field()
        if not should_flip:
            return flows

        flows[:, self._index_to_neg] *= -1
        return flows

    def __init__(self, p=DEFAULT_P, vertical=False):
        super().__init__()
        self._p = p
        if vertical:
            self._dim_to_flip = 2
            self._index_to_neg = 1
        else:
            self._dim_to_flip = 3
            self._index_to_neg = 0

class RandomTranslate(DualTransform):
    def _get_params(self, images):
        n = images.size(0)
        random_n = n if self._independent else 1
        translation = torch.empty(random_n, 2, device=images.device)
        if self._std == 0:
            translation.zero_()
        else:
            translation.normal_(std=self._std)

        translation = translation.expand(n, 2)
        if self._accumulative:
            translation = translation.cumsum(0)

        return (translation, )

    def __init__(self, std, independent=False, accumulative=False):
        super().__init__()
        self._in_pixels = True
        self._std = std
        self._independent = independent
        self._accumulative = accumulative

    def _apply(self, tensor, translation):
        return kornia.translate(tensor, translation)

    def _update_flow(self, flows, backwards, translation):
        translation = translation.reshape(-1, 2, 1, 1)
        diffs = translation[1:] - translation[:-1]
        if backwards:
            flows[:-1] = flows[:-1] + diffs
        else:
            flows[1:] = flows[1:] - diffs

        return flows.field()

class OpticTransform(DualTransform):
    @staticmethod
    def _pixels_to_points(pixels, f, c, homogeneous=False):
        points = (pixels - c) / f
        if homogeneous:
            points = kornia.convert_points_to_homogeneous(points)

        return points

    @staticmethod
    def _points_to_pixels(points, f, c, homogeneous=False):
        if homogeneous:
            points = kornia.convert_points_from_homogeneous(points)

        return f * points + c

    def _adapt_intrinsics(self, h, w, device=None):
        size = torch.as_tensor((w, h, ), dtype=torch.float, device=device)
        scale = size / self._size.to(device=device)
        f = scale * self._f.to(device=device)
        c = scale * self._c.to(device=device)
        return f, c

    def _get_points(self, h, w, homogeneous=False, device=None):
        pixels = kornia.create_meshgrid(h, w, normalized_coordinates=False, device=device)
        f, c = self._adapt_intrinsics(h, w, device=device)
        points = type(self)._pixels_to_points(pixels, f, c, homogeneous=homogeneous)
        return points, f, c

    def _convert_homogrpahy_to_flow(self, tensor, homography, depth=1, is_inverse=False):
        if not is_inverse:
            homography = homography.inverse()

        n, _, h, w = tensor.shape
        points, f, c = self._get_points(h, w, homogeneous=True, device=tensor.device)
        points = points.expand(n, -1, -1, -1).reshape(n, -1, 3, 1)
        points = torch.matmul(homography.unsqueeze(1), points).reshape(n, h, w, 3)
        pixels = type(self)._points_to_pixels(points, f, c, homogeneous=True)
        pixels = kornia.normalize_pixel_coordinates(pixels, h, w)
        return pixels.permute(0, 3, 1, 2).field().from_mapping()

    def __init__(self, f, c, size):
        super().__init__()
        self._f = torch.Tensor(f)
        self._c = torch.Tensor(c)
        self._size = torch.as_tensor(size[::-1], dtype=torch.float)

class RandomRotate(OpticTransform):
    @staticmethod
    def _get_random_angle(std, n=1, device=None):
        std = np.radians(std)
        angles = torch.empty(n, 1, device=device)
        if std == 0:
            return angles.zero_()

        return angles.normal_(std=std)

    def _accumulate(self, matrices, reverse=False):
        if reverse:
            matrices = matrices.inverse().flip(0)

        if self._accumulative:
            for i in range(1, matrices.size(0)):
                matrices[i] = torch.matmul(matrices[i], matrices[i - 1])

        if reverse:
            matrices = matrices.flip(0)

        return matrices

    def _get_params(self, images):
        n = images.size(0)
        if self._centeralized:
            n = n - 1

        random_n = n if self._independent else 1
        roll = self._get_random_angle(self._roll, n=random_n, device=images.device)
        pitch = self._get_random_angle(self._pitch, n=random_n, device=images.device)
        yaw = self._get_random_angle(self._yaw, n=random_n, device=images.device)
        rpy = torch.cat((roll, pitch, yaw, ), dim=-1).expand(n, 3)
        if self._centeralized:
            mid = n // 2
            rpy = torch.cat((rpy[:mid], torch.zeros(1, 3, device=rpy.device), rpy[mid:]))

        matrices = pytorch3d.transforms.euler_angles_to_matrix(rpy, 'ZXY')
        if self._centeralized:
            matrices[:mid] = self._accumulate(matrices[:mid], reverse=True)
            matrices[mid + 1:] = self._accumulate(matrices[mid + 1:])
        else:
            matrices = self._accumulate(matrices)

        rotate = self._convert_homogrpahy_to_flow(images, matrices, depth=self._depth)
        unrotate = self._convert_homogrpahy_to_flow(images, matrices, depth=self._depth,
            is_inverse=True)

        return rotate, unrotate

    def _apply(self, tensor, rotate, unrotate):
        return rotate(tensor)

    def _update_flow(self, flows, backwards, rotate, unrotate):
        n, _, h, w = flows.shape
        if backwards:
            valid_flows = flows[:-1]
            unrotate = unrotate[1:]
            rotate = rotate[:-1]
        else:
            valid_flows = flows[1:]
            unrotate = unrotate[:-1]
            rotate = rotate[1:]

        updated = rotate(valid_flows(unrotate))
        if backwards:
            flows[:-1] = updated
        else:
            flows[1:] = updated

        return flows.field()

    def __init__(
        self,
        f,
        c,
        size,
        roll,
        pitch=0,
        yaw=0,
        depth=1,
        independent=False,
        accumulative=False,
        centeralized=False
    ):
        super().__init__(f, c, size)
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw
        self._depth = depth
        self._independent = independent
        self._accumulative = accumulative
        self._centeralized = centeralized

class Distort(OpticTransform):
    def _distort_pixels(self, points, f, c, inverse=False):
        r2 = points.norm(dim=-1) ** 2
        factor = 1 + self._k[0] * r2 + self._k[1] * r2 ** 2 + self._k[2] * r2 ** 3
        if inverse:
            factor = 1 / factor

        distorted_points = factor.unsqueeze(-1) * points
        distorted_pixels = type(self)._points_to_pixels(distorted_points, f, c)
        return distorted_pixels.permute(0, 3, 1, 2).field().from_pixel_mapping()

    def _get_params(self, images):
        h, w = images.shape[-2:]
        points, f, c = self._get_points(h, w, device=images.device)
        distort = self._distort_pixels(points, f, c)
        undistort = self._distort_pixels(points, f, c, inverse=True)
        return distort, undistort

    def _apply(self, tensor, distort, undistort):
        return distort.expand(tensor.size(0), -1, -1, -1)(tensor)

    def _update_flow(self, flows, backwards, distort, undistort):
        return flows(undistort.expand(flows.size(0), -1, -1, -1))

    def __init__(self, f, c, size, k):
        super().__init__(f, c, size)
        self._k = k

class Blur(OpticTransform):
    def _apply(self, tensor):
        points, f, _ = self._get_points(*tensor.shape[-2:], device=tensor.device)
        r = points.norm(dim=-1, keepdim=True)
        alpha = torch.atan(r)
        sigma = f / torch.cos(alpha) ** 2
        quotient = self._grid.to(device=tensor.device) / (2 * sigma.unsqueeze(1).unsqueeze(2) ** 2)
        kernels = torch.exp(quotient.sum(dim=-1))
        normalized_kernels = kernels / kernels.sum(dim=-3, keepdim=True)
        return utils.adaptive_conv(tensor, normalized_kernels, padding='replicate')

    def __init__(self, f, c, size, kernel_size):
        super().__init__(f, c, size)
        grid = kornia.create_meshgrid(kernel_size, kernel_size, normalized_coordinates=False)
        self._grid = grid.reshape(1, 1, kernel_size ** 2, 1, 1, 2) - kernel_size / 2

class Interpolate(BaseTransform):
    @staticmethod
    def _get_flow_coefs(num_subframes, backwards=False, dtype=None, device=None):
        if backwards:
            start = num_subframes - 1
            end = -1
            step = -1
        else:
            start = 0
            end = num_subframes
            step = 1

        steps = torch.arange(start=start, end=end, step=step, dtype=dtype, device=device)
        return steps.reshape(num_subframes, 1, 1, 1) / num_subframes

    @classmethod
    def _subsample(cls, frames, flows, num_subframes, backwards=False, weighted=False):
        n, c, h, w = frames.shape
        frames = frames.unsqueeze(1).expand(n, num_subframes, c, h, w).flatten(end_dim=1)
        coefs = cls._get_flow_coefs(num_subframes, backwards=backwards, dtype=flows.dtype, device=flows.device)
        flows = coefs * flows.unsqueeze(1)
        flows = flows.flatten(end_dim=1).field()
        frames = flows(frames).reshape(n, num_subframes, c, h, w)
        if weighted:
            frames *= 1 - coefs

        return frames.reshape(n * num_subframes, c, h, w)

    @classmethod
    def _interpolate(cls, frames, flows, num_subframes, backwards=False, weighted=False):
        if backwards:
            extra_frame = frames[0]
            frames = frames[1:]
            flows = flows[1:]
        else:
            extra_frame = frames[-1]
            frames = frames[:-1]
            flows = flows[:-1]

        frames = cls._subsample(frames, flows, num_subframes, backwards=backwards, weighted=weighted)
        extra_frame = extra_frame.unsqueeze(0)
        if backwards:
            frames_to_cat = (extra_frame, frames)
        else:
            frames_to_cat = (frames, extra_frame)

        frames = torch.cat(frames_to_cat)
        frames[:, ::num_subframes] *= 0.5
        return frames

    def __init__(self, num_frames, adaptive=False):
        self._num_frames = num_frames
        self._adaptive = adaptive

    def __call__(self, images, flows=None):
        if flows is None:
            return images

        valid_flows = {
            backwards: flow
            for backwards, flow in zip((False, True, ), flows)
            if flow is not None
        }

        num_flows = len(valid_flows)
        if num_flows == 0:
            return samples

        n, c, h, w = images.shape
        max_num_subframes = int((self._num_frames - 1) / (n - 1))
        if self._adaptive:
            max_shift = max(flow.field().pixels().abs().max() for flow in valid_flows.values())
            num_subframes = int(max_shift / num_flows + 1)
            num_subframes = min(num_subframes, max_num_subframes)
        else:
            num_subframes = max_num_subframes

        interpolated_shape = ((n - 1) * num_subframes + 1, c, h, w, )
        interpolated_frames = torch.zeros(interpolated_shape, device=images.device)
        weighted = num_flows == 2
        for backwards, flow in valid_flows.items():
            interpolated_frames += type(self)._interpolate(
                images, flow, num_subframes, backwards=backwards, weighted=weighted)

        return interpolated_frames, None

class PrintStats:
    DEFAULT_MEAN_BLUR = '?'

    def __call__(self, images, flows=None):
        mean_irradiance = images.mean()
        if flows is None or (flows.forward is None and flows.backward is None):
            mean_blur = type(self).DEFAULT_MEAN_BLUR
        else:
            flow = flows.forward if flows.backward is None else flows.backward
            mean_blur = flow.pixels().magnitude().mean()

        print(f'Mean irradiance (W/m^2): {mean_irradiance}, Mean blur (pixel count per frame): {mean_blur}')
        return images, flows
