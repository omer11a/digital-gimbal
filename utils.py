import numpy as np
import torch
import tensorboardX
import torchvision

import itertools
import collections
import os
import datetime
import shutil
import time

def forward_in_patches(model, x, patch_size, stride=1, *args, **kwargs):
    original_shape = x.shape
    x = x.reshape(-1, *original_shape[-3:])
    ones = torch.ones(1, *original_shape[-3:], device=x.device)
    unfolded_ones = torch.nn.functional.unfold(ones, patch_size, stride=stride)
    _, patch_dim, num_patches = unfolded_ones.shape
    normalizer = torch.nn.functional.fold(
        unfolded_ones, original_shape[-2:], patch_size, stride=stride)
    normalizer[normalizer < 1] = 1

    patches = torch.nn.functional.unfold(x, patch_size, stride=stride)
    results_per_patch = []
    for i in range(num_patches):
        patch = patches[:, :, i].reshape(*original_shape[:-2], *patch_size)
        results_per_patch.append(model(patch, *args, **kwargs))

    results = []
    for i in range(len(results_per_patch[0])):
        result_patches = [
            patch_results[i].reshape(-1, patch_dim)
            for patch_results in results_per_patch
        ]

        result_patches = torch.stack(result_patches, dim=-1)
        result = torch.nn.functional.fold(
            result_patches, original_shape[-2:], patch_size, stride=stride)
        result = torch.reshape(result / normalizer, (original_shape[0], -1, *original_shape[-3:]))
        results.append(result.squeeze(1))

    return results

def adaptive_conv(images, kernels, padding=None):
    h, w = images.shape[-2:]
    kernel_size = int(np.sqrt(kernels.size(-3)))
    kernel_radius = kernel_size // 2
    padding = 'constant' if padding is None else padding
    images = torch.nn.functional.pad(
        images,
        (kernel_radius, kernel_radius, kernel_radius, kernel_radius),
        mode=padding
    )

    slices = [
        images[..., i:i + h, j:j + w]
        for i, j in itertools.product(range(kernel_size), range(kernel_size))
    ]

    stacked_slices = torch.stack(slices, dim=-3)
    return stacked_slices.mul(kernels).sum(dim=-3)

class CNN(torch.nn.Module):
    @staticmethod
    def _build_conv_layers(in_channel_num, out_channel_num, kernel_size, padding):
        return [
            torch.nn.Conv2d(in_channel_num, out_channel_num, kernel_size, padding=padding),
            torch.nn.ReLU(),
        ]

    def __init__(self, channel_nums, kernel_size, end_with_relu=True):
        super().__init__()

        padding = (kernel_size  - 1) // 2
        layers = [
            type(self)._build_conv_layers(in_channel_num, out_channel_num, kernel_size, padding)
            for in_channel_num, out_channel_num in zip(channel_nums[:-1], channel_nums[1:])
        ]

        if not end_with_relu:
            layers[-1] = layers[-1][:1]

        self._model = torch.nn.Sequential(*sum(layers, []))

    def forward(self, x):
        return self._model(x)

class RepeatedConv(CNN):
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_NUM_REPEATS = 3

    def __init__(
        self,
        in_channel_num,
        out_channel_num,
        kernel_size=DEFAULT_KERNEL_SIZE,
        num_repeats=DEFAULT_NUM_REPEATS
    ):
        channel_nums  = (in_channel_num, ) + (out_channel_num, ) * num_repeats
        super().__init__(channel_nums, kernel_size)

class MovingAverage(object):
    def __init__(self, max_length):
        self._cache = collections.deque(maxlen=max_length)

    def update(self, value):
        self._cache.append(value)

    def get(self):
        return sum(self._cache) / len(self._cache)

class Model(object):
    DEFAULT_BACKEND = 'nccl'

    @staticmethod
    def detach(tensor, pil=False):
        tensor = tensor.detach().cpu()
        if pil:
            return torchvision.transforms.functional.to_pil_image(tensor)

        return tensor.numpy()

    def _set_device(self, backend=None):
        self._world_size = torch.cuda.device_count()
        backend = type(self).DEFAULT_BACKEND if backend is None else backend
        torch.distributed.init_process_group(
            backend,
            world_size=self._world_size,
            rank=self._gpu
        )

        torch.cuda.set_device(self._gpu)

    def _prepare_checkpoint_dir(self):
        train_config = self._config['train']
        self._checkpoint_dir = train_config['checkpoint_dir']
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        self._max_num_checkpoints = train_config.getint('max_num_checkpoints')

    def _prepare_output_dir(self):
        self._output_dir = self._config['eval']['output_dir']
        os.makedirs(self._output_dir, exist_ok=True)

    def _prepare_log_writer(self):
        if self._gpu != 0 or not self._should_log:
            return

        main_dir = self._config['train']['log_dir']
        sub_dir = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(main_dir, sub_dir)
        os.makedirs(log_dir, exist_ok=True)
        self._log_writer = tensorboardX.SummaryWriter(log_dir)

    def _get_checkpoint_filename(self, is_best=False):
        basename = 'best' if is_best else f'{self._t:06d}'
        return os.path.join(self._checkpoint_dir, f'{basename}.pth.tar')

    def _load_checkpoint(self):
        filename = self._get_checkpoint_filename(is_best=True)
        checkpoint = torch.load(filename, map_location='cpu')
        self._epoch = checkpoint['epoch']
        self._t = checkpoint['t']
        self._best_loss = checkpoint['best_loss']
        self._model.load_state_dict(checkpoint['state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])

    def _remove_extra_checkpoints(self):
        filenames = sorted(os.listdir(self._checkpoint_dir))
        num_files_to_remove = max(0, len(filenames) - self._max_num_checkpoints)
        for filename in filenames[:num_files_to_remove]:
            os.remove(os.path.join(self._checkpoint_dir, filename))

    def _save_checkpoint(self):
        if self._gpu != 0:
            return

        state = {
            'epoch': self._epoch,
            't': self._t,
            'best_loss': self._best_loss,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

        filename = self._get_checkpoint_filename()
        torch.save(state, filename)

        current_average_loss = self._average_loss.get()
        if current_average_loss < self._best_loss:
            self._best_loss = current_average_loss
            best_filename = self._get_checkpoint_filename(is_best=True)
            shutil.copyfile(filename, best_filename)

        self._remove_extra_checkpoints()

    def _write_log(self, scalars, scalar_dicts, images, image_lists):
        if self._gpu != 0 or not self._should_log:
            return

        for name, scalar in scalars.items():
            self._log_writer.add_scalar(name, scalar, global_step=self._t)

        for name, scalar_dict in scalar_dicts.items():
            self._log_writer.add_scalars(name, scalar_dict, global_step=self._t)

        for name, image in images.items():
            self._log_writer.add_image(name, image, global_step=self._t)

        for name, image_list in image_lists.items():
            self._log_writer.add_images(name, image_list, global_step=self._t)

    def _print(self, to_print):
        if self._gpu == 0:
            print(to_print)

    def _print_iteration(self, step, to_print, elapsed_time):
        prefix = (
            f'{self._t:4d}',
            f'epoch {self._epoch:2d}',
            f'step {step:4d}',
        )

        postfix = (
            f'time:{elapsed_time:.2f} seconds.',
        )

        self._print('\t| '.join(prefix + to_print + postfix))

    def _prepare_data(self, num_workers):
        raise NotImplementedError()

    def _prepare_model(self):
        raise NotImplementedError()

    def _prepare_loss(self):
        raise NotImplementedError()

    def _prepare_optimizer(self):
        raise NotImplementedError()

    def _convert_to_cuda(self, data):
        raise NotImplementedError()

    def _train_step(self, x, y):
        raise NotImplementedError()

    def _eval_step(self, i, x, y):
        raise NotImplementedError()

    def _train(self):
        reached_max_steps = False
        start_epoch = self._epoch
        for self._epoch in range(start_epoch, self._num_epochs):
            epoch_start_time = time.time()
            for step, data in enumerate(self._data_loader):
                step_start_time = time.time()
                data = self._convert_to_cuda(data)
                loss, scalars, scalar_dicts, images, image_lists, to_print = self._train_step(*data)
                elapsed_time = time.time() - step_start_time
                self._average_loss.update(loss)
                self._write_log(scalars, scalar_dicts, images, image_lists)
                self._print_iteration(step, to_print, elapsed_time)

                self._t += 1
                if self._t % self._save_freq == 0:
                    self._save_checkpoint()

                if self._num_steps is not None and self._t == self._num_steps:
                    self._print(f'Reached maximum number of steps.')
                    reached_max_steps = True
                    break

            elapsed_time = time.time() - epoch_start_time
            self._print(f'Epoch {self._epoch} is finished, time elapsed {elapsed_time:.2f} seconds.')
            if reached_max_steps:
                break

        self._save_checkpoint()

    def _metrics_to_str(self, metrics):
        raise NotImplementedError()

    def _eval(self):
        total_metrics = 0
        for i, data in enumerate(self._data_loader):
            data = self._convert_to_cuda(data)

            metrics, images = self._eval_step(i, *data)
            total_metrics += metrics
            str_metrics = self._metrics_to_str(metrics)
            self._print(f'For {i}-th example got ' + ', '.join(str_metrics))
            for name, image in images.items():
                name = '.'.join((name, self._config['eval']['output_suffix'], ))
                path = os.path.join(self._config['eval']['output_dir'], name)
                image.save(path)

        average_metrics = total_metrics / (i + 1)
        str_metrics = self._metrics_to_str(average_metrics)
        self._print('Average metrics: ' + ', '.join(str_metrics))

    def __init__(
        self,
        config,
        gpu,
        eval_mode=False,
        backend=None,
        num_workers=0,
        restart=False,
        should_log=False,
        verbose=False
    ):
        self._config = config
        self._gpu = gpu
        self._eval_mode = eval_mode
        self._should_log = should_log
        self._verbose = verbose
        self._set_device(backend=backend)
        self._data_loader = self._prepare_data(num_workers)
        self._model = self._prepare_model()
        self._loss = self._prepare_loss()
        self._optimizer = self._prepare_optimizer()

        self._prepare_checkpoint_dir()
        if self._eval_mode:
            self._prepare_output_dir()
            should_load = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self._print('=> eval')
        else:
            self._prepare_log_writer()
            train_config = config['train']
            self._num_epochs = train_config.getint('num_epochs')
            self._save_freq = train_config.getint('save_freq')
            self._average_loss = MovingAverage(train_config.getint('save_freq'))
            if train_config.getboolean('force_num_steps'):
                self._num_steps = self._num_epochs * len(self._data_loader)
            else:
                self._num_steps = None

            should_load = not restart
            self._print('=> train')

        if should_load:
            checkpoint = self._load_checkpoint()
            self._print(f'=> loaded checkpoint (epoch {self._epoch}, t {self._t})')
        else:
            self._epoch = 0
            self._t = 0
            self._best_loss = np.inf

    def run(self):
        if self._eval_mode:
            self._eval()
        else:
            self._train()
