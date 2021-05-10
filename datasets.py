import numpy as np
import torchvision
import cv2
import natsort
import torch.distributed
try:
    import nvidia.dali.pipeline
    import nvidia.dali.ops
    import nvidia.dali.plugin.pytorch
    has_dali = True
except ModuleNotFoundError:
    has_dali = False

import glob
import functools

import transform

def make_example(frames, flows=None, transform=None, x_transform=None, y_transform=None):
    with torch.no_grad():
        if transform is not None:
            frames, flows = transform(frames, flows)

        gt = frames[len(frames) // 2].unsqueeze(0)

        if x_transform is not None:
            frames, flows = x_transform(frames, flows)

        if y_transform is not None:
            gt = y_transform(gt)[0].squeeze(0)

        return (frames, flows), gt

def collate(batch):
    x_batch, y_batch = zip(*batch)
    frame_batch, flow_batch = zip(*x_batch)
    frames = torch.stack(frame_batch)
    gt = torch.stack(y_batch)
    if flow_batch[0] is None:
        return (frames, None), gt

    forward_flows, backward_flows = zip(*flow_batch)
    forward = None if forward_flows[0] is None else torch.stack(forward_flows)
    backward = None if backward_flows[0] is None else torch.stack(backward_flows)
    flows = transform.BidirectionalFlow(forward, backward)
    return (frames, flows), gt

class FixedLengthDataset:
    def __init__(
        self,
        dataset,
        length
    ):
        super().__init__()
        self._dataset = dataset
        self._length = length

    def __getitem__(self, index):
        return self._dataset[index % len(self._dataset)]

    def __len__(self):
        return self._length

class FileDataset:
    def __init__(
        self,
        root,
        suffix,
        loader=None,
        num=None
    ):
        super().__init__()
        self._paths = natsort.natsorted(glob.glob(f'{root}/**/*.{suffix}', recursive=True))
        self._loader = (lambda x: x) if loader is None else loader
        if num is not None:
            self._paths = self._paths[:num]

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return [self._loader(path) for path in self._paths[subscript]]

        return self._loader(self._paths[subscript])

    def __len__(self):
        return len(self._paths)

class FrameDataset:
    DEFAULT_SUFFIX = 'jpg'
    DEFAULT_FLOW_SUFFIX = 'npy'

    @staticmethod
    def _load_raw(path):
        return rawpy.imread(path).raw_image

    def __len__(self):
        return self._num_examples

    def _open_frames(self, index):
        image_index = index * self._num_frames
        frames = self._images[image_index:image_index + self._num_frames]
        flows = self._flows[image_index:image_index + self._num_frames]
        return frames, flows

    def _get_frames(self, index):
        if index in self._cache:
            data = self._cache[index]
        else:
            data = self._open_frames(index)
            self._cache[index] = data

        return data

    def _fill_cache(self):
        for i in range(self._num_examples):
            self._open_frames(i)

    def _get_gt(self, frames):
        return frames[len(frames) // 2].unsqueeze(0)

    def __init__(
        self,
        root,
        num_frames,
        suffix=None,
        num_examples=None,
        grayscale=False,
        flow_root=None,
        flow_suffix=None,
        prefetch=False,
        transform=None,
        x_transform=None,
        y_transform=None
    ):
        self._num_frames = num_frames
        suffix = type(self).DEFAULT_SUFFIX if suffix is None else suffix
        num_images = None if num_examples is None else num_examples * num_frames
        if  suffix == 'raw':
            loader = type(self)._load_raw
        else:
            flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            loader = functools.partial(cv2.imread, flags=flags)

        self._images = FileDataset(root, suffix, loader, num=num_images)
        num_images = len(self._images)
        if num_images % num_frames != 0:
            raise ValueError(f'Invalid number of images in {root}!')

        self._num_examples = num_images // num_frames

        if flow_root is None:
            self._flows = []
        else:
            flow_suffix = type(self).DEFAULT_FLOW_SUFFIX if flow_suffix is None else flow_suffix
            self._flows = FileDataset(flow_root, flow_suffix, np.load, num=num_images)

        self._cache = {}
        if prefetch:
            self._fill_cache()

        self._transform = transform
        self._x_transform = x_transform
        self._y_transform = y_transform

    def __getitem__(self, index):
        frames, flows = self._get_frames(index)
        flows = None if len(flows) == 0 else flows
        return make_example(frames, flows=flows, transform=self._transform,
            x_transform=self._x_transform, y_transform=self._y_transform)

if has_dali:
    class VideoPipeline(nvidia.dali.pipeline.Pipeline):
        DEFAULT_BUFFER_SIZE = 4
        DEFAULT_QUALITY = 0
        DEFAULT_NUM_SHARDS = 1
        DEFAULT_SHARD_ID = 0
        DEFAULT_SEED = -1
        OPTICAL_FLOW_OUTPUT_FORMAT = 4
        READER_NAME = 'Reader'

        def __init__(
            self,
            batch_size,
            num_threads,
            device_id,
            num_frames,
            filenames,
            buffer_size=DEFAULT_BUFFER_SIZE,
            num_shards=DEFAULT_NUM_SHARDS,
            shard_id=DEFAULT_SHARD_ID,
            shuffle=False,
            read_ahead=False,
            effective_num_frames=None,
            size=None,
            optical_flow=False,
            quality=DEFAULT_QUALITY,
            seed=DEFAULT_SEED
        ):
            super().__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                seed=seed)

            self._input = nvidia.dali.ops.VideoReader(
                device='gpu',
                sequence_length=num_frames,
                filenames=filenames,
                initial_fill=buffer_size,
                num_shards=num_shards,
                shard_id=shard_id,
                random_shuffle=shuffle,
                read_ahead=read_ahead
            )

            self._should_slice = (effective_num_frames is not None) and (effective_num_frames < num_frames)
            if self._should_slice:
                self._effective_num_frames = nvidia.dali.ops.Constant(fdata=effective_num_frames)
                self._anchor = nvidia.dali.ops.Constant(fdata=0)
                #self._rng_anchor = nvidia.dali.ops.Uniform(range=(0, 1))
                self._slice = nvidia.dali.ops.Slice(device='gpu', axes=0, normalized_shape=False,
                    out_of_bounds_policy='pad')

            self._should_crop = size is not None
            if self._should_crop:
                self._rng_x = nvidia.dali.ops.Uniform(range=(0, 1))
                self._rng_y = nvidia.dali.ops.Uniform(range=(0, 1))
                self._crop = nvidia.dali.ops.Crop(device='gpu', crop=size)

            self._should_calculate_flow = optical_flow
            if self._should_calculate_flow:
                self._optical_flow = nvidia.dali.ops.OpticalFlow(
                    device='gpu',
                    enable_temporal_hints=True,
                    output_format=type(self).OPTICAL_FLOW_OUTPUT_FORMAT,
                    preset=quality
                )

            self._transpose = nvidia.dali.ops.Transpose(device='gpu', perm=(0, 3, 1, 2, ))

        def define_graph(self):
            video = self._input(name=type(self).READER_NAME)
            if self._should_slice:
                video = self._slice(video, self._anchor(), self._effective_num_frames())
                #video = self._slice(video, self._rng_anchor(), self._effective_num_frames())
            if self._should_crop:
                video = self._crop(video, crop_pos_x=self._rng_x(), crop_pos_y=self._rng_y())

            if self._should_calculate_flow:
                flow = self._optical_flow(video)
                return self._transpose(video), self._transpose(flow)

            return self._transpose(video)

    class VideoDataset:
        DEFAULT_SUFFIX = 'mp4'

        def __init__(
            self,
            batch_size,
            num_workers,
            device_id,
            num_frames,
            root,
            suffix=None,
            num_examples=None,
            distributed=False,
            buffer_size=VideoPipeline.DEFAULT_BUFFER_SIZE,
            shuffle=False,
            prefetch=False,
            effective_num_frames=None,
            size=None,
            optical_flow=False,
            quality=VideoPipeline.DEFAULT_QUALITY,
            seed=VideoPipeline.DEFAULT_SEED,
            transform=None,
            x_transform=None,
            y_transform=None
        ):
            suffix = type(self).DEFAULT_SUFFIX if suffix is None else suffix
            filenames = [filename for filename in FileDataset(root, suffix, num=num_examples)]
            if distributed:
                num_shards = torch.distributed.get_world_size()
                shard_id = torch.distributed.get_rank()
            else:
                num_shards = VideoPipeline.DEFAULT_NUM_SHARDS
                shard_id = VideoPipeline.DEFAULT_SHARD_ID

            self._optical_flow = optical_flow
            pipe = VideoPipeline(
                batch_size,
                num_workers,
                device_id,
                num_frames,
                filenames,
                buffer_size=buffer_size,
                num_shards=num_shards,
                shard_id=shard_id,
                shuffle=shuffle,
                read_ahead=prefetch,
                effective_num_frames=effective_num_frames,
                size=size,
                optical_flow=optical_flow,
                quality=quality,
                seed=seed
            )

            pipe.build()
            names = ('video', )
            if optical_flow:
                names += ('flow', )

            self._epoch_started = False
            self._iterator = nvidia.dali.plugin.pytorch.DALIGenericIterator(
                pipe,
                names,
                pipe.epoch_size(type(pipe).READER_NAME) / num_shards
            )

            self._transform = transform
            self._x_transform = x_transform
            self._y_transform = y_transform

        def __iter__(self):
            if self._epoch_started:
                self._iterator.reset()
            else:
                self._epoch_started = True

            return self

        def __next__(self):
            with torch.no_grad():
                data = next(self._iterator)[0]
                videos = data['video']
                if self._optical_flow:
                    flows = data['flow']
                    missing_flow = torch.zeros_like(flows[:, 0]).unsqueeze(1)
                    flows = torch.cat((flows, missing_flow, ), dim=1)
                else:
                    flows = (None, ) * videos.size(0)

                batch = []
                for video, flow in zip(videos, flows):
                    batch.append(make_example(video, flows=flow, transform=self._transform,
                        x_transform=self._x_transform, y_transform=self._y_transform))

                return collate(batch)

else:
    class VideoDataset:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError('Not supported without DALI installed')
