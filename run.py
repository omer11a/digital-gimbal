import torch
import kornia

import configparser
import argparse
import os

import utils
import transform
import datasets
import bpn
import losses

DEFAULT_ADDR = '127.0.0.1'
DEFAULT_PORT = 8888

class Model(utils.Model):
    MAX_VALUE = 1
    SSIM_WINDOW_SIZE = 11

    @classmethod
    def _calculate_metrics(cls, pred, gt):
        psnr = kornia.psnr_loss(pred, gt, cls.MAX_VALUE)
        ssim = kornia.ssim(pred, gt, cls.SSIM_WINDOW_SIZE, reduction='mean')
        return psnr, ssim

    @classmethod
    def _get_image_item(cls, i, name, image, y=None, j=None):
        psnr_text = ''
        if y is not None:
            with torch.no_grad():
                psnr = kornia.psnr_loss(image, y, cls.MAX_VALUE)
                psnr_text = f'_{psnr:.2f}dB'

        j_text = '' if j is None else str(j)
        title = f'{i}_{name}{j_text}{psnr_text}'
        return title, cls.detach(image, pil=True)

    @classmethod
    def _get_burst_items(cls, i, name, burst, y=None):
        return [
            cls._get_image_item(i, name, image[0], y=y, j=j)
            for j, image in enumerate(burst.split(1))
        ]

    def _prepare_data(self, num_workers):
        specific_config = self._config['eval'] if self._eval_mode else self._config['train']
        dali = specific_config.getboolean('dali')
        flow_format = specific_config.get('flow_format', None)
        grayscale = specific_config.getboolean('grayscale')
        generated_flow_format = 'f' if (self._verbose and flow_format is None) else flow_format
        transforms = (transform.ToTensor(
            num_bits=specific_config.getint('num_bits'),
            generate_flow=self._verbose,
            flow_format=generated_flow_format
        ), )

        x_transforms = tuple()
        y_transforms = tuple()
        if dali and grayscale:
            transforms += (transform.RGBToGrayscale(), )

        editing_crop_size = (
            specific_config.getint('editing_crop_height'),
            specific_config.getint('editing_crop_width'),
        )

        if not self._eval_mode or specific_config.getboolean('fake'):
            size = (
                specific_config.getint('height'),
                specific_config.getint('width'),
            )

            crop_size = (
                specific_config.getint('crop_height'),
                specific_config.getint('crop_width'),
            )

            gt_size = (
                specific_config.getint('gt_height'),
                specific_config.getint('gt_width'),
            )

            f = (
                specific_config.getfloat('focal_length_x'),
                specific_config.getfloat('focal_length_y'),
            )

            c = (
                specific_config.getfloat('principal_point_x'),
                specific_config.getfloat('principal_point_y'),
            )

            calibrated_size = (
                specific_config.getint('calibrated_height'),
                specific_config.getint('calibrated_width'),
            )

            #k = (
            #    specific_config.getfloat('k1'),
            #    specific_config.getfloat('k2'),
            #    specific_config.getfloat('k3'),
            #)

            roll = specific_config.getfloat('roll')
            pitch = specific_config.getfloat('pitch')
            yaw = specific_config.getfloat('yaw')
            depth = specific_config.getfloat('depth')
            transforms += (
                transform.RandomCrop(editing_crop_size),
                transform.Replicate(specific_config.getint('num_replicas')),
                transform.RandomTranslate(specific_config.getfloat('translation'), independent=True,
                    accumulative=True),
                transform.RandomRotate(f, c, calibrated_size, roll, pitch=pitch, yaw=yaw,
                    independent=True, accumulative=True, centeralized=True),
                #transform.Distort(f, c, calibrated_size, k),
                transform.RandomCrop(crop_size),
            )

            if not self._eval_mode:
                transforms += (
                    transform.RandomFlip(),
                    transform.RandomFlip(vertical=True),
                )

            x_transforms += (
                transform.Resize(size),
                #transform.Blur(f, c, calibrated_size, specific_config.getint('blur_kernel_size')),
                transform.InvertGammaCorrection(),
                transform.MultiplyColor(
                    specific_config.getfloat('min_white_level'),
                    specific_config.getfloat('max_white_level')
                ),
            )

            if self._verbose:
                x_transforms += (
                    transform.PrintStats(),
                )

            y_transforms += (
                transform.Resize(gt_size),
            )

        batch_size = 1 if self._eval_mode else specific_config.getint('batch_size')
        num_frames = specific_config.getint('num_frames')
        root = specific_config['data_dir']
        suffix = specific_config['suffix']
        num_examples = specific_config.getint('num_examples')
        shuffle = not self._eval_mode
        prefetch = specific_config.getboolean('prefetch')
        all_transform = transform.Compose(transforms)
        x_transform = transform.Compose(x_transforms)
        y_transform = transform.Compose(y_transforms)
        seed = specific_config.getint('seed')
        torch.manual_seed(seed)
        if dali:
            optical_flow = (flow_format is not None) and ('f' in flow_format)
            return datasets.VideoDataset(
                batch_size,
                num_workers,
                self._gpu,
                num_frames,
                root,
                suffix=suffix,
                num_examples=num_examples,
                distributed=True,
                buffer_size=specific_config.getint('buffer_size'),
                shuffle=shuffle,
                prefetch=prefetch,
                effective_num_frames=specific_config.getint('effective_num_frames'),
                size=editing_crop_size,
                optical_flow=optical_flow,
                quality=specific_config.getfloat('optical_flow_quality'),
                seed=seed,
                transform=all_transform,
                x_transform=x_transform,
                y_transform=y_transform
            )

        else:
            flow_root = root if flow_format is not None else None
            dataset = datasets.FrameDataset(
                root,
                num_frames,
                suffix=suffix,
                num_examples=num_examples,
                grayscale=grayscale,
                flow_root=flow_root,
                prefetch=prefetch,
                transform=all_transform,
                x_transform=x_transform,
                y_transform=y_transform
            )

            if 'num_iterations' in specific_config:
                dataset = datasets.FixedLengthDataset(
                    dataset,
                    length=specific_config.getint('num_iterations')
                )

            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._gpu
            )

            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=datasets.collate,
                pin_memory=True,
                sampler=sampler
            )

    def _prepare_model(self):
        model_config = self._config['model']

        model = bpn.BPN(
            model_config.getint('burst_length'),
            model_config.getint('channel_num'),
            model_config.getfloat('max_total_time'),
            model_config.getint('kernel_size'),
            uniform=model_config.getboolean('uniform'),
            fix_total_time=model_config.getboolean('fix_total_time'),
            min_time=model_config.getfloat('min_time'),
            delay=model_config.getfloat('delay'),
            pixel_size=model_config.getfloat('pixel_size'),
            quantum_efficiency=model_config.getfloat('quantum_efficiency'),
            wavelength=model_config.getfloat('wavelength'),
            dark_current=model_config.getfloat('dark_current'),
            shot_n=model_config.getint('shot_n'),
            shot_threshold=model_config.getfloat('shot_threshold'),
            read_std=model_config.getfloat('read_std'),
            fwc=model_config.getfloat('fwc'),
            response_tail=model_config.getfloat('response_tail'),
            gain=model_config.getfloat('gain'),
            prnu=model_config.getfloat('prnu'),
            num_bits=model_config.getint('num_bits'),
            normalize=model_config.getboolean('normalize'),
            reference_index=model_config.getint('reference_index', fallback=None),
            registration=model_config.getboolean('registration'),
            use_unregistered_burst=model_config.getboolean('use_unregistered_burst'),
            blind=model_config.getboolean('blind'),
            scale_factor=model_config.getint('scale_factor')
        )

        model = model.eval() if self._eval_mode else model.train()
        model = model.cuda(self._gpu)
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=(self._gpu, ))

    def _prepare_loss(self):
        train_config = self._config['train']
        gradient_p = train_config.getint('gradient_p')
        if self._eval_mode:
            loss = losses.BasicLoss()
        else:
            loss = losses.TotalLoss(
                coeff_basic=train_config.getfloat('coeff_basic'),
                coeff_anneal=train_config.getfloat('coeff_anneal'),
                alpha=train_config.getfloat('alpha'),
                beta=train_config.getfloat('beta')
            )

        return loss.cuda()

    def _prepare_optimizer(self):
        config = self._config['train']
        lr = config.getfloat('lr')
        fm_lr = config.getfloat('fm_lr')
        return torch.optim.Adam([
            {'params': self._model.module.get_reconstruction_parameters()},
            {'params': self._model.module.get_forward_model_parameters(), 'lr': fm_lr},
            ], lr=lr
        )

    def _convert_to_cuda(self, data):
        with torch.no_grad():
            x, y = data
            samples, _ = x
            return samples.cuda(), y.cuda()

    def _train_step(self, x, y):
        effective_t = self._t * x.size(0) * self._world_size
        (burst, max_exposure_image, normalized_burst, registered_burst, convolved_burst1, pred1,
            convolved_burst, pred) = self._model(x, t=effective_t)

        convolved_burst = transform.apply_gamma_correction(convolved_burst)
        pred1 = transform.apply_gamma_correction(pred1)
        pred = transform.apply_gamma_correction(pred)
        anneal_in = convolved_burst if self._config['model']['uniform'] else pred1
        basic_loss, anneal_loss = self._loss(effective_t, anneal_in, pred, y)
        total_loss = basic_loss + anneal_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        with torch.no_grad():
            burst_sum = transform.apply_gamma_correction(burst.sum(dim=1))
            burst = transform.apply_gamma_correction(burst)
            max_exposure_image = transform.apply_gamma_correction(max_exposure_image)
            normalized_burst = transform.apply_gamma_correction(normalized_burst)
            registered_burst = transform.apply_gamma_correction(registered_burst)
            convolved_burst1 = transform.apply_gamma_correction(convolved_burst1)

            psnr, ssim = type(self)._calculate_metrics(pred, y)
            time_division = {
                str(i) : precentage.item()
                for i, precentage in enumerate(self._model.module.get_time_division())
            }

        scalars = {
            'basic_loss' : basic_loss.item(),
            'anneal_loss' : anneal_loss.item(),
            'total_loss' : total_loss.item(),
            'psnr' : psnr.item(),
            'ssim' : ssim.item(),
        }

        scalar_dicts = {
            'time_division' : time_division,
        }

        images = {
            'gt' : type(self).detach(y[0]),
            'max_exposure_image' : type(self).detach(max_exposure_image[0]),
            'burst_sum' : type(self).detach(burst_sum[0]),
            'pred1' : type(self).detach(pred1[0]),
            'pred' : type(self).detach(pred[0]),
        }

        image_lists = {
            'burst' : type(self).detach(burst[0]),
            'normalized_burst' : type(self).detach(normalized_burst[0]),
            'registered_burst' : type(self).detach(registered_burst[0]),
            'convolved_burst1' : type(self).detach(convolved_burst1[0]),
            'convolved_burst' : type(self).detach(convolved_burst[0]),
        }

        to_print = (
            f'basic_loss: {basic_loss:.4f}',
            f'anneal_loss: {anneal_loss:.4f}',
            f'total_loss: {total_loss:.4f}',
            f'PSNR: {psnr:.2f}dB',
            f'SSIM: {ssim:.4f}',
        )

        return total_loss.item(), scalars, scalar_dicts, images, image_lists, to_print

    def _eval_step(self, i, x, y):
        config = self._config['eval']
        force_fm = config.getboolean('fake')
        snr = config.getfloat('snr')
        with torch.no_grad():
            if config.getboolean('divide_to_patches'):
                patch_stride = config.getint('patch_stride')
                patch_size = (
                    config.getint('patch_height'),
                    config.getint('patch_width'),
                )

                results = utils.forward_in_patches(
                    self._model, x, patch_size, stride=patch_stride, force_fm=force_fm, snr=snr)

            else:
                results = self._model(x, force_fm=force_fm, snr=snr)

            (burst, max_exposure_image, normalized_burst, registered_burst, convolved_burst1, pred1,
                convolved_burst, pred) = results

            burst_sum = transform.apply_gamma_correction(burst.sum(dim=1))
            burst = transform.apply_gamma_correction(burst)
            max_exposure_image = transform.apply_gamma_correction(max_exposure_image)
            normalized_burst = transform.apply_gamma_correction(normalized_burst)
            registered_burst = transform.apply_gamma_correction(registered_burst)
            convolved_burst1 = transform.apply_gamma_correction(convolved_burst1)
            pred1 = transform.apply_gamma_correction(pred1)
            convolved_burst = transform.apply_gamma_correction(convolved_burst)
            pred = transform.apply_gamma_correction(pred)
            loss = self._loss(pred, y)

        images = [
            type(self)._get_image_item(i, 'gt', y[0]),
            type(self)._get_image_item(i, 'sum', burst_sum[0], y=y[0]),
            type(self)._get_image_item(i, 'max', max_exposure_image[0], y=y[0]),
            type(self)._get_image_item(i, 'pred1', pred1[0], y=y[0]),
            type(self)._get_image_item(i, 'pred', pred[0], y=y[0]),
        ]

        images += type(self)._get_burst_items(i, 'burst', burst[0], y=y[0])
        images += type(self)._get_burst_items(i, 'normalized', normalized_burst[0], y=y[0])
        images += type(self)._get_burst_items(i, 'registered', registered_burst[0], y=y[0])
        images += type(self)._get_burst_items(i, 'convolved1', convolved_burst1[0], y=y[0])
        images += type(self)._get_burst_items(i, 'convolved', convolved_burst[0], y=y[0])

        with torch.no_grad():
            pred_psnr, pred_ssim = type(self)._calculate_metrics(pred, y)
            metrics = torch.Tensor([loss, pred_psnr, pred_ssim])

        return metrics, dict(images)

    def _metrics_to_str(self, metrics):
        loss, psnr, ssim = metrics
        return (f'loss: {loss}', f'PSNR: {psnr:.2f}dB', f'SSIM {ssim:.4f}')

    def _print_exposure_times(self):
        if self._gpu != 0:
            return

        exposure_times = [time.item() for time in self._model.module.get_exposure_times()]
        print('Exposure times (sec): ' + ', '.join(str(time) for time in exposure_times))

    def _train(self):
        super()._train()
        self._print_exposure_times()

    def _eval(self):
        self._print_exposure_times()
        super()._eval()

def run(gpu, args):
    config = configparser.ConfigParser()
    config.read(args.config)

    model = Model(
        config,
        gpu,
        eval_mode=args.eval,
        backend=args.backend,
        num_workers=args.num_workers,
        restart=args.restart,
        should_log=args.log,
        verbose=args.verbose
    ).run()

def main():
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--config', '-c', default='config.ini', help='path to configuration file')
    parser.add_argument('--eval', '-e', action='store_true', help='whether to evaluate instead of train')
    parser.add_argument('--backend', '-b', help='the backend to for distributed learning')
    parser.add_argument('--num_workers', '-nw', default=0, type=int, help='number of workers in data loader')
    parser.add_argument('--restart', '-r', action='store_true', help='whether to restart the train process')
    parser.add_argument('--log', '-l', action='store_true', help='whether to write tensorboard logs')
    parser.add_argument('--port', '-p', default=DEFAULT_PORT, type=int, help='port for synchronizing processes')
    parser.add_argument('--verbose', '-v', action='store_true', help='whether to print extra information')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = DEFAULT_ADDR
    os.environ['MASTER_PORT'] = str(args.port)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    torch.multiprocessing.spawn(
        run,
        nprocs=torch.cuda.device_count(),
        args=(args, )
    )

if __name__ == '__main__':
    main()
