import torch
import kornia

class DarkChannelLoss(torch.nn.Module):
    def _get_dark_channel(image):
        patches = torch.nn.functional.unfold(image, self._patch_size, stride=self._stride)
        return patches.min(dim=1)

    def __init__(self, patch_size, stride=1):
        super().__init__()
        self._loss = torch.nn.MSELoss()
        self._patch_size = patch_Size
        self._stride = stride

    def forward(self, pred, gt):
        pred_dark_channel = self._get_dark_channel(pred)
        gt_dark_channel = self._get_dark_channel(gt)
        return self._loss(pred_dark_channel, gt_dark_channel)

class GradientLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.L1Loss()

    def forward(self, pred, gt):
        pred_gradient = kornia.spatial_gradient(pred)
        gt_gradient = kornia.spatial_gradient(gt)
        return self._loss(pred_gradient, gt_gradient)

class BasicLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self._main_loss = torch.nn.L1Loss()
        self._gradient_loss = GradientLoss()

    def forward(self, pred, gt):
        if pred.ndim > gt.ndim:
            n, d, c, h, w = pred.shape
            gt = gt.unsqueeze(1).expand_as(pred)
            pred = pred.reshape(n * d, c, h, w)
            gt = gt.reshape(n * d, c, h, w)

        return self._main_loss(pred, gt) + self._gradient_loss(pred, gt)

class AnnealLoss(torch.nn.Module):
    DEFAULT_ALPHA = 0.9998
    DEFAULT_BETA = 100

    def __init__(
        self,
        alpha=DEFAULT_ALPHA,
        beta=DEFAULT_BETA
    ):
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._loss = BasicLoss()

    def forward(self, t, pred, gt):
        return self._beta * self._alpha ** t * self._loss(pred, gt)

class TotalLoss(torch.nn.Module):
    DEFAULT_COEFF_BASIC = 1
    DEFAULT_COEFF_ANNEAL = 1

    def __init__(
        self,
        coeff_basic=DEFAULT_COEFF_BASIC,
        coeff_anneal=DEFAULT_COEFF_ANNEAL,
        alpha=AnnealLoss.DEFAULT_ALPHA,
        beta=AnnealLoss.DEFAULT_BETA
    ):
        super().__init__()
        self._coeff_basic = coeff_basic
        self._coeff_anneal = coeff_anneal
        self._loss_basic = BasicLoss()
        self._loss_anneal = AnnealLoss(alpha=alpha, beta=beta)

    def forward(self, t, pred1, pred, gt):
        basic_loss = self._coeff_basic * self._loss_basic(pred, gt)
        anneal_loss = self._coeff_anneal * self._loss_anneal(t, pred1, gt)
        return basic_loss, anneal_loss
