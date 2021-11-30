import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as transF

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

class BCEDiceLoss(nn.Module):
  def __init__(self):
    super(BCEDiceLoss, self).__init__()

  def _crop(self,img, input_shape):
    h, w = input_shape[0], input_shape[1]

    dh = int((768 - h)/2)
    dw = int((1024 - w)/2)
    cropped = transF.crop(img, dw, dh, h, w)

    return cropped
    

  def forward(self, x, target, input_shape):
    x = self._crop(x, input_shape)
    target = self._crop(target, input_shape)
    bce = F.binary_cross_entropy_with_logits(x, target)
    smooth = 1e-5
    x = torch.sigmoid(x)
    batch_num = x.size(0)
    x = x.view(batch_num, -1)
    target = target.view(batch_num, -1)
    intersection = (x * target)
    # channel에 대해서 더해주고
    dice = (intersection.sum(1) * 2. + smooth) / (x.sum(1) + target.sum(1) + smooth)
    dice = 1-dice.sum() / batch_num

    return 0.5*bce + dice

class LovaszHingeLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input, target):
    input = input.squeeze(1)
    target = target.squeeze(1)
    loss = lovasz_hinge(input, target, per_image=True)

    return loss