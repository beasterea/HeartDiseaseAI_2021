import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as transF
import numpy as np

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

class BCEDiceLoss(nn.Module):
  def __init__(self, crop = False, crop_batch = False):
    super(BCEDiceLoss, self).__init__()
    self.crop = crop
    self.crop_batch = crop_batch
  
  def _crop_single(self, img, input_shape):
    h, w = input_shape[0], input_shape[1]
    dh = int((768-h)/2)
    dw = int((1024-w)/2)
    cropped = transF.crop(img, dw, dh, h, w)
    return cropped

  def _crop_batch(self, img, input_shape):
    n = img.size(0) # batch size
    h,w = input_shape[0][0], input_shape[1][0]
    dh = int((768-h)/2)
    dw = int((1024-w)/2)
    cropped = transF.crop(img[0], dw, dh, h, w)

    for i in range(1,n):
      h,w = input_shape[0][i], input_shape[1][i]

      dh = int((768-h)/2)
      dw = int((1024-w)/2)
      cropped = torch.cat((cropped, transF.crop(img[i], dw, dh, h, w)), dim = 1)
        
      return cropped

  def forward(self, x, target, input_shape):
    """
    bce = 0
    for i in range(len(x)):
      x_ = self._crop(x, input_shape)
      target_ = self._crop(target, input_shape)
      bce += F.binary_cross_entropy_with_logits(x_, target_)
    bce /= len(x)
    """
    n = x.size(0)
    loss = 0
    if self.crop:
      if self.crop_batch:
        x = self._crop_batch(x, input_shape)
        target = self._crop_batch(target, input_shape)
      else:
        bce = 0
        intersection, sum_x, sum_target = 0,0,0
        smooth = 1e-5
        for i in range(n):
          x_, target_ = self._crop_single(x[i], (input_shape[0][i], input_shape[1][i])),self._crop_single(target[i], (input_shape[0][i], input_shape[1][i]))
          #print(f"shape of x : {x_.shape} shape of target : {target_.shape}")
          bce += F.binary_cross_entropy_with_logits(x_, target_)
          #print(f"bce : {bce}")
          x_ = x_.reshape(1, -1)
          target_ = target_.reshape(1, -1)
          intersection += (x_ * target_).sum(1)
          sum_x += x_.sum(1)
          sum_target += target_.sum(1)
        dice = (intersection * 2 + smooth) / (sum_x + sum_target + smooth)
        #print(f"dice : {dice}")
        dice = (1-dice)/n
        loss = 0.5*bce + dice
        return loss


    bce = F.binary_cross_entropy_with_logits(x, target) # 알아서 sigmoid를 붙여준 값을 바탕으로 계산을 한다.
    #print(f"bce : {bce}")
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