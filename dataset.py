import torch, cv2, os
import numpy as np

class HeartDiseaseDataset(torch.utils.data.Dataset):
  def __init__(self, img_ids, img_dir, mask_dir, num_classes, transform = None):

    self.img_ids = img_ids
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.num_classes = num_classes
    self.transform = transform

  def _pad_img(self, img):
    shape = img.shape

    w,h = shape[1], shape[0]

    if (shape[0] != 768 or shape[1] != 1024):
      new_img = np.zeros(shape = (768, 1024,3), dtype = np.uint8)
    else:
      new_img = img
    
    dh = int((768 - shape[0])/2)
    dw = int((1024 - shape[1])/2)

    #new_img[:,:,-1] = 255 # 마지막 alpha channel값은 255로
    new_img[dh:dh+h,dw:dw+w,:] = img[:,:,:]

    return new_img


  def __len__(self):
    return len(self.img_ids)

  def __getitem__(self, idx):
    """
    idx (int) : numbers of the index for the image id in the list
    """
    img_id = self.img_ids[idx]
    img = cv2.imread(os.path.join(self.img_dir, img_id+'.png'))
    img_shape = img.shape[:-1] # (W,H)
  
    mask = np.load(os.path.join(self.mask_dir, img_id+'.npy'))

    img = self._pad_img(img)
    mask = np.expand_dims(mask, axis = -1)
    mask = self._pad_img(mask)

    if self.transform is not None:
      transformed = self.transform(image = img, mask = mask)
      img = transformed['image']
      mask = transformed['mask']

  
    img = img.astype('float32') / 255 # 0-255사이의 uint8 -> 0-1 사이의 float32
    if len(img.shape) == 2:
      img = img.expand_dims(img, axis = -1)
    print(f"img : {img.shape}")
    print(f"mask : {mask.shape}")
    img = img.transpose(2, 0, 1) # (3, 434, 636) 또는 (3, 600, 800)

    mask = mask.transpose(2, 0, 1)

    return img, mask, {'img_id' : img_id, 'img_shape':img_shape}
