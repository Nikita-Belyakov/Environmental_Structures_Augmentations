import numpy as np
import torch
def shift_image(X,dx,dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X
class DublicateMaskAug(torch.nn.Module):
    def __init__(self,p=1.0,cls='shadows',max_shift=10):
        super().__init__()
        self.p = p
        self.cls = cls
        self.max_shift = max_shift
        self.dx = np.random.randint(5,self.max_shift)
        self.dy = np.random.randint(5,self.max_shift)
        self.seg_classes = {'bg':0,
                           'shadows':1,
                           'snow':2,
                           'clouds':3}
    def forward(self, datacube, seglabel):
        eps = np.random.random()
        if self.p>eps:
            mask = seglabel.clone()
            mask[mask!=self.seg_classes[self.cls]] = 0 # leave only 1 class in mask for further inpainting
            mask_area_bands = datacube.clone()
            shifted_mask = shift_image(mask, dx = self.dx, dy = self.dy)
            for i in range(len(datacube)):
                mask_area_bands[i][mask==0] = 0
            seglabel[shifted_mask!=0] = 0
            mask_area_bands = mask_area_bands.numpy()
            corrupted_bands = datacube.clone()
            datacube = datacube.numpy()
            corrupted_bands= corrupted_bands.numpy()
            for i in range(len(datacube)):
                corrupted_bands[i][shifted_mask!=0] = 0
            for i in range(len(mask_area_bands)):
                mask_area_bands[i]= shift_image(mask_area_bands[i], dx = self.dx, dy = self.dy)
            datacube = corrupted_bands+mask_area_bands
            new_mask = torch.tensor(shifted_mask)+(seglabel)
        return torch.tensor(datacube), new_mask