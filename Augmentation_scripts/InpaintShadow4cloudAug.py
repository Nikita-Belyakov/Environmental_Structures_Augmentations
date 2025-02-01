import numpy as np
import torch
def open_sample_as_3bands(datacube):
    rgb = np.dstack((datacube[0,:,:],datacube[1,:,:],datacube[2,:,:]))*255
    return rgb.astype('uint8')
class InpaintShadow4cloudAug(torch.nn.Module):
    def __init__(self,p = 1.0, max_shift = 50, full_img_inpaint = False, max_ratio_of_clouds = 0.5):
        super().__init__()
        self.p = p
        self.cls = 'clouds'
        self.full_img_inpaint = full_img_inpaint
        self.max_shift = max_shift
        self.max_ratio_of_clouds = max_ratio_of_clouds
        self.dx = np.random.randint(10,self.max_shift)
        self.dy = np.random.randint(10,self.max_shift)
        self.seg_classes = {'bg':0,
                           'shadows':1,
                           'snow':2,
                           'clouds':3}
        self.model_dir = 'Inpaint_models/UNet_Ef_b0_InpaintShadows_2losses_ep_150'
        self.model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=8,
            classes=8)
    def forward(self, datacube, seglabel):
        self.model.load_state_dict(torch.load(self.model_dir))
        self.model.eval().cuda()
        eps = np.random.random()
        if self.p>eps:
            mask = seglabel.clone()
            mask[mask!=self.seg_classes[self.cls]] = 0 # leave only 1 class(clouds) in mask for further inpainting
            clouds_ratio = torch.sum(mask)/(256*256*3)
            if clouds_ratio<=self.max_ratio_of_clouds:
                shadow_mask = shift_image(mask, dx = self.dx, dy = self.dy)
                shadow_mask[shadow_mask!=0] = 1
                shadow_mask[mask!=0] = 0
                new_mask = torch.tensor(shadow_mask)+(seglabel)
                corrupted_bands = datacube.clone()
                for i in range(len(datacube)):
                    corrupted_bands[i][shadow_mask!=0] = 0
                datacube_ = self.model(corrupted_bands.cuda().unsqueeze(0))
                datacube_ = torch.clip(datacube_, min=0, max=1)
                datacube_ = datacube_.type(torch.float32).squeeze(0).cpu()
                if self.full_img_inpaint: 
                    return datacube_.cpu(),new_mask
                else:
                    for i in range(len(datacube)):
                        datacube_[i][shadow_mask==0] = 0
                    datacube = corrupted_bands+datacube_
                    return datacube.cpu(),new_mask
            else: 
                return datacube.cpu(),seglabel