class InpaintSnowEmptyPatchAug(torch.nn.Module):
    def __init__(self,p = 1.0, inpaint_classes = 'snow',
                 snow_masks_dir = 'snow_masks_for_inpainting/*'):
        super().__init__()
        self.p = p
       
        self.snow_masks_dir = glob.glob(snow_masks_dir)
        self.inpaint_classes = inpaint_classes
        self.seg_classes = {
                           'snow':2,}
        self.models_dict = {
                           'snow':'Inpaint_models/UNet_Ef_b0_InpaintSnow_2losses_ep_300',
                           }
        self.model_sn = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=8,
            classes=8)
        self.model_sn.load_state_dict(torch.load(self.models_dict['snow']))
        
    def forward(self, datacube, seglabel):
        self.model_sn.eval().cuda()
        eps = np.random.random()
        mask_idx = np.random.randint(0, len(self.snow_masks_dir))
        if (self.p>eps): 
            snow_mask_inv = tff.imread(self.snow_masks_dir[mask_idx])
            snow_mask = snow_mask_inv.copy()
            snow_mask[snow_mask==0] = 2
            snow_mask[snow_mask==1] = 0
            
            corrupted_bands_sn = datacube.clone().detach()
            corrupted_bands_all = datacube.clone().detach()
            for i in range(len(datacube)):
                
                corrupted_bands_sn[i] = corrupted_bands_sn[i]*snow_mask_inv
       
            #snow_mask = snow_mask*shadow_mask_inv*cloud_mask_inv # for avoiding overlapping
            datacube_sn = self.model_sn(corrupted_bands_sn.type(torch.float32).cuda().unsqueeze(0))
            datacube_sn = torch.clip(datacube_sn, min=0, max=1)
            datacube_sn = datacube_sn.type(torch.float32).squeeze(0).cpu()
            for i in range(len(datacube)):
                datacube_sn[i][snow_mask==0] = 0
            datacube = corrupted_bands_all+datacube_sn
            new_mask = torch.tensor(snow_mask)
            return datacube.cpu(),new_mask
        else: return datacube.cpu(), seglabel