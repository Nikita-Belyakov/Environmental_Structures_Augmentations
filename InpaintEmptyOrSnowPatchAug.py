class InpaintEmptyOrSnowPatchAug(torch.nn.Module):
    def __init__(self,p = 1.0, inpaint_classes = ['shadows', 'clouds'],
                 cloud_masks_dir = 'cloud_masks_for_inpainting/*',
                 shadows_masks_dir = 'shadows_masks_for_inpainting/*',
                 snow_masks_dir = 'snow_masks_for_inpainting/*'):
        super().__init__()
        self.p = p
        self.cloud_masks_dir = glob.glob(cloud_masks_dir)
        self.shadows_masks_dir = glob.glob(shadows_masks_dir)
        self.inpaint_classes = inpaint_classes
        self.seg_classes = {'bg':0,
                           'shadows':1,
                           'snow':2,
                           'clouds':3}
        self.models_dict = {'shadows':'Inpaint_models/UNet_Ef_b0_InpaintShadows_2losses_ep_150',
                           'clouds':'Inpaint_models/UNet_Ef_b0_InpaintClouds_2losses_ep_100'}
        self.model_sh = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=8,
            classes=8)
        self.model_cl = deepcopy(self.model_sh)
        self.model_sh.load_state_dict(torch.load(self.models_dict['shadows']))
        self.model_cl.load_state_dict(torch.load(self.models_dict['clouds']))
    def forward(self, datacube, snow_mask):
        self.model_cl.eval().cuda()
        self.model_sh.eval().cuda()
        eps = np.random.random()
        mask_idx = np.random.randint(0, len(self.shadows_masks_dir))
        if (self.p>eps): 
            cloud_mask_inv = tff.imread(self.cloud_masks_dir[mask_idx])
            shadow_mask_inv = tff.imread(self.shadows_masks_dir[mask_idx])
            # cloud & shadows masks
            shadow_mask = shadow_mask_inv.copy()
            shadow_mask[shadow_mask==0] = 2
            shadow_mask[shadow_mask==1] = 0
            shadow_mask = shadow_mask//2 # shadow mask is labeled as 1
            cloud_mask = cloud_mask_inv.copy()
            cloud_mask[cloud_mask==0] = 3
            cloud_mask[cloud_mask==1] = 0
            snow_mask = snow_mask*shadow_mask_inv*cloud_mask_inv # for avoiding overlapping
            corrupted_bands_sh, corrupted_bands_cl = datacube.clone(),datacube.clone()
            corrupted_bands_all = datacube.clone()
            for i in range(len(datacube)):
                corrupted_bands_sh[i] = corrupted_bands_sh[i]*shadow_mask_inv
                corrupted_bands_cl[i] = corrupted_bands_cl[i]*cloud_mask_inv
                corrupted_bands_all[i] = corrupted_bands_all[i]*cloud_mask_inv*shadow_mask_inv
            datacube_cl = self.model_cl(corrupted_bands_cl.type(torch.float32).cuda().unsqueeze(0))
            datacube_cl = torch.clip(datacube_cl, min=0, max=1)
            datacube_cl = datacube_cl.type(torch.float32).squeeze(0).cpu()
            datacube_sh = self.model_sh(corrupted_bands_sh.type(torch.float32).cuda().unsqueeze(0))
            datacube_sh = torch.clip(datacube_sh, min=0, max=1)
            datacube_sh = datacube_sh.type(torch.float32).squeeze(0).cpu()
            datacube_sn = torch.zeros(datacube_sh.shape)# just 4 working without snow inpainting
            for i in range(len(datacube)):
                datacube_cl[i][cloud_mask==0] = 0
                datacube_sh[i][shadow_mask==0] = 0
            datacube = corrupted_bands_all+datacube_cl+datacube_sh
            new_mask = torch.tensor(shadow_mask)+torch.tensor(cloud_mask)+torch.tensor(snow_mask)
            return datacube.cpu(),new_mask
        else: return datacube.cpu(), snow_mask