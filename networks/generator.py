from torch import nn
from .encoder import Encoder
from .encoder import EncoderRef
from .styledecoder import Synthesis


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)

        #reference Encoder
        self.refenc = EncoderRef(size, style_dim)  

        #synthesis/decoder network   
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)
        print('loaded synthesis network')

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)
        return img

    def forward(self, img_source, img_drive, img_ref=None, h_start=None):
        #img_ref = img_source
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        if img_ref is not None:
            #_,_,ref_feats = self.enc(img_ref, None)
            
            ref_feats = self.refenc(img_ref)
            img_recon = self.dec(wa, alpha, feats, ref_feats)
        else:
            img_recon = self.dec(wa, alpha, feats)
        #print(img_recon.shape)

        return img_recon
