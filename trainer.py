import torch
from networks.discriminator import Discriminator
from networks.generator import Generator
import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from torch.nn.parallel import DistributedDataParallel as DDP


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        
        self.lambda_L1 = 1 # earlier 10
        self.lambda_adv = 0.1
        self.lambda_vgg = 10 #earlier 1

        self.lambda_Lip = 1
        self.lambda_vgg_lip = 1 #recently added

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier)
        print('loaded Generator')

        self.gen = self.gen.to(device)
        print('moved to gpu')
        print(self.gen)

        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.gen.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        # if resume_ckpt is not None:
        #     print(f'loading ckpt : {resume_ckpt}')
        #     self.resume(resume_ckpt)

        # distributed computing
        # self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        # self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        self.criterion_vgg = VGGLoss().to(rank)
        self.rank = rank

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, img_source, img_target, img_ref, lip_boxes=None):
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True)
        requires_grad(self.dis, False)

        img_target_recon = self.gen(img_source, img_target, img_ref)
        
        # print('test', img_target_recon.shape, img_target_recon.dtype, img_target_recon.max(), img_target_recon.min())
        # print('Batch range', img_target_recon.min(), img_target_recon.max())
        img_recon_pred = self.dis(img_target_recon)

        lip_loss = torch.tensor(0).to(self.rank).float()
        if lip_boxes is not None:
            for i in range(len(lip_boxes)):
                try:
                    x1, y1, x2, y2 = lip_boxes[i]
                except:
                    x1, y1, x2, y2 = lip_boxes[0][i]
                if x2 - x1 <=0 or y2 - y1 <=0:
                    print('Improper lip box')
                    loss = torch.tensor(1e-6).to(self.rank).float()
                    continue
                loss = torch.mean(torch.abs(img_target_recon[i, :, y1:y2, x1:x2] - img_target[i, :, y1:y2, x1:x2]))
                if torch.isnan(loss):
                    print('Nan encountered')
                    loss = torch.tensor(1e-6).to(self.rank).float()
                lip_loss += loss

        ## weighted loss 
        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()    
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)                
        l1_loss = F.l1_loss(img_target_recon, img_target)                      

        lip_loss = lip_loss                                                  
        vgg_lip_loss = self.criterion_vgg(img_target_recon[:, :, 143:367, 288:], img_target[:, :, 143:367, 288:]).mean() 

        g_loss = self.lambda_adv * gan_g_loss + self.lambda_vgg * vgg_loss + self.lambda_vgg_lip * vgg_lip_loss #+ self.lambda_L1 * l1_loss + self.lambda_Lip * lip_loss 
        ## 

        g_loss.backward()
        self.g_optim.step()

        loss_dict = {
            'vgg_loss' : vgg_loss.item(),
            'l1_loss' : l1_loss.item(),
            'gan_g_loss' : gan_g_loss.item(),
            'vgg_lip_loss' : vgg_lip_loss.item(),
            'total_g_loss' : g_loss.item(),
        }

        if lip_boxes is not None:
            loss_dict['lip_loss'] = lip_loss.item()
            return loss_dict, img_target_recon

        return loss_dict, img_target_recon

    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        real_img_pred = self.dis(img_real)
        recon_img_pred = self.dis(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, img_source, img_target, img_ref):
        with torch.no_grad():
            self.gen.eval()

            img_recon = self.gen(img_source, img_target, img_ref)
            img_source_ref = self.gen(img_source, None, None)
            img_src_ref = self.gen(img_source, None, img_ref)

        return img_recon, img_source_ref, img_src_ref

    def resume(self, args):
        print("load model:", args.resume_ckpt)
        ckpt = torch.load(args.resume_ckpt)
        ckpt_name = os.path.basename(args.resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])
        
        self.gen.load_state_dict(ckpt["gen"])
        self.dis.load_state_dict(ckpt["dis"])
        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])

        print('\n model loaded \n')
        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.state_dict(),
                "dis": self.dis.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )

    def resume_(self, args):
        print("load model:", args.resume_ckpt)
        ckpt = torch.load(args.resume_ckpt)
        ckpt_name = os.path.basename(args.resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])

        ### Adding new parameters 
        weight = ckpt['gen']

        state_dict = self.gen.state_dict()

        refenc_dict = {}
        for key in state_dict.keys():
            if 'refenc' in key:
                new_key = key.replace('refenc', 'enc')
                state_dict[key] = weight[new_key].clone()
                
            else:
                state_dict[key] = weight[key]


        # params_dict = {
        #     'dec.to_flows.0.alpha' : 512,
        #     'dec.to_flows.1.alpha' : 512,
        #     'dec.to_flows.2.alpha' : 512,
        #     'dec.to_flows.3.alpha' : 256,
        #     'dec.to_flows.4.alpha': 128,
        #     'dec.to_flows.5.alpha' : 64,
        #     'dec.to_flows.6.alpha' : 32
        # }  

        # restore_dict = [
        #     'dec.to_flows.o.conv.weight', 
        #     'dec.to_flows.o.bias'
        # ]
        # i = 0
        # for key in params_dict.keys():
        #     weight[key] = torch.rand(1, params_dict[key], 1, 1)
            # weight[f'dec.to_flows.{i}.conv.weight'] = torch.rand(1, 3, params_dict[key], 1, 1)
            # weight[f'dec.to_flows.{i}.bias'] = torch.rand(1, 3, 1, 1)
            # print(key, weight[key].shape)
            # i+=1
        # for i in range(0, 7):
        #     weight[f'dec.to_flows.{i}.conv.weight'] = torch.rand(1, params_dict[key], 1, 1)
        #     weight[f'dec.to_flows.{i}.bias'] = weight[f'dec.to_flows.{i}.bias'][:, :3, :, :]

        self.gen.load_state_dict(state_dict)
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

        self.g_optim = optim.Adam(
            self.gen.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )
        
        ###

        # self.gen.load_state_dict(ckpt["gen"])
        self.dis.load_state_dict(ckpt["dis"])
        # self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])

        # for param_group in self.g_optim.param_groups:
        #     param_group['lr'] /= 10
        
        # for param_group in self.d_optim.param_groups:
        #     param_group['lr'] /= 10

        print('\n model loaded \n')
        return start_iter