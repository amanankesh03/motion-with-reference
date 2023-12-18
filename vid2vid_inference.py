#src_vid = [sf1, sf2, ... sfn]
#drv_vid = [df1, df2, ... dfn]
#ref_img = sfk  (where sfk is the frame where opening of mouth of the person is max)



import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils import data
from inference_dataset import VideoDataset
from PIL import ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch



def get_resized_video(vid):
    # print('vid : ', vid.shape)
    frame_list = []
    for frame in vid:
        pil_img = transforms.ToPILImage()(frame)
        pil_img = pil_img.resize((512, 512))
        ten_img = transforms.ToTensor()(pil_img)
        frame_list.append(ten_img.unsqueeze(0))

    return torch.cat(frame_list)

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    # print("vid_dic: ", vid_dict[0][0].shape)

    vid = vid_dict[0].permute(0, 3, 1, 2)

    if vid[0].shape[1] != 512:
        vid = get_resized_video(vid)
        print(vid.shape)

    vid = vid.unsqueeze(0)

    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args
        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval() 

        print('==> loading data')
        self.save_path = args.save_folder
        os.makedirs(self.save_path, exist_ok=True)
        self.img_save_path = self.save_path
        self.save_path = os.path.join(self.save_path, Path(args.driving_path).stem + '.mp4')

        # self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        # self.vid_target, self.fps = vid_preprocessing(args.driving_path)
        # self.vid_target = self.vid_target.cuda()

        # self.vid_src, self.fps = vid_preprocessing(args.source_path)
        # self.vid_src = self.vid_src.cuda()

    def run(self, src, target, ref):

        # driving_video, fps = vid_preprocessing(driving_path)
        print('==> running')


        with torch.no_grad():

            vid_target_recon = []

            # for i in tqdm(range(driving_video.size(1))):
            #     img_target = driving_video[:, i, :, :, :]


            img_recon = self.gen(src.cuda(), target.cuda(), ref.cuda())
            print("ref : ", ref.shape)
            print("target : ",target.shape)
            print("src : ", src.shape)
            print('recon : ', img_recon.shape)
            # vid_target_recon.append(img_recon.unsqueeze(2))
            # break

            self.display_img(target, 'target.png')
            self.display_img(img_recon, 'output.png')
            self.display_img(ref, 'ref.png')
            self.display_img(src, 'src.png')
            # vid_target_recon = torch.cat(vid_target_recon, dim=2)
            # save_video(vid_target_recon, self.save_path, fps)

    def display_img(self, img, name):
        img = img.clamp(-1, 1)
        img = ((img - img.min()) / (img.max() - img.min())).data
        torchvision.utils.save_image(img, os.path.join(self.img_save_path, name))

if __name__ == '__main__':

    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, default='./exps/exp5oct2023/checkpoint/663000.pt')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--driving_path", type=str, default='/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/driving/1.mp4')
    parser.add_argument("--save_folder", type=str, default='./vid2vid_inference_data/output/test/')
   
    args = parser.parse_args()

    data_dir = ['/data/gbucket_data']
    lms_dir = './gbucket_lms'
    
    size = 512
    batch_size = 1
    
    src = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/src.png'
    drv = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/targt.png'
    ref = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/ref.png'

    src = Image.open(src)
    drv = Image.open(drv)
    ref = Image.open(ref)

    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )


    # dataset = VideoDataset(data_dir, transform, False, batch_size,
    #     ref_img=True, device='cpu', lms_device= 'cpu',
    #     lms_dir=lms_dir
    # )
    
    # print("len : ",len(dataset))
    # loader = data.DataLoader(dataset, num_workers=8, batch_size=None, pin_memory=True)
    # loader = sample_data(loader)

    # img_source, img_target, img_ref, lip_boxes = next(loader)
    # img_source, img_target, img_ref, lip_boxes = dataset[0]
    # print(img_source.shape, img_target.shape, img_ref.shape, lip_boxes)

    src = transform(src)
    drv = transform(drv)
    ref = transform(ref)

    # demo
    demo = Demo(args)
    demo.run(src, drv, ref)
