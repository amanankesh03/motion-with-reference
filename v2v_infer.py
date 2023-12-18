#src_vid = [sf1, sf2, ... sfn]
#drv_vid = [df1, df2, ... dfn]
#ref_img = sfk  (where sfk is the frame where opening of mouth of the person is max)


import matplotlib.pyplot as plt
import imageio
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
from torchvision.io import read_video
from FaceDetect import pipnet, face_detector_fa, ref_from_video
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_resized_video(vid):
    frame_list = []
    for frame in vid:
        pil_img = transforms.ToPILImage()(frame)
        pil_img = pil_img.resize((512, 512))
        ten_img = transforms.ToTensor()(pil_img)
        frame_list.append(ten_img.unsqueeze(0))
    return torch.cat(frame_list)

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
        self.save_path = os.path.join(self.save_path, Path(args.drv_video_path).stem + '.mp4')

    def run(self, src, target, ref):
        # print('==> running')
        with torch.no_grad():

            # vid_target_recon = []
            img_recon = self.gen(src.cuda(), target.cuda(), ref.cuda())

            # print("ref : ", ref.shape)
            # print("target : ",target.shape)
            # print("src : ", src.shape)
            # print('recon : ', img_recon.shape)

        self.save_image(img_recon, "output.png")
        return img_recon

def save_image(img, name):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    torchvision.utils.save_image(img, name)

def TensorToImage(img_tensor):
    img = img_tensor.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    return img

def test_img(args):
    src = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/src.png'
    drv = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/targt.png'
    ref = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/ref.png'


    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.size, args.size)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    src = Image.open(src)
    drv = Image.open(drv)
    ref = Image.open(ref)

    src = transform(src)
    drv = transform(drv)
    ref = transform(ref)

    demo = Demo(args)
    img = demo.run(src, drv, ref)
    img = TensorToImage(img)
    pth = os.path.join(args.save_folder, "output.png")
    save_image(img, pth)

def get_frames_tensor(video_path, transfrom):
    tensor_frames = []
    frames, _, fps = read_video(video_path)
    
    for frame in frames:
        frame = frame.permute(2, 0, 1).float()
        tensor_frame = transform(frame)
        tensor_frames.append(tensor_frame)
        # print(tensor_frame.shape)
    return torch.cat(tensor_frames), fps

def save_video_frames(video_path, output_directory):
    video_reader = imageio.get_reader(video_path)

    # Initialize a frame counter
    frame_count = 0

    # Define a torchvision transform to convert frames to tensors
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    # Loop through the frames and save them as images
    for frame in video_reader:
        # Convert the frame to a PyTorch tensor
        image_tensor = transform(frame)
        print(image_tensor.shape)
        # Save the tensor as an image file
        image_filename = os.path.join(output_directory, f'frame_{frame_count:04d}.jpg')
        torchvision.utils.save_image(image_tensor, image_filename)

        # Print progress
        print(f'Saved frame {frame_count}')

        # Increment the frame counter
        frame_count += 1
        break

    print('All frames saved successfully.')

def get_output(src_video_path, drv_video_path, save_video_path, video_transform, fps=30):

    driving_video = imageio.get_reader(drv_video_path)
    src_video = imageio.get_reader(src_video_path)
    ref = ref_from_video(src_video_path, face_detector_fa, pipnet)

    video_frames = []

    for src_frame, drv_frame in zip(src_video, driving_video):
        
        plt.imshow(drv_frame)
        plt.show()
        plt.imshow(src_frame)
        plt.show()
        drv = video_transform(drv_frame)
        src = video_transform(src_frame)
        

        img = demo.run(src, drv, ref)[0]


        img = transforms.ToPILImage()(TensorToImage(img)).resize( (int((1920/1080)*512), 512) )
        print('img shape ', img.size)
        img = transforms.ToTensor()(img).permute(1, 2, 0).cpu()
        # img = TensorToImage(img).permute(1,2,0).cpu()
        # img = img.resize((512, int((1920/1080)*512)))
        video_frames.append(img)

    video_frames = torch.stack(video_frames, dim=0)   
    imageio.mimsave(save_video_path, video_frames.numpy(), fps=fps)

    print(video_frames.shape)

if __name__ == '__main__':

    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--checkpoint_path", type=str, default='./exps/exp5oct2023/checkpoint/663000.pt')
    parser.add_argument("--src_video_path", type=str, default='./vid2vid_inference_data/driving')
    parser.add_argument("--drv_video_path", type=str, default='./vid2vid_inference_data/driving')
    parser.add_argument("--save_folder", type=str, default='./vid2vid_inference_data/output/test3')
    
    args = parser.parse_args()
    size = args.size

    src = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/src.png'
    drv = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/targt.png'
    ref = '/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/output/ref.png'

    # save_path = os.path.join(args.save_folder, "output_770000.png")
    # output_video_path = os.path.join(args.save_folder, 'output_770000.mp4')

    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.size, args.size)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )
    video_transform = torchvision.transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    demo = Demo(args)

    src_name = 'src.mp4'
    drv_name = 'drv.mp4'
    des_name = 'out.mp4'

    src_video_path = os.path.join(args.src_video_path, src_name)
    drv_video_path = os.path.join(args.drv_video_path, drv_name)
    save_folder = os.path.join(args.save_folder, des_name)

    get_output(src_video_path, drv_video_path, save_folder, video_transform)

    
    