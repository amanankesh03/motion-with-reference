import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from torch.utils import data
from augmentations import AugmentationTransform
from PIL import ImageFile
from torchvision.io import read_video
import torch
import gc
from networks.landmarks import PIPNet
import torchvision
from torchvision import transforms
import pickle
import numpy as np
import subprocess
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

class VideoDataset(Dataset):
    def __init__(self, datasets, transform, augmentation, batch_size, vid_formats=['mp4'], frames_per_vid=0.8,
                 ref_img=False, img_size=512, device='cpu', lms_dir='./gbucket_lms', lms_device='cpu'):
        self.videos = []
        for dir_ in datasets:
            
            self.videos += [str(os.path.join(dir_, v)) for v in os.listdir(dir_) if v.split('.')[-1] in vid_formats]
            self.videos = [v for v in self.videos if
                           os.path.exists(v.replace('_with_audio.mp4', '_face_coordinates.txt'))]

        self.transform = transform
        self.aug = augmentation
        self.batch_size = batch_size
        self.frames_per_vid = frames_per_vid
        self.num_vid_frames = 0
        self.frame_ids = []
        self.frames = []
        self.lms = []
        self.indices = []
        self.kpd = PIPNet()
        
        self.ref_img = None
        self.device = device
        if not os.path.exists(lms_dir):
            os.mkdir(lms_dir)
        self.lms_dir = lms_dir
        if not ref_img:
            self.imgs = 2
        else:
            self.imgs = 3
        self.resize = transforms.Resize((img_size, img_size))
        self.lip_box_resize = transforms.Resize((64, 64))
        self.ratio = None
        self.ids = None

    def __len__(self):
        return len(self.videos)

    def read_video(self):
        idx = random.randint(0, len(self.videos))
        self.frames, _, _ = read_video(str(self.videos[idx]), output_format='TCHW')
        self.get_fb_lms(idx)
        self.indices = random.sample(range(0, len(self.frames)), len(self.frames))
        self.num_vid_frames = len(self.frames)

    def get_fb_lms(self, idx):

        self.lms = {}
        path = self.videos[idx]
        fc_path = path.replace('_with_audio.mp4', '_face_coordinates.txt')

        try:
            lms_path = path.replace('_with_audio.mp4', '_landmarks.pkl').split('/')[-1]
            lms_path = os.path.join(self.lms_dir, lms_path)
            if not os.path.exists(lms_path):

                coordinates = []
                scale = max(max(self.frames.shape[2] / 640, self.frames.shape[3]) / 640, 1)
                rsz = transforms.Resize((int(self.frames.shape[2] // scale), int(self.frames.shape[3] // scale)))
                frames = [rsz(frame / 255) for frame in self.frames]

                with open(fc_path, 'r') as file:
                    for i, line in enumerate(file):
                        x1, y1, x2, y2 = map(int, line.strip().split(','))
                        coordinates.append([[int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]])
                self.lms = (np.array(self.kpd(frames, coordinates)) * scale).tolist()

                lms = np.array(self.lms)[:, 76:93, :]
                with open(lms_path, 'wb') as file:
                    pickle.dump(np.array(self.lms), file)
            else:
                with open(lms_path, 'rb') as file:
                    self.lms = pickle.load(file)

        except Exception as e:

            print('REREADING', np.array(self.lms).shape, e)
            self.read_video()
        self.set_ref_img()

        self.frames = [self.resize(frame / 255) for frame in self.frames]

    def get_frames(self):
        while (len(self.indices) < self.num_vid_frames * (1 - self.frames_per_vid)) or (len(self.indices) <
                                                                                        (self.batch_size * 2)):
            try:
                self.read_video()
            except Exception as e:
                print('Rereading', e)
                self.read_video()
        return

    def get_lip_box(self, lms, img):
        arr = lms[76:87]
        a, b, c, d = cv2.boundingRect(arr)
        if c < d:
            c = d
        else:
            d = c
        c = c + 0.1 * c
        d = d + 0.1 * d
        a = a - 0.1 * c
        b = b - 0.1 * d
        diff = 64 - c

        if diff > 0:
            a -= diff // 2
            c += diff
            b -= diff / 2
            d += diff

        try:
            lip = img[:, int(b):int(b + d), int(a):int(a + c)]
            lip = (lip.permute(1, 2, 0).numpy() * 255).astype('uint8')
            lip = cv2.resize(lip, (64, 64))
            int_ = random.randint(1, 100)

        except:
            print('Could not generate lipbox')
            return np.array([0, 0, 0, 0])

        return np.array([int(a), int(b), int(a + c), int(b + d)])

    def set_ref_img(self):
        lms = np.array(self.lms)
        lip_landmarks = lms[:, 76:93, :]

        # Find the difference between the highest and lowest y-coordinate for each frame's lip landmarks
        ref_idx = np.argmax(np.max(lip_landmarks[:, :, 1], axis=1) - np.min(lip_landmarks[:, :, 1], axis=1))
        self.ref_img = self.frames[ref_idx] / 255
        shape_y, shape_x = self.ref_img.shape[1] / 512, self.ref_img.shape[2] / 512
        
        if not isinstance(self.lms, list):
            self.lms = self.lms.tolist()
        lms = self.lms[ref_idx][3:30] + self.lms[ref_idx][55:59]
        lms = [[int(x / shape_x), int(y / shape_y)] for x, y in lms]
        lms = np.array(lms).astype(np.int32).reshape((-1, 1, 2))
        mask = np.zeros((512, 512), dtype=np.float32)
        cv2.fillPoly(mask, [lms], 255)
        mask = torch.from_numpy(mask / 255).float()
        self.ref_img = self.resize(self.ref_img)
        mask = torch.stack([mask, mask, mask], dim=0)
        # print(mask.shape, self.ref_img.shape, mask.dtype, self.ref_img.dtype, torch.sum(mask))
        self.ref_img = (self.ref_img * mask).float()
        for i, lm in enumerate(self.lms):
            self.lms[i] = [[int(x / shape_x), int(y / shape_y)] for x, y in lm]
        self.lms = np.array(self.lms).astype('float32')

    def __getitem__(self, index):
        source, target, ref = [], [], []
        lip_boxes = []

        self.get_frames()
        # print('self.indices : ', self.indices)
        step_indexes = self.indices[:self.batch_size * 2]
        ref_indexes, img_ref = None, None
        for i, (j, k) in enumerate(zip(step_indexes[:self.batch_size], step_indexes[self.batch_size:])):

            img_source = self.frames[j]
            img_target = self.frames[k]

            lip_boxes.append(self.get_lip_box(self.lms[k], img_target))
            if self.imgs == 3:
                img_ref = self.ref_img

            if self.aug:
                if self.imgs == 3:
                    img_source, img_target, img_ref = self.aug(img_source, img_target, img_ref)
                else:
                    img_source, img_target = self.aug(img_source, img_target)

            if self.transform is not None:
                source.append(self.transform(img_source))
                target.append(self.transform(img_target))
                if self.imgs == 3:
                    ref.append(self.transform(img_ref))
            else:
                source.append(img_source)
                target.append(img_target)
                if self.imgs == 3:
                    ref.append(img_ref)
        self.indices = self.indices[self.batch_size * 2:]

        if self.imgs == 3:
            return torch.stack(source), torch.stack(target), torch.stack(ref), lip_boxes
        return torch.stack(source), torch.stack(target), None, lip_boxes

if __name__ == '__main__':
    
    data_dir = ['/home/YOTTAVDI/neuralgarage4/Processed/Motion-with-ref/vid2vid_inference_data/driving']
    lms_dir = './gbucket_lms'
    
    size = 512
    batch_size = 1


    transform = torchvision.transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )


    dataset = VideoDataset(data_dir, transform, False, batch_size,
        ref_img=True, device='cpu', lms_device= 'cpu',
        lms_dir=lms_dir
    )
    
    print("len : ",len(dataset))
    loader = data.DataLoader(dataset, num_workers=8, batch_size=None, pin_memory=True)
    loader = sample_data(loader)

    img_source, img_target, img_ref, lip_boxes = next(loader)
    print('238')
    print(img_source.shape, img_target.shape, img_ref.shape)
    print(lip_boxes)