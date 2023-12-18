import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from augmentations import AugmentationTransform
from PIL import ImageFile
from torchvision.io import read_video
import torch
import gc
from networks.landmarks import PIPNet
from torchvision import transforms
import pickle
import numpy as np
import subprocess
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Vox256(Dataset):
    def __init__(self, dirs, split, transform=None, augmentation=False, use_lms=False):
        self.videos = []
        self.use_ref = False
        if split == 'train':
            #self.ds_path1 = '../../../../../data/dagan_data/processed_frames'  # 'F:/data-face-annimation/Processed_frames_512/processed_webm_all_processed/'
            #self.ds_path2 = '../../../../../data/dagan_data/processed_webm_all_processed'
            for path in dirs:
                self.videos += [str(os.path.join(path, v)) for v in os.listdir(path)]
        elif split == 'test':
            self.ds_path = './datasets/vox/test'
        else:
            raise NotImplementedError
        if use_lms:
            self.videos = [v for v in self.videos if len(os.listdir(v)) > 10 and os.path.exists(v + '/lms.txt')]
            self.use_ref = True
        print(f'{len(self.videos)} videos found')
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform
        self.ref_img = None
        self.ids = None
        self.batch_lms = {}

    def set_ref(self, path):
        lm = []
        ks = []
        # try:
        lms = os.path.join(path, 'lms.txt')
        # if lms:
        with open(lms) as f:
            lines = f.readlines()
            for l in lines:
                content = l.split(':')
                k = content[0]
                v = content[1].split(',')
                if not os.path.exists(os.path.join(path, k)):
                    continue
                try:
                    v[-1] = v[-1].replace('\n', '')
                    v = np.array(v).reshape(-1, 2).astype('float32')
                    lm.append(v)
                    ks.append(k)
                    self.batch_lms[k] = v
                except:
                    continue
        if len(lm) == 0:
            print('No landmarks, returning')
            self.__getitem__(self.ids)
        lmss = np.array(lm)
        lip_landmarks = lmss[:, 49:60, ]
        ref_idx = np.argmax(np.max(lip_landmarks[:, :, 1], axis=1) - np.min(lip_landmarks[:, :, 1], axis=1))


        self.ref_img = Image.open(os.path.join(path, f'{ks[ref_idx]}')).convert('RGB')
        # print('\nref path : ', path, f'{ks[ref_idx]}')
        # print(abd)
        lms = np.concatenate([lm[ref_idx][2:16], lm[ref_idx][32:36]])
        lms = np.array(lms).astype(np.int32).reshape((-1, 1, 2))

        ##uncomment this for ref 
        # mask = np.zeros((512, 512), dtype=np.uint8)
        # cv2.fillPoly(mask, [lms], 1)
        # mask = np.stack([mask, mask, mask], axis=-1)
        # self.ref_img = Image.fromarray(np.array(self.ref_img) * mask)
        #####

        # self.batch_lms = np.array(lmss).reshape(-1, 2).astype('float32')
        # except:
        #     self.ref_img = Image.fromarray(np.zeros((512,512,3)))

    def get_lip_box(self, lms, img_ref):
        arr = lms[49:60]
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
        lip = np.asarray(img_ref)[int(b):int(b + d), int(a):int(a + c), :]
        try:
            lip = cv2.resize(lip, (64, 64))
            int_ = random.randint(1, 100)
            cv2.imwrite(f'./check/{int_}.jpg', lip)
        except:
            print('Could not generate lipbox')
            return np.array([0, 0, 0, 0])
        return np.array([int(a), int(b), int(a + c), int(b + d)])

    def __getitem__(self, idx):
        video_path = os.path.join(self.videos[idx])

        if self.use_ref:
            self.set_ref(video_path)
            img_ref = self.ref_img
        frames_paths = sorted(glob.glob(video_path + '/*.jpg'))
        nframes = len(frames_paths)
        lip_boxes = [np.array([0, 0, 0, 0])]
        items = random.sample(list(range(nframes)), 3)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')
        # Image.open(frames_paths[items[2]]).convert('RGB')
        target_lms = self.batch_lms.get(frames_paths[items[1]].split('/')[-1])
        lip_boxes[0] = self.get_lip_box(target_lms, img_target)

        if self.augmentation:
            img_source, img_target, img_ref = self.aug(img_source, img_target, img_ref)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)
            img_ref = self.transform(img_ref)
        self.ids = idx
        return img_source, img_target, img_ref, lip_boxes

    def __len__(self):
        return len(self.videos)


class Vox256_vox2german(Dataset):
    def __init__(self, transform=None):
        self.source_root = './datasets/german/'
        self.driving_root = './datasets/vox/test/'

        self.anno = pd.read_csv('pairs_annotations/german_vox.csv')

        self.source_imgs = os.listdir(self.source_root)
        self.transform = transform

    def __getitem__(self, idx):
        source_name = str('%03d' % self.anno['source'][idx])
        driving_name = self.anno['driving'][idx]

        source_vid_path = self.source_root + source_name
        driving_vid_path = self.driving_root + driving_name

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.source_imgs)


class Vox256_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class Vox256_cross(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.anno = pd.read_csv('pairs_annotations/vox256.csv')
        self.transform = transform

    def __getitem__(self, idx):
        source_name = self.anno['source'][idx]
        driving_name = self.anno['driving'][idx]

        source_vid_path = os.path.join(self.ds_path, source_name)
        driving_vid_path = os.path.join(self.ds_path, driving_name)

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.videos)


class Taichi(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/taichi/train/'
        else:
            self.ds_path = './datasets/taichi/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(True, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):

        video_path = self.ds_path + self.videos[idx]
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Taichi_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/taichi/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class TED(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/ted/train/'
        else:
            self.ds_path = './datasets/ted/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class TED_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/ted/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class VideoDataset(Dataset):
    def __init__(self, datasets, transform, augmentation, batch_size, vid_formats=['mp4'], frames_per_vid=0.8,
                 ref_img=False, img_size=512, device='cuda', lms_dir='./gbucket_lms', lms_device='cuda: 0'):
        self.videos = []
        for dir_ in datasets:
            self.videos += [str(os.path.join(dir_, v)) for v in os.listdir(dir_) if v.split('.')[-1] in vid_formats]
            # print(dir_)
            # print(d)
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
        self.kpd = PIPNet(device=lms_device)
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

        # mask = np.zeros((512, 512), dtype=np.float32)
        # cv2.fillPoly(mask, [lms], 255)
        # mask = torch.from_numpy(mask / 255).float()

        self.ref_img = self.resize(self.ref_img).float()

        # mask = torch.stack([mask, mask, mask], dim=0)
        # self.ref_img = (self.ref_img * mask).float()

        for i, lm in enumerate(self.lms):
            self.lms[i] = [[int(x / shape_x), int(y / shape_y)] for x, y in lm]
        self.lms = np.array(self.lms).astype('float32')

    def get_half_batch(self):
        lms = np.array(self.lms)
        # print(lms)
        lip_landmarks = lms[:, 76:93, :]
        result = np.max(lip_landmarks[:, :, 1], axis=1) - np.min(lip_landmarks[:, :, 1], axis=1)
        # sorted_result = np.sort(result)
        sorted_args = np.argsort(result)
        # for i in range(len(sorted_result)):
        #     print(sorted_args[i], sorted_result[i])
        return sorted_args.tolist()
    
    def __getitem__(self, index):
        source, target, ref = [], [], []
        lip_boxes = []

        self.get_frames()
        indices = self.get_half_batch()  

        hb = int(self.batch_size/2)  
              
        half_batch_src = indices[:hb]
        half_batch_tgt = indices[-hb:]  

        step_indexes = self.indices[:self.batch_size] 

        for idx in step_indexes[:hb]:
            half_batch_src.append(idx) 

        for idx in step_indexes[hb:]:
            half_batch_tgt.append(idx) 

        ref_indexes, img_ref = None, None   

        for i, (j, k) in enumerate(zip(half_batch_src, half_batch_tgt)):

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


    # def __getitem__(self, index):
    #     source, target, ref = [], [], []
    #     lip_boxes = []

    #     self.get_frames()
    #     step_indexes = self.indices[:self.batch_size * 2]
    #     ref_indexes, img_ref = None, None
    #     for i, (j, k) in enumerate(zip(step_indexes[:self.batch_size], step_indexes[self.batch_size:])):

    #         img_source = self.frames[j]
    #         img_target = self.frames[k]

    #         lip_boxes.append(self.get_lip_box(self.lms[k], img_target))
    #         if self.imgs == 3:
    #             img_ref = self.ref_img

    #         if self.aug:
    #             if self.imgs == 3:
    #                 img_source, img_target, img_ref = self.aug(img_source, img_target, img_ref)
    #             else:
    #                 img_source, img_target = self.aug(img_source, img_target)

    #         if self.transform is not None:
    #             source.append(self.transform(img_source))
    #             target.append(self.transform(img_target))
    #             if self.imgs == 3:
    #                 ref.append(self.transform(img_ref))
    #         else:
    #             source.append(img_source)
    #             target.append(img_target)
    #             if self.imgs == 3:
    #                 ref.append(img_ref)
    #     self.indices = self.indices[self.batch_size * 2:]

    #     if self.imgs == 3:
    #         return torch.stack(source), torch.stack(target), torch.stack(ref), lip_boxes
    #     return torch.stack(source), torch.stack(target), None, lip_boxes

if __name__ == '__main__':
    dataset = VideoDataset()