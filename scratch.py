# import pandas as pd
# import random
# from PIL import Image
# from torch.utils.data import Dataset
# import glob
# import os
# from augmentations import AugmentationTransform
# from PIL import ImageFile
from torchvision.io import read_video
# import torch
# import gc
# from networks.landmarks import PIPNet
# from torchvision import transforms
# import pickle
# import numpy as np
# import subprocess
# import cv2
# from networks.landmarks import PIPNet
# import matplotlib.pyplot as plt
# import matplotlib
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# matplotlib.use("Agg")
# from tqdm import tqdm
# # pbar = tqdm(total=len(os.listdir('./gbucket_lms')))
# # for fil in os.listdir('./gbucket_lms'):
# #     vid_name = fil.replace('_landmarks.pkl', '_with_audio.mp4')
# #     vid = os.path.join('../../../../../data/gbucket_data', vid_name)
# #     with open(os.path.join('./gbucket_lms',fil), 'rb') as file:
# #         lms = pickle.load(file)
# #     frames, _, _ = read_video(vid, output_format='TCHW')
# #     lms = np.array(lms)
# #     max_x = np.max(lms[:,:,0])
# #     max_y = np.max(lms[:,:,1])
# #     if max_x > frames.shape[3] or max_y > frames.shape[2]:
# #         print('Error', frames.shape, max_x, max_y, fil)
# #     pbar.update(1)
# vid = os.path.join('../../../../../data/gbucket_data', 'Instant Gratification what is it, and how can we beat it _clip0059_000_with_audio.mp4')
# fc_path = os.path.join('../../../../../data/gbucket_data', 'Why you need contingency plans when managing projects._clip0033_000_face_coordinates.txt')
# frames, _, _ = read_video(vid, output_format='TCHW')
# print(frames.shape)
# coordinates = []
# kpd = PIPNet(device='cuda:0')
# scale = max(max(frames.shape[2], frames.shape[3]) / 640, 1)
# # scale = 1
# rsz = transforms.Resize((int(frames.shape[2] // scale), int(frames.shape[3] // scale)))
# frames = [rsz(frame) for frame in frames]
# with open(fc_path, 'r') as file:
#     for i, line in enumerate(file):
#         x1, y1, x2, y2 = map(int, line.strip().split(','))
#         coordinates.append([[int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]])
# lms = kpd(frames, coordinates)
# lm = lms[0]
# im_arr = np.array(frames[0].permute(1,2,0))
# print(im_arr.shape)
# print(lm)
# for pts in lm:
#     x, y = pts
#     im_arr[int(y),int(x),:] = [255, 255, 255]
#     x1, y1, x2, y2 = (np.array(coordinates[0][0]) * scale).astype(int)
#     bbox = matplotlib.patches.Rectangle((x1, y1),x2-x1,y2-y1, linewidth=2, edgecolor='blue',facecolor='none')
#     plt.gca().add_patch(bbox)
# plt.imshow(im_arr)
# plt.savefig("outputt.png")
# print(max(np.array(lm[:,:,0])), max(np.array(lm[:,:,1])))
import os
from collections import Counter
from scipy.stats import gaussian_kde

# l = [s.split('_')[0] for s in os.listdir('../../../../../data/gbucket_data') if '.mp4' in s]
# l = Counter(l)
# thresh = sum(l.values())/len(l)
# p_map = {k:min(1,thresh/v) for k,v in l.items()}
# print(p_map)
vids = [s for s in os.listdir('../../../../../data/gbucket_data') if '.mp4' in s][:10]
frame_ct = {v.split('_')[0]: 0 for v in vids}
total = 0

for v in vids:
    vid = vid = os.path.join('../../../../../data/gbucket_data', v)
    frames, _, _ = read_video(vid, output_format='TCHW')
    frames = frames.shape[0]
    frame_ct[v.split('_')[0]] += frames
    total += frames
    # if frames in f_bins:
    #     f_bins[frames] += 1
    # else:
    #     f_bins[frames] = 1
f_bins = [v for v in frame_ct.values()]

kde = gaussian_kde(f_bins, bw_method='silverman')
frame_ct2 =frame_ct.copy()
for k, v in frame_ct.items():
    frame_ct[k] = kde.evaluate(v)
    frame_ct2[k] = v/total
sum_ = 0
for v in frame_ct.values:
    sum_ += v
print(sum_)
sum_ = 0
for v in frame_ct2.values:
    sum_ += v
print(sum_)
print(frame_ct)
print(frame_ct2)
