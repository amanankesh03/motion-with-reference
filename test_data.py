import torch
import os
from torchvision.io import read_video

video_dir = '/data/dagan_data/processed_frames'
vid_l = os.listdir(video_dir)

video_path = os.path.join(video_dir, vid_l[1]) 
print(f'video_path : {video_path}')

frames, _, _ = read_video(str(video_path), output_format="TCHW")

# img1_batch = torch.stack([frames[100], frames[150]])
# img2_batch = torch.stack([frames[101], frames[151]])
print(frames)
# plot(img1_batch)