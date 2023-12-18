import os
from torchvision.io import read_video

path = '/data/gbucket_data'
log_file = './stats.txt'

def get_stats(path):
    stat_dict = {}
    vid_fldrs = os.listdir(path)
    for vid_fldr in vid_fldrs:
        vid_list = os.listdir(os.path.join(path, vid_fldr))
        stat_dict[vid_fldr] = len(vid_list)
    
    return stat_dict   

# for i, key in enumerate(stat_dict.keys()):
#     print(i, key, stat_dict[key])
    



vids = [s for s in os.listdir(path) if '.mp4' in s]
frame_ct = {v.split('_')[0]: 0 for v in vids}
total = 0
with open(log_file, 'w') as file:

    for i,v in enumerate(vids):
        vid = os.path.join(path, v)
        frames, _, _ = read_video(vid, output_format='TCHW')
        frames = frames.shape[0]
        key = v.split('_')[0]
        frame_ct[key] += frames
        total += frames
        file.write(f'{key} {frame_ct[key]}\n')
        print(i)
    file.close()

#     # if frames in f_bins:
#     #     f_bins[frames] += 1
#     # else:
#     #     f_bins[frames] = 1
# f_bins = [v for v in frame_ct.values()]

# kde = gaussian_kde(f_bins, bw_method='silverman')
# frame_ct2 = frame_ct.copy()
# for k, v in frame_ct.items():
#     frame_ct[k] = kde.evaluate(v)
#     frame_ct2[k] = v/total
# sum_ = 0
# for v in frame_ct.values:
#     sum_ += v
# print(sum_)
# sum_ = 0
# for v in frame_ct2.values:
#     sum_ += v
# print(sum_)
# print(frame_ct)
# print(frame_ct2)