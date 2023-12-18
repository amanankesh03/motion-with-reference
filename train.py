import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)
import argparse
import numpy as np
import os
import torch
from torch.utils import data
from dataset import Vox256, Taichi, TED, VideoDataset
import torchvision
import torchvision.transforms as transforms
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist  
from tqdm import tqdm
from networks.landmarks import PIPNet

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

loss_book = open('./exps/exp15nov2023.txt','a')

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data

    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)


def write_loss(i, vgg_loss, l1_loss, g_loss, lip_l1, vgg_lip_loss, total_gen_loss, writer, d_loss=0.0, d_every_step = 1):
    # print(f'd_every_step {d_every_step}')
    if i%d_every_step == 0:
        writer.add_scalar('dis_loss', d_loss.item(), i)

    writer.add_scalar('vgg_loss', vgg_loss.item(), i)
    writer.add_scalar('l1_loss', l1_loss.item(), i)
    writer.add_scalar('gen_loss', g_loss.item(), i)
    writer.add_scalar('vgg_lip_loss', vgg_lip_loss.item(), i)
    writer.add_scalar('lip_l1_loss', lip_l1.item(), i)
    writer.add_scalar('Total gen loss', total_gen_loss.item(), i)

    if i%d_every_step == 0:
        loss_book.write(f'vgg:{vgg_loss.item()}, l1:{l1_loss.item()}, gen_loss:{g_loss.item()}, dis_loss:{d_loss.item()}, lip_l1_loss:{lip_l1.item()}, vgg_lip_loss:{vgg_lip_loss.item()}, total_gen_loss:{total_gen_loss}')
    else:
        loss_book.write(f'vgg:{vgg_loss.item()}, l1:{l1_loss.item()}, gen_loss:{g_loss.item()}, lip_l1_loss:{lip_l1.item()}, vgg_lip_loss:{vgg_lip_loss.item()}, total_gen_loss:{total_gen_loss}')
    
    loss_book.write('\n')     
    writer.flush()


def log_loss(i, loss_dict, writer):
    # print(f'\n loss_dict : {loss_dict} \n')
    for name, loss in loss_dict.items():
        writer.add_scalar(name, loss, i)

    loss_book.write(str(loss_dict))
    loss_book.write('\n')     

    writer.flush()


def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    # init distributed computing
    # ddp_setup(args, rank, world_size)
    world_size = world_size
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{str(rank)}")

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )
    print(args.dataset)
    loader2 = None
    if args.dataset == 'ted':
        dataset = TED(args.data_dirs, 'train', transform, True)
        dataset_test = TED('test', transform)
    elif args.dataset == 'vox':
        dataset = Vox256('train', transform, False)
        # dataset_test = Vox256('test', transform)
    elif args.dataset == 'taichi':
        dataset = Taichi('train', transform, True)
        dataset_test = Taichi('test', transform)
    elif args.dataset == 'videoframes':
        transform = torchvision.transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        dataset = VideoDataset(args.data_dirs, transform, False, args.batch_size,
                               ref_img=True, device=f'cuda:{rank}', lms_device=f'cuda:{args.lms_device}',
                               lms_dir=args.lms_dir)
        # print(f'dataset : {dataset}')
    elif args.dataset == 'videoframested':
        transform = torchvision.transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        dataset = VideoDataset(args.data_dirs, transform, False, args.batch_size,
                               ref_img=True, device=f'cuda:{rank}', lms_device=f'cuda:{args.lms_device}',
                               lms_dir=args.lms_dir)
        transform2 = torchvision.transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        dataset2 = Vox256(args.data_dirs2, 'train', transform2, False, True)
        loader2 = data.DataLoader(
            dataset2,
            num_workers=8,
            batch_size=args.batch_size // world_size,
            # sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
            pin_memory=True,
            drop_last=False,
        )
        print('Initialized custom loader2')
    else:
        raise NotImplementedError

    if args.dataset not in ['videoframes', 'videoframested']:
        loader = data.DataLoader(
            dataset,
            num_workers=8,
            batch_size=args.batch_size // world_size,
            #sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
            pin_memory=True,
            drop_last=False,
        )
    else:
        loader = data.DataLoader(dataset, num_workers=8, batch_size=None, pin_memory=True)
    '''loader_test = data.DataLoader(
        dataset_test,
        num_workers=8,
        batch_size=4,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False),
        pin_memory=True,
        drop_last=False,
    )'''

    loader = sample_data(loader)
    if loader2 is not None:
        loader2 = sample_data(loader2)
    # loader_test = sample_data(loader_test)
    
    # img_source, img_target, img_ref, lip_boxes = next(loader)

    # print(fadkjfdalfd;fdflkadf)

    # Trainer
    print('==> initializing trainer')
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args)
        print('==> resume from iteration %d' % (args.start_iter))

    print('==> training')
    pbar = tqdm(total=int(args.iter))
    for idx in range(args.iter):
        # if idx == 0:
        #     print('FIRST STEP')
        i = idx + args.start_iter

        # laoding data
        if loader2 is not None:
            random_number = np.random.choice([0, 1])
            if random_number == 0:
                print('using loader1')
                img_source, img_target, img_ref, lip_boxes = next(loader)
            else:
                print('using loader2')
                img_source, img_target, img_ref, lip_boxes = next(loader2)
        else:
            img_source, img_target, img_ref, lip_boxes = next(loader)
        img_source = img_source.to(rank, non_blocking=True)
        img_target = img_target.to(rank, non_blocking=True)
        img_ref = img_ref.to(rank, non_blocking=True)

        # update generator
        loss_dict, img_recon = trainer.gen_update(img_source, img_target, img_ref, lip_boxes)

        # update discriminator
        if idx%args.d_every_step == 0:
            # print(f'idx {idx}, {args.d_every_step}')
            gan_d_loss = trainer.dis_update(img_target, img_recon)
            loss_dict['gan_d_loss'] = gan_d_loss.item()

        if rank == 1:
            log_loss(idx, loss_dict, writer)

           # display
        if i % args.display_freq == 0:# and rank:
            print(loss_dict)

            # if rank:
            img_test_source, img_test_target, img_test_ref, lip_boxes = next(loader)
            img_test_source = img_test_source.to(rank, non_blocking=True)
            img_test_target = img_test_target.to(rank, non_blocking=True)
            img_test_ref = img_test_ref.to(rank, non_blocking=True)

            img_recon, img_source_ref, img_src_ref = trainer.sample(img_test_source, img_test_target, img_test_ref)
            
            display_img(i, img_test_source, 'source', writer)
            display_img(i, img_test_target, 'target', writer)
            display_img(i, img_recon, 'recon', writer)
            display_img(i, img_source_ref, 'source_ref', writer)
            display_img(i, img_test_ref, 'reference', writer)
            display_img(i, img_src_ref, 'src_ref', writer)
            #display_img(i, lips, 'lips', writer)

            sample_dict = {
                'img_recon' : img_recon[0],
                'img_src' : img_test_source[0],
                'img_target' : img_test_target[0],
                'img_ref' : img_test_ref[0],
                'img_src_ref_ref' : img_src_ref[0],
                'img_src_ref' : img_source_ref[0]
            }
            sample = [s for s in sample_dict.values()]
            # for s in sample:
            #     print(s.shape)
            # print(img_test_source.shape)

            sample = torch.stack(sample)
            display_img(i, sample, 'sample', writer)
            writer.flush()

        # save model
        if i % args.save_freq == 0:# and rank == 1:
            trainer.save(i, checkpoint_path)
        pbar.update(1)

    return   
        


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='vox')
    parser.add_argument("--exp_path", type=str, default='./exps/')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--data_dirs", type=str, nargs='+', default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--lms_device", type=int, default=0)
    parser.add_argument("--lms_dir", type=str, default='./gbucket_lms')
    parser.add_argument("--data_dirs2", type=str, nargs='+', default=None)
    parser.add_argument("--d_every_step", type=int, default=1)
    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2

    world_size = 1
    rank = opts.rank
    print('==> training on %d gpus' % n_gpus)
    # mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)
    main(rank, world_size, opts)
