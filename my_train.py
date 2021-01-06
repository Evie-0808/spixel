# import os
# os.environ('CUDA_VISIBLE_DEVICES') = '0, 1'
from mini_batch_loader import *
from chainer import serializers
from MyFCN import *
from Superpix import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
import os
import torch
import tqdm
from pixelwise_a3c import *
from train_util import *
import torchvision.transforms as transforms
import flow_transforms
import datasets
import argparse
from loss import compute_semantic_pos_loss
from loss2 import  compute_fp
from imageio import imread
from imageio import imsave
# _/_/_/ paths _/_/_/
TRAINING_DATA_PATH = "/home/amax/Desktop/newBSDS500/train.txt"
TESTING_DATA_PATH = "/home/amax/Desktop/newBSDS500/test.txt"
IMAGE_DIR_PATH = "/home/amax/Desktop/newBSDS500/train"
SAVE_PATH = "./model_supernet_"

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1  # must be 1
N_EPISODES = 30000
EPISODE_LEN = 5
SNAPSHOT_EPISODES = 3000
TEST_EPISODES = 3000
GAMMA = 0.95  # discount factor

# noise setting
MEAN = 0
SIGMA = 15

N_ACTIONS = 9
MOVE_RANGE = 3  # number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 320

GPU_ID = 0
save_path = ''

def test(agent, epoch):
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = 'inputs/Lena.jpg'
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]

    # may get 4 channel (alpha channel) for some format
    img_ = imread(load_path)[:, :, :3]
    # H, W, _ = img_.shape
    H, W = (320, 320)
    H_, W_ = int(np.ceil(H / 16.) * 16), int(np.ceil(W / 16.) * 16)

    # get spixel id
    n_spixl_h = int(np.floor(H_ / args.downsize))
    n_spixl_w = int(np.floor(W_ / args.downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float)

    n_spixel = int(n_spixl_h * n_spixl_w)

    img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    data = img1.unsqueeze(0).numpy().astype(np.float32)
    ori_img = input_transform(img)

    current_state = State.State((args.batch_size, N_ACTIONS, CROP_SIZE, CROP_SIZE), MOVE_RANGE)
    with chainer.no_backprop_mode():
        input_cuda = chainer.cuda.to_gpu(data)
        pout, vout = agent.model.pi_and_v(input_cuda)
    pout = chainer.cuda.to_cpu(pout.sample().data)
    pout = current_state.to_probin(pout)
    fp0 = spixeIds
    current_state.reset(data, fp0.numpy())  # 载入原图
    reward = np.zeros(data.shape, data.dtype)
    sum_reward = 0
    for t in range(0, EPISODE_LEN):
        prev_fp = current_state.current_fp.copy()
        action = agent.act(current_state.current_fp, current_state.oriimage)
        current_state.step(action, prev_fp)


    agent.stop_episode()
    # compute output
    output = torch.Tensor(current_state.p)


    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(),
                                                    n_spixels=n_spixel, b_enforce_connect=True)


    # save spixel viz
    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    spixl_save_name = os.path.join(save_path, 'spixel_viz', str(epoch) + '_sPixel.png')
    imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))

dataset_names = sorted(name for name in datasets.__all__)
parser = argparse.ArgumentParser(description='PyTorch SpixelFCN Training on BSDS500',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', metavar='DATASET', default='BSD500',  choices=dataset_names,
                    help='dataset type : ' +  ' | '.join(dataset_names))
parser.add_argument('--train_img_height', '-t_imgH', default=208,  type=int, help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default=208, type=int, help='img width')
parser.add_argument('--input_img_height', '-v_imgH', default=320, type=int, help='img height_must be 16*n')  #
parser.add_argument('--input_img_width', '-v_imgW', default=320,  type=int, help='img width must be 16*n')
parser.add_argument('-b', '--batch-size', default=TRAIN_BATCH_SIZE, type=int,   metavar='N', help='mini-batch size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers')
parser.add_argument('--data', metavar='DIR',default='newBSDS500', help='path to input dataset')

parser.add_argument('--downsize', default=16, type=float,help='grid cell size for superpixel training ')
parser.add_argument('--epoch_size', default= 6000,  help='choose any value > 408 to use all the train and val data')
parser.add_argument('--pos_weight', '-p_w', default=0.003, type=float, help='weight of the pos term')
args = parser.parse_args()

def save_pic(input_t, current_state, i, t=0):
    ###################
    # get spixel id
    H, W = (320, 320)
    H_, W_ = int(np.ceil(H / 16.) * 16), int(np.ceil(W / 16.) * 16)
    n_spixl_h = int(np.floor(H_ / args.downsize))
    n_spixl_w = int(np.floor(W_ / args.downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float)

    n_spixel = int(n_spixl_h * n_spixl_w)

    ori_img = input_t[0]

    # assign the spixel map
    output = torch.Tensor(current_state.p)
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(
        torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=input_t.unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1),
                                                    ori_sz_spixel_map.squeeze(),
                                                    n_spixels=n_spixel, b_enforce_connect=True)

    # save spixel viz
    if not os.path.isdir(os.path.join( 'spixel_viz',str(i))):
        os.makedirs(os.path.join( 'spixel_viz', str(i)))
    spixl_save_name = os.path.join( 'spixel_viz',str(i), str(t) + '_sPixel.png')
    spixl_save_name2 = os.path.join('spixel_viz', str(i), str(t) + '_sLabel.png')
    imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))
    imsave(spixl_save_name2, spixel_label_map)
def train(train_loader, current_state, optimizer, episode, init_spixl_map_idx, xy_feat, agent,fout):
    global n_iter, args, intrinsic
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
    epoch_reward = 0
    for i, (input_t, label) in enumerate(tqdm.tqdm(train_loader)):
        # if i == 2:
        #     break
        iteration = i + episode * epoch_size



        # ========== complete data loading ================
        label_1hot = label2one_hot_torch(label, C=50) # set C=50 as SSN does
        input = input_t.numpy()
        LABXY_feat_tensor = build_LABXY_feat(label_1hot, xy_feat).cuda()  # B* (50+2 )* H * W
        LABXY_feat_numpy = LABXY_feat_tensor.cpu().numpy()

        # ========== predict association map ============

        # 设置fp0 1 52 320 320
        fp0 = LABXY_feat_tensor
        current_state.reset(input, fp0.detach().cpu().numpy())  # 载入原图
        reward = np.zeros(input.shape, input.dtype)
        sum_reward = 0
        # 载入初始loss
        loss0, _, _ = compute_semantic_pos_loss(
            fp0, LABXY_feat_tensor,
            pos_weight=args.pos_weight,
            kernel_size=args.downsize)
        current_state.set_prev_loss(loss0.detach().cpu().numpy())
        for t in range(0, EPISODE_LEN):
            previous_loss, previous_fp, previous_oriimage = current_state.old_loss, current_state.current_fp.copy(), current_state.oriimage.copy()
            action = agent.act_and_train(current_state.current_fp, current_state.oriimage, reward, LABXY_feat_numpy)
            current_state.step(action, previous_fp)
            # 计算奖励


            slic_loss_new, loss_sem, loss_pos = compute_semantic_pos_loss(torch.tensor(current_state.current_fp, device='cuda'), LABXY_feat_tensor,
                                                                          pos_weight=args.pos_weight,
                                                                          kernel_size=args.downsize)
            slic_loss_new = slic_loss_new.detach().cpu().numpy()
            print(slic_loss_new)
            reward = previous_loss - slic_loss_new
            sum_reward += reward * np.power(GAMMA, t)
            # 保存本轮loss
            current_state.set_prev_loss(slic_loss_new)
            save_pic(input_t, current_state, i, t)

        agent.stop_episode_and_train(current_state.current_fp, reward, True)
        # print("[{e}|{t}]train total reward {a}".format(e=i, t=epoch_size, a=sum_reward ))
        epoch_reward += sum_reward
        fout.write("train total reward {a}\n".format(a=sum_reward ))
        sys.stdout.flush()


        # if episode % TEST_EPISODES == 0:
        #     # _/_/_/ testing _/_/_/
        #     test(val_loader, agent, fout)

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH + str(episode))


        optimizer.alpha = LEARNING_RATE * ((1 - episode / N_EPISODES) ** 0.9)
    print("[{e}|{t}]train total reward {a}".format(e=episode, t=epoch_size, a=epoch_reward/len(train_loader)))


def main(fout):
    global args

    # transformer
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    val_input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
    ])

    co_transform = flow_transforms.Compose([
        flow_transforms.RandomCrop((args.train_img_height, args.train_img_width)),
        flow_transforms.RandomVerticalFlip(),
        flow_transforms.RandomHorizontalFlip()
    ])

    print("=> loading img pairs from '{}'".format(args.data))
    train_set, val_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        val_transform=val_input_transform,
        target_transform=target_transform,
        co_transform=co_transform
    )
    print('{} samples found, {} train samples and {} val samples '.format(len(val_set) + len(train_set),
                                                                          len(train_set),
                                                                          len(val_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((args.batch_size, N_ACTIONS, CROP_SIZE, CROP_SIZE), MOVE_RANGE)

    # load myfcn model
    model = SuperNet(True, N_ACTIONS)
    # model = MyFcn( N_ACTIONS)
    # _/_/_/ setup _/_/_/

    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()

    # _/_/_/ training _/_/_/

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    # spixelID: superpixel ID for visualization,
    # XY_feat: the coordinate feature for position loss term
    spixelID, XY_feat_stack = init_spixel_grid(args)
    val_spixelID, val_XY_feat_stack = init_spixel_grid(args, b_train=False)

    for episode in range(1, N_EPISODES + 1):
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        train(train_loader, current_state, optimizer, episode, spixelID, XY_feat_stack, agent=agent,fout=fout)
        test(agent, episode)


if __name__ == '__main__':
    try:
        fout = open('log.txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start) / 60))
        print("{s}[h]".format(s=(end - start) / 60 / 60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start) / 60))
        fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
        fout.close()
    except Exception as error:
        print(error.message)
