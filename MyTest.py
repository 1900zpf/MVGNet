import os
import torch
import argparse
import numpy as np
from scipy import misc
import imageio

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.hub import load_state_dict_from_url

from utils.dataset import test_dataset as EvalDataset
from Net.MyNet_v10_xr_cnn import MyNet as Network


def evaluator(model, val_root, map_save_path, edge_save_path, trainsize=352):
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)

    model.eval()
    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, gt_tensor, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()

            output_sum = model(image)
            output = F.upsample(output_sum, size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            # cv2.imwrite(map_save_path + name, output)
            imageio.imsave(map_save_path + name, output)
            # misc.imsave(map_save_path + name, output)
            print('>>> saving prediction at: {}'.format(map_save_path + name))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MyNet-P2T-large')
    parser.add_argument('--snap_path', type=str, default='./Checkpoints/P2T-l-v10Net_416_CAMO_50epoch_xr_CNN/Net_epoch_best.pth',
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='train use gpu')
    opt = parser.parse_args()
 
    txt_save_path = './Result/Test_result/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')

    cudnn.benchmark = True
    if opt.model == 'MyNet-PVTv2-B2':
        model = Network(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-PVTv2-B1':
        model = Network(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-PVTv2-B5':
        model = Network(channel=64, arc='PVTv2-B5', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-P2T-large':
        model = Network(channel=64, arc='P2T-large', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-P2T-base':
        model = Network(channel=64, arc='P2T-base', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-P2T-small':
        model = Network(channel=64, arc='P2T-small', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-P2T-tiny':
        model = Network(channel=48, arc='P2T-tiny', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    else:
        raise Exception("Invalid Model Symbol: {}".format(opt.model))

    model.load_state_dict(torch.load(opt.snap_path))
    model.eval()

    for data_name in ['CAMO', 'COD10K', 'NC4K','CHAMELEON']:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)

        edge_save_path = txt_save_path + "{}/".format(data_name) + 'edge/'
        # os.makedirs(edge_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root='./Dataset/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            edge_save_path=edge_save_path,
            trainsize=416)
