import os
import torch
# import shutil
# import numpy as np
# from PIL import Image
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn

import eval.python.metrics as Measure
import prettytable as pt
# import sys
import argparse
import cv2
from tqdm import tqdm


def get_competitors(root):
    for model_name in os.listdir(root):
        print('\'{}\''.format(model_name), end=', ')


def evaluator(gt_pth_lst, pred_pth_lst):
    # define measures
    FM = Measure.Fmeasure()
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    assert len(gt_pth_lst) == len(pred_pth_lst)

    # evaluator
    with torch.no_grad():
        # for idx, gt_pth in enumerate(gt_pth_lst):
        for idx in tqdm(range(len(gt_pth_lst))):
            gt_pth = gt_pth_lst[idx]
            pred_pth = pred_pth_lst[idx]

            assert os.path.isfile(gt_pth) and os.path.isfile(pred_pth)
            pred_ary = cv2.imread(pred_pth, cv2.IMREAD_GRAYSCALE)

            gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)
            if gt_ary.shape != pred_ary.shape:
                h,w = gt_ary.shape
                pred_ary = cv2.resize(pred_ary, (w,h))
            # print(pred_ary, gt_ary)
            FM.step(pred=pred_ary, gt=gt_ary)
            WFM.step(pred=pred_ary, gt=gt_ary)
            SM.step(pred=pred_ary, gt=gt_ary)
            EM.step(pred=pred_ary, gt=gt_ary)
            MAE.step(pred=pred_ary, gt=gt_ary)

        fm = FM.get_results()['fm']
        wfm = WFM.get_results()['wfm']
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']

    return fm, wfm, sm, em, mae


def eval_all(opt, txt_save_path):
    # evaluation for whole dataset
    for _data_name in opt.data_lst:
        print('#' * 20, _data_name, '#' * 20)
        filename = os.path.join(txt_save_path, '{}_eval.txt'.format(_data_name))
        with open(filename, 'w+') as file_to_write:
            tb = pt.PrettyTable()
            tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm",
                              "meanFm", "maxFm"]
            for _model_name in opt.model_lst:
                print('#' * 10, _model_name, '#' * 10)
                gt_src = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_src = os.path.join(opt.pred_root, _model_name, _data_name)

                # get the valid filename list
                img_name_lst = os.listdir(gt_src)

                fm, wfm, sm, em, mae = evaluator(
                    gt_pth_lst=[os.path.join(gt_src, i) for i in img_name_lst],
                    pred_pth_lst=[os.path.join(pred_src, i) for i in img_name_lst]
                )
                tb.add_row([_data_name, _model_name, sm.round(3), wfm.round(3), mae.round(3), em['adp'].round(3),
                            em['curve'].mean().round(3), em['curve'].max().round(3), fm['adp'].round(3),
                            fm['curve'].mean().round(3), fm['curve'].max().round(3)])
            print(tb)
            file_to_write.write(str(tb))
            file_to_write.close()


def eval_super_class(opt):
    # evaluation for super-class in COD10K
    # [{'Aquatic': 474}, {'Flying': 714}, {'Other': 15}, {'Amphibian': 124}, {'Terrestrial': 699}]
    _super_cls_lst = ['Aquatic', 'Flying', 'Amphibian', 'Terrestrial', 'Other']

    # evaluation for super class of COD10K
    _data_name = 'COD10K'
    print('#' * 20, _data_name, '#' * 20)
    tb = pt.PrettyTable()
    tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm",
                      "meanFm", "maxFm"]
    for _model_name in opt.model_lst:
        for _super_cls in _super_cls_lst:
            fm, wfm, sm, em, mae = evaluator(
                gt_pth_lst=[i for i in os.listdir(os.path.join(opt.gt_root, _data_name, 'GT')) if _super_cls in i],
                pred_pth_lst=[i for i in os.listdir(os.path.join(opt.pred_root, _model_name, _data_name)) if
                              _super_cls in i]
            )
            tb.add_row([_super_cls, _model_name, sm.round(3), wfm.round(3), mae.round(3), em['adp'].round(3),
                        em['curve'].mean().round(3), em['curve'].max().round(3), fm['adp'].round(3),
                        fm['curve'].mean().round(3), fm['curve'].max().round(3)])
        print(tb)


def eval_sub_class(opt):
    # evaluation for super-class in COD10K
    # [{'Aquatic': 474}, {'Flying': 714}, {'Other': 15}, {'Amphibian': 124}, {'Terrestrial': 699}]
    _sub_cls_lst = []

    # evaluation for super class of COD10K
    _data_name = 'COD10K'
    print('#' * 20, _data_name, '#' * 20)
    tb = pt.PrettyTable()
    tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm",
                      "meanFm", "maxFm"]
    for _model_name in opt.model_lst:
        for _sub_cls in _sub_cls_lst:
            fm, wfm, sm, em, mae = evaluator(
                gt_pth_lst=[i for i in os.listdir(os.path.join(opt.gt_root, _data_name, 'GT')) if _sub_cls in i],
                pred_pth_lst=[i for i in os.listdir(os.path.join(opt.pred_root, _model_name, _data_name)) if
                              _sub_cls in i]
            )
            tb.add_row([_sub_cls, _model_name, sm.round(3), wfm.round(3), mae.round(3), em['adp'].round(3),
                        em['curve'].mean().round(3), em['curve'].max().round(3), fm['adp'].round(3),
                        fm['curve'].mean().round(3), fm['curve'].max().round(3)])
        print(tb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default='./Dataset/TestDataset')
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        # default='/media/server/HDD1/zpf/otherNet/') # 预测图的根路径
        default='./Result/Test_result/') # 预测图的根路径
    parser.add_argument(
        '--gpu_id', type=str, help='gpu device',
        default='0')
    parser.add_argument(
        '--data_lst', type=list, help='test dataset',
        default=['CAMO', 'COD10K', 'NC4K','CHAMELEON'],
        # default=['CHAMELEON'],
        choices=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'])

    '''
    2020-CVPR-SINet-4          ok
    2021-CVPR-UJSC-4           ok
    2022-IJCAI-BGNet-4         ok
    2021-IJCAI-C2FNet-4        ok
    2022-TPAMI-ICON-4          ok
    2021-CVPR-MGL-R-4          ok
    2022-AAAI-BASNet-4         ok
    2022-TPAMI-SINetV2-4       ok
    2021-CVPR-PFNet-4          ok
    2022-CVPR-FDNet-4          ok
    2023-MIR-DGNet-4           ok
    2021-CVPR-SLSR-4           ok
    2022-CVPR-SegMaR-4         ok
    2021-ICCV-UGTR-4           ok
    2022-CVPR-ZoomNet-4        ok
    '''
    ##  2021-AAAI-TINet-333333333

    parser.add_argument(
        '--model_lst', type=list, help='candidate competitors',
        default=['P2T-l-v10Net_416_CAMO_50epoch_xr_CNN']) # 要测试的文件夹名字  xr_0421_no_MFFM_initdata
    parser.add_argument(
        '--txt_name', type=str, help='candidate competitors',
        default='P2T-l-v10Net_416_CAMO_50epoch_xr_CNN') # 结果保存为txt文件  txt文件的保存路径
    parser.add_argument(
        '--check_integrity', type=bool, help='whether to check the file integrity',
        default=True)
    parser.add_argument(
        '--eval_type', type=str, help='evaluation type',
        default='eval_all',
        choices=['eval_all', 'eval_super', 'eval_sub'])
    opt = parser.parse_args()

    # txt_save_path = '/media/server/HDD1/zpf/result_save/{}/'.format(opt.txt_name)
    txt_save_path = './Result/txt_result/{}/'.format(opt.txt_name)
    os.makedirs(txt_save_path, exist_ok=True)

    # sys.stdout = Logger(txt_save_path + 'evaluation_results.log')
    # print('>>> current configs:', opt)

    # check the integrity of each candidates
    if opt.check_integrity:
        for _data_name in opt.data_lst:
            for _model_name in opt.model_lst:
                gt_pth = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_pth = os.path.join(opt.pred_root, _model_name, _data_name)
                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                    print('The {} Dataset of {} Model is not matching to the ground-truth'.format(_data_name,
                                                                                                  _model_name))
    else:
        print('>>> skip check the integrity of each candidates')

    if opt.eval_type == 'eval_all':
        eval_all(opt, txt_save_path)
    elif opt.eval_type == 'eval_super':
        eval_super_class(opt, txt_save_path)
    elif opt.eval_type == 'eval_sub':
        eval_sub_class(opt, txt_save_path)
    else:
        raise Exception