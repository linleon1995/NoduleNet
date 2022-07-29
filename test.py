from hypothesis import target
import numpy as np
import torch
import os
import traceback
import time
import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
import SimpleITK as sitk
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from net.nodule_net import NoduleNet
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from dataset.mask_reader import MaskReader
from config import config
from utils.visualize import draw_gt, draw_pred, generate_image_anim
from utils.util import dice_score_seperate, get_contours_from_masks, merge_contours, hausdorff_distance
from utils.util import onehot2multi_mask, normalize, pad2factor, load_dicom_image, npy2submission
from utils.util import average_precision
import pandas as pd
# from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation
from evaluationScript.noduleCADEvaluation_new import noduleCADEvaluation
from post.post_processing import simple_post_processor
import cc3d
import cv2
from utils.vis import save_mask_in_3d, visualize
from utils.nodule_to_nrrd import save_nodule_in_nrrd
from inference import crop_boxes2mask_single

plt.rcParams['figure.figsize'] = (24, 16)
plt.switch_backend('agg')
this_module = sys.modules[__name__]
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser()
parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                    help='neural net')
parser.add_argument("--mode", type=str, default='eval',
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default=config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test-set-name", type=str, default=config['test_set_name'],
                    help="path to save the results")


def main(fold):
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()
    # params_eye_L = np.load('weights/params_eye_L.npy').item()
    # params_eye_R = np.load('weights/params_eye_R.npy').item()
    # params_brain_stem = np.load('weights/params_brain_stem.npy').item()

    if args.mode == 'eval':
        data_dir = config['preprocessed_data_dir']
        # TODO:
        # test_set_name = args.test_set_name
        test_set_name = f'split/tmh_old/{fold}_val.csv'
        # test_set_name = f'split/tmh_old/extra_data.csv'

        num_workers = 0
        # initial_checkpoint = args.weight
        net = args.net
        out_dir = args.out_dir
        save_dir = os.path.join(out_dir, f'{fold}_train')
        initial_checkpoint = os.path.join(
            save_dir, 'model', '260.ckpt')

        net = getattr(this_module, net)(config)
        net = net.cuda()

        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            # out_dir = checkpoint['out_dir']
            epoch = checkpoint['epoch']

            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No model weight file specified')
            return

        print('out_dir', out_dir)
        # save_dir = os.path.join(out_dir, 'res', str(epoch))
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # if not os.path.exists(os.path.join(save_dir, 'FROC')):
        #     os.makedirs(os.path.join(save_dir, 'FROC'))
        
        dataset = MaskReader(data_dir, test_set_name, config, mode='eval')
        eval(net, dataset, save_dir)
    else:
        logging.error('Mode %s is not supported' % (args.mode))


def get_pid_tmh_mapping():
    f = rf'/workspace/Dataset/TMH-Nodule/TMH-preprocess/merge'
    dir_list = [name for name in os.listdir(f) if os.path.isdir(os.path.join(f, name))]
    mapping = {}
    for _dir in dir_list:
        file_path = os.path.join(f, _dir, 'raw')
        folder_list = [name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name))]
        pid = folder_list[0][:-4]
        mapping[pid] = _dir
    return mapping


def eval(net, dataset, save_dir=None):
    # TODO:
    out_dir = os.path.split(save_dir)[0]
    fold = os.path.split(dataset.set_name)[1].split('_')[0]
    save_dir = os.path.join(save_dir, 'no_cls_new_260')
    os.makedirs(save_dir, exist_ok=True)
    net.set_mode('eval')
    net.use_mask = True
    # net.use_mask = False
    net.use_rcnn = True
    aps = []
    dices = []
    raw_dir = config['data_dir']
    preprocessed_dir = config['preprocessed_data_dir']
    lung_mask_dir = config['lung_mask_dir']
    img_root = os.path.join(save_dir, 'img')

    print('Total # of eval data %d' % (len(dataset)))
    run_case = 0
    runtime_error_cases = []
    inference_time = []
    post_process_time = []
    classify_time = []
    plot_time = []
    pid_to_tmh_name = get_pid_tmh_mapping()
    for i, (input, truth_bboxes, truth_labels, truth_masks, mask, image) in enumerate(dataset):
        # if i in [0, 1, 10]: continue
        # if i <4: continue
        # if i>2: break
        # if i != 4: continue
        try:
            D, H, W = image.shape
            pid = dataset.filenames[i]
            
            tmh_name = pid_to_tmh_name[pid]

            print('')
            print('[%d] Predicting %s' % (i, pid), image.shape)

            gt_mask = mask.astype(np.uint8)

            with torch.no_grad():
                input = input.cuda().unsqueeze(0)
                try:
                    start = time.time()
                    net.forward(input, truth_bboxes, truth_labels, truth_masks, mask)
                    end = time.time()
                    run_case += 1
                except ValueError:
                # except RuntimeError:
                    print(f'[{i}] CUDA out of memory on case {pid}')
                    runtime_error_cases.append(pid)
                    continue
            
            inference_time.append(end-start)
            rpns = net.rpn_proposals.cpu().numpy()
            detections = net.detections.cpu().numpy()
            ensembles = net.ensemble_proposals.cpu().numpy()
            # print('ensembles', ensembles)
            if len(detections) and net.use_mask:
                crop_boxes = net.crop_boxes
                segments = [torch.sigmoid(m).cpu().numpy() > 0.5 for m in net.mask_probs]
                
                # TODO: The reason we change to cc3d is because some bbox is too big that
                # can cover multiple nodules This is not the behavior we expected for evalution.
                # But need to be find out why this happen and what caused may be made to FROC.
                # This is okay that we use cc3d for segmentation evaluation, but this is not a 
                # robust result as a detection result.
                pred_mask = crop_boxes2mask_single(crop_boxes[:, 1:], segments, input.shape[2:])
                b_pred_mask = np.where(pred_mask>0, 1, 0)
                pred_index = cc3d.connected_components(b_pred_mask, connectivity=26)
                mapping =[0]
                for p_idx in np.unique(pred_index)[1:]:
                    mapping.append(pred_mask[pred_index==p_idx][0])

                pred_mask = pred_mask.astype(np.uint8)
                pred_index = pred_index.astype(np.uint8)

                # print(gt_mask.shape, pred_mask.shape, image.shape)
                # plt.imsave('test_pred.png', pred_mask[362])
                # plt.imsave('test_gt.png', gt_mask[0,362])
                # a = np.where(gt_mask[0])
                # gg = cc3d.connected_components(gt_mask[0], 26)
                # print(np.unique(gg)[1:])
                # print(np.unique(
                #     cc3d.connected_components(pred_mask*gt_mask[0], 26))[1:])
                # print(f'Before process: {np.unique(pred_mask*gt_mask[0])[1:]}')
                
                # #######
                _1SR = False
                RUNLS = False
                nodule_cls = False
                if _1SR or RUNLS or nodule_cls:
                    lung_mask_vol = np.load(
                        os.path.join(lung_mask_dir, f'{pid}_lung_mask.npy'))
                    lung_mask_vol = pad2factor(lung_mask_vol)
                    # NC_ckpt = os.path.join(
                    #     out_dir, 
                    #     'run_060', fold, 'ckpt_best.pth')
                    NC_ckpt = os.path.join(
                        out_dir, 
                        'run_047', fold, 'ckpt_best.pth')
                    pred_index, post_time, cls_time = simple_post_processor(
                            input.cpu().detach().numpy()[0, 0], 
                            gt_mask[0], 
                            pred_index,
                            lung_mask_vol,
                            pid,
                            FP_reducer_checkpoint=NC_ckpt,
                            _1SR=_1SR,
                            RUNLS=RUNLS,
                            nodule_cls=nodule_cls)
                    post_process_time.append(post_time)
                    classify_time.append(cls_time)
                    keep_indices =  np.unique(pred_index)[1:]
                    keep_mask_labels = np.unique(np.array(mapping, 'int')[keep_indices])
                    ensembles = ensembles[keep_mask_labels-1]
                    pred_mask = cc3d.connected_components(pred_index, connectivity=26)
                    print(f'After process: {np.unique(pred_mask*gt_mask[0])[1:]}')

                # compute average precisions
                # TODO:
                # print(np.unique(pred_index), len(np.unique(pred_index)))
                # print(np.unique(pred_mask), len(np.unique(pred_mask)))
                ap, dice = average_precision(gt_mask, pred_mask)
                # TODO: 
                aps.append(ap)
                dices.extend(dice.tolist())
                print(ap)
                print('AP: ', np.mean(ap))
                print('DICE: ', dice)
                print
            else:
                pred_mask = np.zeros((input[0].shape))
            
            np.save(os.path.join(save_dir, '%s.npy' % (pid)), pred_mask)
            row_df = []
            # # TODO: 

            b_pred_mask = np.int32(pred_index)
            b_gt_mask = np.int32(gt_mask[0])
            # b_pred_mask = np.int32(np.where(pred_index>0, 1, 0))
            # b_gt_mask = np.int32(np.where(gt_mask[0]>0, 1, 0))
            fig, ax = plt.subplots(1,1)
            # TODO: 
            resample_ct = np.load(
                os.path.join(config['ori_data_dir'], f'{pid}_img.npy'))
            lung_box = np.load(
                os.path.join(preprocessed_dir, f'{pid}_lung_box.npy'))
            resample_mask = np.zeros(resample_ct.shape, np.int32)
            resample_pred = np.zeros(resample_ct.shape, np.int32)
            zmin, zmax = lung_box[0]
            ymin, ymax = lung_box[1]
            xmin, xmax = lung_box[2]
            # print(resample_ct.shape, b_gt_mask.shape, b_pred_mask.shape,
            # image.shape, zmin, zmax, ymin, ymax, xmin, xmax)

            ori_img_shape = image.shape
            resample_mask = \
                b_gt_mask[:ori_img_shape[0], :ori_img_shape[1], :ori_img_shape[2]]
            resample_pred = \
                b_pred_mask[:ori_img_shape[0], :ori_img_shape[1], :ori_img_shape[2]]
            # resample_mask[zmin:zmax, ymin:ymax, xmin:xmax] = \
            #     b_gt_mask[:ori_img_shape[0], :ori_img_shape[1], :ori_img_shape[2]]
            # resample_pred[zmin:zmax, ymin:ymax, xmin:xmax] = \
            #     b_pred_mask[:ori_img_shape[0], :ori_img_shape[1], :ori_img_shape[2]]
            ############
            print('Saving images and pred mask')
            nodule_visualize(save_dir, pid, resample_ct, resample_mask, resample_pred, 
                             preprocessed_dir, nodule_probs=None, save_all_images=False)
            ############
            nrrd_path = os.path.join(save_dir, 'images', pid, 'nrrd')
            direction = np.eye(3)
            origin = np.load(os.path.join(preprocessed_dir, f'{pid}_origin.npy'))
            spacing = np.ones(3)
            # spacing = np.load(os.path.join(preprocessed_dir, f'{pid}_spacing.npy'))
            # pred_without_target = resample_pred.copy()
            # target_labels = np.unique(pred_without_target*resample_mask)[1:]
            # for target_label in target_labels:
            #     pred_without_target = np.where(pred_without_target==target_label, 0, pred_without_target)
            save_nodule_in_nrrd(resample_ct, resample_pred, direction, origin, spacing, nrrd_path, pid)
            ############

            print('rpn', rpns.shape)
            print('detection', detections.shape)
            print('ensemble', ensembles.shape)

            # print('rpn', rpns.max())
            # print('detection', detections.max())
            # print('ensemble', ensembles.max())


            if len(rpns):
                rpns = rpns[:, 1:]
                np.save(os.path.join(save_dir, '%s_rpns.npy' % (pid)), rpns)

            if len(detections):
                detections = detections[:, 1:-1]
                np.save(os.path.join(save_dir, '%s_rcnns.npy' % (pid)), detections)

            if len(ensembles):
                ensembles = ensembles[:, 1:]
                np.save(os.path.join(save_dir, '%s_ensembles.npy' % (pid)), ensembles)


            # Clear gpu memory
            del input, truth_bboxes, truth_labels, truth_masks, mask, image, pred_mask#, gt_mask, gt_img, pred_img, full, score
            torch.cuda.empty_cache()

        except Exception as e:
            del input, truth_bboxes, truth_labels, truth_masks, mask, image,
            torch.cuda.empty_cache()
            traceback.print_exc()
                        
            print
            return
    
    # Generate prediction csv for the use of performning FROC analysis
    # Save both rpn and rcnn results
    rpn_res = []
    rcnn_res = []
    ensemble_res = []
    for pid in dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % (pid))):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % (pid)))
            rpns = rpns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_rcnns.npy' % (pid))):
            rcnns = np.load(os.path.join(save_dir, '%s_rcnns.npy' % (pid)))
            rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_ensembles.npy' % (pid))):
            ensembles = np.load(os.path.join(save_dir, '%s_ensembles.npy' % (pid)))
            ensembles = ensembles[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(ensembles))
            ensemble_res.append(np.concatenate([names, ensembles], axis=1))
    
    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    os.makedirs(eval_dir, exist_ok=True)
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    
    df = pd.DataFrame(rpn_res, columns=col_names)
    df.to_csv(rpn_submission_path, index=False)

    df = pd.DataFrame(rcnn_res, columns=col_names)
    df.to_csv(rcnn_submission_path, index=False)

    df = pd.DataFrame(ensemble_res, columns=col_names)
    df.to_csv(ensemble_submission_path, index=False)

    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
        os.makedirs(os.path.join(eval_dir, 'ensemble'))

    # noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    # 'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    # dataset.set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    # noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    # 'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    # dataset.set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    # noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    # 'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    # dataset.set_name, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))
        
    noduleCADEvaluation('evaluationScript/annotations/TMH_new/annotations.csv',
    'evaluationScript/annotations/TMH_old/annotation_excluded.csv',
    dataset.set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    noduleCADEvaluation('evaluationScript/annotations/TMH_new/annotations.csv',
    'evaluationScript/annotations/TMH_old/annotation_excluded.csv',
    dataset.set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    noduleCADEvaluation('evaluationScript/annotations/TMH_new/annotations.csv',
    'evaluationScript/annotations/TMH_old/annotation_excluded.csv',
    dataset.set_name, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))
    # print

    # TODO: save in txt
    print(f'Acutal running cases {run_case}')
    print(f'Out of memory case {len(runtime_error_cases)}')
    print(runtime_error_cases)
    aps = np.array(aps)
    dices = np.array(dices)
    print(60*'-')
    print('mAP: ', np.mean(aps, 0))
    print('mean dice:%.4f(%.4f)' % (np.mean(dices), np.std(dices)))
    print('mean dice (exclude fn):%.4f(%.4f)' % (np.mean(dices[dices != 0]), np.std(dices[dices != 0])))
    if len(inference_time) > 0:
        mean_inference_time = sum(inference_time)/len(inference_time)
        print(f'Average inference time {mean_inference_time} sec. in {len(inference_time)} times')
    if len(post_process_time) > 0:
        mean_post_time = sum(post_process_time)/len(post_process_time)
        print(f'Average post time {mean_post_time} sec. in {len(post_process_time)} times')
    if len(classify_time) > 0:
        mean_classify_time = sum(classify_time)/len(classify_time)
        print(f'Average classify time {mean_classify_time} sec. in {len(classify_time)} times')

    with open(os.path.join(eval_dir, f'{fold}_result.txt'), 'w+') as fw:
        fw.write(f'mAP: {np.mean(aps, 0)}\n')
        fw.write(f'mean dice: {dices.shape} {np.mean(dices):.4f} ({np.std(dices):.4f})\n')
        fw.write(f'mean dice (exclude fn): {dices[dices != 0].shape} {np.mean(dices[dices != 0]):.4f} ({np.std(dices[dices != 0]):.4f})\n')
        if len(inference_time) > 0:
            mean_inference_time = sum(inference_time)/len(inference_time)
            fw.write(f'Average inference time {mean_inference_time} sec. in {len(inference_time)} times\n')
        if len(post_process_time) > 0:
            mean_post_time = sum(post_process_time)/len(post_process_time)
            fw.write(f'Average post time {mean_post_time} sec. in {len(post_process_time)} times\n')
        if len(classify_time) > 0:
            mean_classify_time = sum(classify_time)/len(classify_time)
            fw.write(f'Average classify time {mean_classify_time} sec. in {len(classify_time)} times')


def nodule_visualize(save_path, pid, vol, target_vol_category, pred_vol_category, 
                     preprocessed_dir, nodule_probs=None, save_all_images=False):

    # b_pred_mask = np.where(pred_vol_category>0, 1, 0)
    # b_gt_mask = np.where(target_vol_category>0, 1, 0)
    # fig, ax = plt.subplots(1,1)
    # resample_ct = np.load(
    #     os.path.join(preprocessed_dir, f'{pid}_img.npy'))
    # lung_box = np.load(
    #     os.path.join(preprocessed_dir, f'{pid}_lung_box.npy'))
    # resample_mask = np.zeros(resample_ct.shape)
    # resample_pred = np.zeros(resample_ct.shape)
    # zmin, zmax = lung_box[0]
    # ymin, ymax = lung_box[1]
    # xmin, xmax = lung_box[2]
    # # print(resample_ct.shape, b_gt_mask.shape, b_pred_mask.shape,
    # # vol.shape, zmin, zmax, ymin, ymax, xmin, xmax)
    
    # ori_img_shape = vol.shape
    # resample_mask[zmin:zmax, ymin:ymax, xmin:xmax] = \
    #     b_gt_mask[:ori_img_shape[0], :ori_img_shape[1], :ori_img_shape[2]]
    # resample_pred[zmin:zmax, ymin:ymax, xmin:xmax] = \
    #     b_pred_mask[:ori_img_shape[0], :ori_img_shape[1], :ori_img_shape[2]]
        
    origin_save_path = os.path.join(save_path, 'images', pid, 'origin')
    enlarge_save_path = os.path.join(save_path, 'images', pid, 'enlarge')
    _3d_save_path = os.path.join(save_path, 'images', pid, '3d')
    for path in [origin_save_path, enlarge_save_path, _3d_save_path]:
        os.makedirs(path, exist_ok=True)

    # TODO: only for binary currently, because it directly select the 1st class prob for visualize
    pred_vol_category = np.asarray(pred_vol_category, dtype=np.uint8)
    target_vol_category = np.asarray(target_vol_category, dtype=np.uint8)
    vis_vol, vis_indices, vis_crops = visualize(
        vol, pred_vol_category, target_vol_category, nodule_probs)
    if save_all_images:
        vis_indices = np.arange(vis_vol.shape[0])

    for vis_idx in vis_indices:
        # plt.savefig(vis_vol[vis_idx])
        cv2.imwrite(os.path.join(origin_save_path, f'vis-{pid}-{vis_idx}.png'), vis_vol[vis_idx])
        if vis_idx in vis_crops:
            for crop_idx, vis_crop in enumerate(vis_crops[vis_idx]):
                cv2.imwrite(os.path.join(enlarge_save_path, f'vis-{pid}-{vis_idx}-crop{crop_idx:03d}.png'), vis_crop)

    temp = np.where(target_vol_category+pred_vol_category>0, 1, 0)
    if np.sum(temp) > 0:
        zs_c, ys_c, xs_c = np.where(temp)
        crop_range = {'z': (np.min(zs_c), np.max(zs_c)), 'y': (np.min(ys_c), np.max(ys_c)), 'x': (np.min(xs_c), np.max(xs_c))}
        if crop_range['z'][1]-crop_range['z'][0] > 2 and \
        crop_range['y'][1]-crop_range['y'][0] > 2 and \
        crop_range['x'][1]-crop_range['x'][0] > 2:
            save_mask_in_3d(target_vol_category, 
                            save_path1=os.path.join(_3d_save_path, f'{pid}-raw-mask.png'),
                            save_path2=os.path.join(_3d_save_path, f'{pid}-preprocess-mask.png'), 
                            crop_range=crop_range)
            save_mask_in_3d(pred_vol_category,
                            save_path1=os.path.join(_3d_save_path, f'{pid}-raw-pred.png'),
                            save_path2=os.path.join(_3d_save_path, f'{pid}-preprocess-pred.png'),
                            crop_range=crop_range)

def eval_single(net, input):
    with torch.no_grad():
        input = input.cuda().unsqueeze(0)
        logits = net.forward(input)
        logits = logits[0]
    
    masks = logits.cpu().data.numpy()
    masks = (masks > 0.5).astype(np.int32)
    return masks
 

if __name__ == '__main__':
    # for fold in range(4,5):
    #     main(fold)
    fold = 4
    main(fold)

