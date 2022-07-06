import sys
import os
import logging
import torch
import numpy as np
from dataset.mask_reader import MaskReader
from config import config
from utils.nodule_to_nrrd import seg_nrrd_write
from net.nodule_net import NoduleNet
this_module = sys.modules[__name__]



def inference(test_set_name, model_name, image_in, image_out):
    net = prepare_model(model_name, config)

    data_dir = config['preprocessed_data_dir']
    dataset = MaskReader(data_dir, test_set_name, config, mode='eval')
    data_iter = iter(dataset)
    input_image, truth_bboxes, truth_labels, truth_masks, mask, image = next(data_iter)
    # input_image, truth_bboxes, truth_labels, truth_masks, mask, image = next(data_iter)
    input_image = input_image.cuda().unsqueeze(0)
    logging.debug(f'Loading file from {data_dir}')
    logging.debug(f'Image shape: {input_image.shape}')

    with torch.no_grad():
        net.forward(input_image, truth_bboxes, truth_labels, truth_masks, mask)
    crop_boxes = net.crop_boxes
    segments = [torch.sigmoid(m).cpu().numpy() > 0.5 for m in net.mask_probs]
    pred_mask = crop_boxes2mask_single(crop_boxes[:, 1:], segments, input_image.shape[2:])

    direction = np.eye(3)
    pid = dataset.filenames[0]
    result_file = os.path.join(image_out, 'test.seg.nrrd')
    preprocessed_dir = config['preprocessed_data_dir']
    origin = np.load(os.path.join(preprocessed_dir, f'{pid}_origin.npy'))
    spacing = np.load(os.path.join(preprocessed_dir, f'{pid}_spacing.npy'))
    seg_nrrd_write(result_file, pred_mask, direction, origin, spacing)
            
    # pred_mask_bytes = pred_mask.tobytes()
    # print(pred_mask.shape)
    # print(len(pred_mask_bytes))
    # result_file = 'test.nii.gz'
    # with open(result_file, 'wb') as fw:
    #     fw.write(pred_mask_bytes)

    logging.debug(f'Saving to {result_file}')
    return result_file


def prepare_model(model_name, config):
    net = getattr(this_module, model_name)(config)
    net = net.cuda()
    initial_checkpoint = config['initial_checkpoint']
    checkpoint = torch.load(initial_checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.set_mode('eval')
    net.use_mask = True
    net.use_rcnn = True
    return net



def crop_boxes2mask_single(crop_boxes, masks, img_reso):
    """
    Apply results of mask-rcnn (detections and masks) to mask result.

    crop_boxes: detected bounding boxes [z, y, x, d, h, w, category]
    masks: mask predictions correponding to each one of the detections config['mask_crop_size']
    img_reso: tuple with 3 elements, shape of the image or target resolution of the mask
    """
    D, H, W = img_reso
    mask = np.zeros((D, H, W))
    for i in range(len(crop_boxes)):
        z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]

        cat = int(cat)

        m = masks[i]
        D_c, H_c, W_c = m.shape
        mask[z_start:z_end, y_start:y_end, x_start:x_end][m > 0.5] = i + 1
    
    return mask


if __name__ == '__main__':
    test_set_name = 'split/tmh/0_val.csv'
    model_name = 'NoduleNet'
    result_file = inference(test_set_name, model_name, None, './results')