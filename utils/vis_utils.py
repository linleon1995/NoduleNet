import matplotlib.pyplot as plt
import cv2
import numpy as np


def raw_preprocess(img, lung_segment=False, norm=True, change_channel=True, output_dtype=np.int32):
    if norm:
        dtype = img.dtype
        img = np.float32(img) # avoid overflow
        if np.max(img)==np.min(img):
            img = np.zeros_like(img)
        else:
            img = 255*((img-np.min(img))/(np.max(img)-np.min(img)))
        img = np.array(img, dtype)
    
    if change_channel:
        img = np.tile(img[...,np.newaxis], np.append(np.ones(img.ndim, dtype=np.int32), 3))

    img = output_dtype(img)
    return img


def compare_result(image, label, pred, show_mask_size=False, **imshow_params):
    if 'alpha' not in imshow_params: imshow_params['alpha'] = 0.2
    fig, ax = plt.subplots(1,2, figsize=(8,4), tight_layout=True)
    ax[0].imshow(image)
    ax[0].imshow(label, **imshow_params)
    ax[1].imshow(image)
    ax[1].imshow(pred, **imshow_params)
    ax[0].set_title(f'Label (Mask size: {np.sum(label)})' if show_mask_size else 'Label')
    ax[1].set_title(f'Prediction (Mask size: {np.sum(pred)})' if show_mask_size else 'Prediction')
    return fig, ax


def compare_result_enlarge(image, label, pred, show_mask_size=False, **imshow_params):
    crop_range = 30
    if np.sum(label) > 0:
        item = label
    else:
        if np.sum(pred) > 0:
            item = pred
        else:
            item = None
    
    fig, ax = None, None
    if item is not None:
        if image.ndim == 2:
            image = raw_preprocess(image, lung_segment=False, norm=False)
        image = np.uint8(image)
        h, w, c = image.shape
        ys, xs = np.where(item)
        x1, x2 = max(0, min(xs)-crop_range), min(max(xs)+crop_range, min(h,w))
        y1, y2 = max(0, min(ys)-crop_range), min(max(ys)+crop_range, min(h,w))
        bbox_size = np.max([np.abs(x1-x2), np.abs(y1-y2)])

        image = cv2.resize(image[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)

        fig, ax = compare_result(image, label, pred, show_mask_size, **imshow_params)
    return fig, ax


def cv2_imshow(img, save_path=None):
    # pass
    cv2.imshow('My Image', img)
    cv2.imwrite(save_path if save_path else 'sample.png', img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
