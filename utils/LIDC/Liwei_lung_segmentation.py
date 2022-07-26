from skimage.segmentation import clear_border
import cv2
import cc3d
import numpy as np
def lung_segmentation(ct_scans):
    binary = ct_scans < -400
    for i, b in enumerate(binary): 
        cleared = clear_border(b)
        Contours_mask = np.zeros(cleared.shape)
        ret, thresh = cv2.threshold(cleared.astype('uint8'), 0, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            Area = [[cv2.contourArea(contour), contour] for contour in contours]
            Area = sorted(Area, key=lambda x: x[0], reverse=True)
            contours = np.array(Area)[:, 1][:2] if len(Area) > 1 else  np.array(Area)[:, 1]
            cv2.drawContours(Contours_mask, contours, -1, 255, -1)
            binary[i] = Contours_mask > 0
        else:
            binary[i] = cleared
    
    labels_out, N = cc3d.largest_k(binary, k=2, connectivity=26, delta=0, return_N=True,)

    if (labels_out == 1).sum()/(labels_out == 2).sum() < 0.1:
        binary *= (labels_out == 2)
    elif (labels_out == 1).sum()/(labels_out == 2).sum() > 10:
        binary *= (labels_out == 1)
    else:
        binary *= (labels_out > 0)

    binary = binary.astype('int')
    return binary