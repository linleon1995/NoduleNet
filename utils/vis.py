import os
import cv2
import numpy as np
# import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.vis_utils import compare_result, compare_result_enlarge
# from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
import cv2
import cc3d
from post.data_postprocess import VolumePostProcessor
logging.basicConfig(level=logging.INFO)


def show_mask_base(x, y, vis_type='mask', dpi=300, save_path=None):
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    ax.imshow(x, 'gray')
    if vis_type == 'mask':
        ax.imshow(y, alpha=0.2)
    if save_path is not None:
        fig.savefig(save_path)
    else:
        fig.show()
    plt.close(fig)





def plot_image_truth_prediction(x, y, p, rows=12, cols=12, name=None):
    x, y, p = np.squeeze(x), np.squeeze(y), np.squeeze(p>0.5)
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(25*3, 25))

    large_image = np.zeros((rows*x.shape[0], cols*x.shape[1]))
    for b in range(rows*cols):
        large_image[(b//rows)*x.shape[0]:(b//rows+1)*x.shape[0],
                    (b%cols)*x.shape[1]:(b%cols+1)*x.shape[1]] = np.transpose(np.squeeze(x[:, :, b]))
    plt.subplot(1, 3, 1)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1); plt.axis('off')

    large_image = np.zeros((rows*x.shape[0], cols*x.shape[1]))
    for b in range(rows*cols):
        large_image[(b//rows)*y.shape[0]:(b//rows+1)*y.shape[0],
                    (b%cols)*y.shape[1]:(b%cols+1)*y.shape[1]] = np.transpose(np.squeeze(y[:, :, b]))
    plt.subplot(1, 3, 2)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1); plt.axis('off')

    large_image = np.zeros((rows*p.shape[0], cols*p.shape[1]))
    for b in range(rows*cols):
        large_image[(b//rows)*p.shape[0]:(b//rows+1)*p.shape[0],
                    (b%cols)*p.shape[1]:(b%cols+1)*p.shape[1]] = np.transpose(np.squeeze(p[:, :, b]))
    plt.subplot(1, 3, 3)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1); plt.axis('off')

    if name is not None:
        plt.savefig(name)
    else:
        plt.show()   


class ScatterVisualizer():
    def __init__(self, scatter_size=50, cmap=plt.cm.jet, title=''):
        self.fig, self.ax = plt.subplots(1, 1)
        self.scatter_size = scatter_size
        self.cmap = cmap
        self.nodule_sizes = []
        self.nodule_hu = []
        self.scores = []

    def record(self, study):
        nodule_instances = study.nodule_instances
        for nodule_id in nodule_instances:
            nodule = nodule_instances[nodule_id]
            self.nodule_sizes.append(nodule.nodule_size)
            # TODO: nodule.hu+100
            self.nodule_hu.append(nodule.hu+100)
            self.scores.append(nodule.nodule_score['DSC'])
    
    def show_scatter(self, title='', xlabel='', ylabel='', save_path=None):
        self.fig, self.ax = plot_scatter(self.fig, self.ax, x=self.nodule_sizes, y=self.nodule_hu, scores=self.scores, size=self.scatter_size, cmap=self.cmap, quant_steps=10, 
                               tiitle=title, xlabel=xlabel, ylabel=ylabel)

        if save_path is not None:
            self.fig.savefig(save_path)


def plot_scatter(fig, ax, x, y, scores, size, cmap, quant_steps=10, tiitle='', xlabel='', ylabel='', alpha=0.5):
    # TODO: fix the color bar from 0 to 1
    if not isinstance(size, list):
        size = len(x)*[size]

    cmap = plt.get_cmap(cmap.name, quant_steps+1)

    sc = ax.scatter(x, y, c=scores, s=size, cmap=cmap, alpha=alpha)
    fig.colorbar(sc, ax=ax)
    
    ax.set_title(tiitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_yscale('log')
    return fig, ax

def get_color_from_cmap(cmap, output_format='rgb', keep_transparent=False, output_uint8=True):
    # TODO: number of color --> 20 in below should be changed
    # TODO: continuous color?
    if keep_transparent:
        select_idx = 4
    else:
        select_idx = 3

    convert_func = lambda value: np.uint8(255*value) if output_uint8 else value
    if output_format == 'rgb':
        colors = [convert_func(np.array(cmap(idx))[:select_idx]) for idx in range(cmap.N)]
    elif output_format == 'bgr':
        colors = [convert_func(np.array(cmap(idx))[:select_idx])[::-1] for idx in range(cmap.N)]
    return colors


def crop_img(image, crop_parameters, keep_size=True):
    if not isinstance(crop_parameters, list):
        crop_parameters = [crop_parameters]
    
    height, width = image.shape[:2]
    crop_images = []
    for crop_parameter in crop_parameters:
        crop_center, crop_height, crop_width = crop_parameter['center'], crop_parameter['height'], crop_parameter['width']
        y_start, y_end = np.clip(crop_center['y']-crop_height//2, 0, height-1), np.clip(crop_center['y']+crop_height//2, 0, height-1)
        x_start, x_end = np.clip(crop_center['x']-crop_width//2, 0, width-1), np.clip(crop_center['x']+crop_width//2, 0, width-1)

        crop_image = image[y_start:y_end, x_start:x_end]
        if keep_size:
            crop_image = cv2.resize(crop_image, dsize=(width, height))
        crop_images.append(crop_image)

    return crop_images
        
# class SegVisualizer():
#     def __init__():
        
def visualize(input_vol, pred_vol, target_vol, nodule_probs=None, enlarge_crop_size=(128, 128)):
    """
    input_vol: [D, H, W]
    pred_vol: [D, H, W]
    target_vol: [D, H, W]
    """
    # pred_vol = cc3d.connected_components(pred_vol, connectivity=26)
    # target_vol = cc3d.connected_components(target_vol, connectivity=26)
    input_vol = np.tile(input_vol[...,None], (1, 1, 1, 3))
    pred_category = np.unique(pred_vol)[1:]
    zs, ys, xs = np.where(pred_vol)
    pred_zs = np.unique(zs)
    zs, ys, xs = np.where(target_vol)
    target_zs = np.unique(zs)
    total_zs = np.unique(np.concatenate((pred_zs, target_zs)))
    depth, height, width = pred_vol.shape
    
    bgr_colors = get_color_from_cmap(cmap=cm.tab20, output_format='bgr', keep_transparent=False)
    target_colors = bgr_colors[4:5]
    pred_colors = bgr_colors[:4]+bgr_colors[5:14]+bgr_colors[16:] # Filter out not obvious color 
    draw_vol = input_vol.copy()
    center_shift = 20

    # Find prediction contours
    total_contours = {}
    for nodule_id in pred_category:
        nodule_vol = np.uint8(pred_vol==nodule_id)
        # nodule_vol = np.tile(nodule_vol[...,np.newaxis], (1,1,1,3))
        zs, ys, xs = np.where(nodule_vol)
        total_contours[nodule_id] = {}
        for z_idx in range(np.min(zs), np.max(zs)+1):
            pred = nodule_vol[z_idx]
            contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            total_contours[nodule_id][z_idx] = contours

    # Draw prediction contours
    for nodule_id in total_contours:
        contour_pair = total_contours[nodule_id]
        color = pred_colors[nodule_id%len(pred_colors)].tolist()
        if nodule_probs is not None:
            prob = nodule_probs[nodule_id][1]
        for slice_idx in contour_pair:
            contours = contour_pair[slice_idx]
            draw_img = draw_vol[slice_idx]
            # TODO: no contour exist condition
            # print(contours, 'xxx')
            contour_center = np.int32(np.mean(contours[0], axis=0)[0]) + center_shift
            contour_center = np.clip(contour_center, 0, min(height, width))

            for contour_idx in range(len(contours)):
                draw_img = cv2.drawContours(draw_img, contours, contour_idx, color, 1)
            
            # probability text
            if nodule_probs is not None:
                draw_img = cv2.putText(draw_img, f'{prob:.4f}', contour_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            draw_vol[slice_idx] = draw_img

    # Find and draw target contours
    target_vol = np.uint8(target_vol)
    for z_idx in target_zs:
        target = target_vol[z_idx]
        contours, hierarchy = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw_vol[z_idx] = cv2.drawContours(draw_vol[z_idx], contours, -1, target_colors[0].tolist(), 1)

    crop_draw_imgs = {}
    candidate_vol = cc3d.connected_components(np.where(pred_vol+target_vol>0, 1, 0), connectivity=26)
    for z_idx in total_zs:
        candidate_slice = candidate_vol[z_idx]
        category = np.unique(candidate_slice).tolist()
        category.remove(0)
        for ccl_id in category:
            ys, xs = np.where(candidate_slice==ccl_id)
            center = {'y': np.int16(np.mean(ys)), 'x': np.int16(np.mean(xs))}
            crop_parameter = {'center': center, 'height': enlarge_crop_size[0], 'width': enlarge_crop_size[1]}
            crops = crop_img(draw_vol[z_idx], [crop_parameter])
            if z_idx in crop_draw_imgs:
                crop_draw_imgs[z_idx].extend(crops)
            else:
                crop_draw_imgs[z_idx] = crops

    return draw_vol, total_zs, crop_draw_imgs


def save_mask_in_3d(volume, save_path1, save_path2, enlarge=True, crop_range=None):
    # TODO: pass if cropped volume smaller than 2x2x2
    if crop_range is not None:
        z_slice = np.arange(crop_range['z'][0], crop_range['z'][1])
        y_slice = np.arange(crop_range['y'][0], crop_range['y'][1])
        x_slice = np.arange(crop_range['x'][0], crop_range['x'][1])
    else:
        if enlarge:
            zs, ys, xs = np.where(volume)
            z_slice = np.arange(np.min(zs), np.max(zs))
            y_slice = np.arange(np.min(ys), np.max(ys))
            x_slice = np.arange(np.min(xs), np.max(xs))
        else:
            z, y, x = volume.shape
            z_slice = np.arange(z)
            y_slice = np.arange(y)
            x_slice = np.arange(x)

    # TODO: better way to do the slicing
    # slice_indices = np.meshgrid(y_slice, z_slice, x_slice)
    # volume = volume[slice_indices]
    crop_volume = volume.copy()
    crop_volume = crop_volume[z_slice]
    crop_volume = crop_volume[:, y_slice]
    crop_volume = crop_volume[...,x_slice]

    if np.sum(volume==0) == volume.size:
        print('No mask')
    else:
        # volume = volumetric_data_eval.volume_preprocess(volume, connectivity=26, area_threshold=0)
        volume_list = [np.int32(crop_volume==label) for label in np.unique(crop_volume)[1:]]
        plot_volume_in_mesh(volume_list, 0, save_path2)
        binary_volume = np.where(volume>0, 1, 0)
        plot_volume_in_mesh(binary_volume, 0, save_path1)


# def show_mask_in_2d(cfg, vol_generator):
#     volume_generator = vol_generator(cfg.FULL_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
#                                      only_nodule_slices=cfg.ONLY_NODULES)
#     fig, ax = plt.subplots(1, 1, figsize=(4,4))
#     post_processer = VolumePostProcessor(connectivity=26, area_threshold=30)
#     for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
#         pid, scan_idx = infos['pid'], infos['scan_idx']
#         mask_vol = np.int32(mask_vol)
#         if vol_idx in [0, 2, 3, 7, 10 ,11]:
#             if np.sum(mask_vol==0) == mask_vol.size:
#                 print('No mask')
#                 continue
#             mask_vol2 = post_processer(mask_vol)
#             def plot_func(volume, name):
#                 zs, ys, xs = np.where(volume)
#                 # min_nonzero_slice, max_nonzero_slice = np.min(zs), np.max(zs)
#                 zs = np.unique(zs)
#                 for slice_idx in zs:
#                     ax.imshow(volume[slice_idx], vmin=0, vmax=5)
#                     fig.savefig(os.path.join(cfg.SAVE_PATH, '2d_mask', f'{name}-{vol_idx:03d}-{slice_idx:03d}.png'))

#             plot_func(mask_vol, 'raw')
#             plot_func(mask_vol2, 'preprocess')


def plot_volume_in_mesh(volume_geroup, threshold=-300, save_path=None): 
    if not isinstance(volume_geroup, list):
        volume_geroup = [volume_geroup]

    # colors = [[0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5], [0.5, 1, 1], [1, 1, 0.5], [1, 0.5, 1],
    #           [0.1, 0.7, 1], [0.7, 1, 0.1], [1, 0.7, 0.1], [0.1, 0.7, 0.7], [0.7, 0.7, 0.1], [0.7, 0.1, 0.7]]
    colors = get_color_from_cmap(cmap=cm.tab20, output_format='bgr', keep_transparent=False, output_uint8=False)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for vol_idx, vol in enumerate(volume_geroup):
        p = vol.transpose(2,1,0)
        verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
        mesh = Poly3DCollection(verts[faces], alpha=0.3)
        face_color = colors[vol_idx%len(colors)]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)

def plot_3d(image, threshold=-300): 
    p = image.transpose(2,1,0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def save_mask(img, mask, pred, num_class, save_path, save_name='img', saving_condition=True):
    save_name = save_name.split('.')[-1]
    if saving_condition:
        condition = (np.sum(mask)>0 or np.sum(pred)>0)
    else:
        condition = True
        
    if condition:
        origin_save_path = os.path.join(save_path, 'origin')
        if not os.path.isdir(origin_save_path):
            os.makedirs(origin_save_path)
        fig1, _ = compare_result(img, mask, pred, show_mask_size=True, alpha=0.2, vmin=0, vmax=num_class-1)
        fig1.savefig(os.path.join(origin_save_path, f'{save_name}.png'))
        plt.close(fig1)

        enlarge_save_path = os.path.join(save_path, 'enlarge')
        if not os.path.isdir(enlarge_save_path):
            os.makedirs(enlarge_save_path)
        fig2, _ = compare_result_enlarge(img, mask, pred, show_mask_size=False, alpha=0.2, vmin=0, vmax=num_class-1)
        if fig2 is not None:
            fig2.savefig(os.path.join(enlarge_save_path, f'{save_name}-en.png'))
            plt.close(fig2)


class visualizer():
    def __init__(self, save_path):
        self.save_path = save_path

    @staticmethod
    def visualize_segmentation_in_2d(image, mask, pred, num_class, save_path, save_name='img', mask_or_pred_exist=True):
        if mask_or_pred_exist:
            condition = (np.sum(mask)>0 or np.sum(pred)>0)
        else:
            condition = True
            
        if condition:
            sub_save_path = save_path
            if not os.path.isdir(sub_save_path):
                os.makedirs(sub_save_path)

            fig1, _ = compare_result(image, mask, pred, show_mask_size=True, alpha=0.2, vmin=0, vmax=num_class-1)
            fig1.savefig(os.path.join(sub_save_path, f'{save_name}.png'))
            plt.close(fig1)

            fig2, _ = compare_result_enlarge(image, mask, pred, show_mask_size=False, alpha=0.2, vmin=0, vmax=num_class-1)
            if fig2 is not None:
                fig2.savefig(os.path.join(sub_save_path, f'{save_name}-en.png'))
                plt.close(fig2)

    @staticmethod
    def visualize_segmentation_in_3d(cls):
        pass

    def visulize_sample(self, ):
        self.visualize_segmentation_in_2d()
        self.visualize_segmentation_in_3d()

