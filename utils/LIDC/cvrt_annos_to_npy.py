import sys

from importlib_metadata import metadata
sys.path.append('./')
from pylung.annotation import *
from tqdm import tqdm
import sys
import nrrd
import SimpleITK as sitk
import cv2
from config import config
import pandas as pd



def get_files(path, keys=[], return_fullpath=True, sort=True, sorting_key=None, recursive=True, get_dirs=False, ignore_suffix=False):
    """Get all the file name under the given path with assigned keys
    Args:
        path: (str)
        keys: (list, str)
        return_fullpath: (bool)
        sort: (bool)
        sorting_key: (func)
        recursive: The flag for searching path recursively or not(bool)
    Return:
        file_list: (list)
    """
    file_list = []
    assert isinstance(keys, (list, str))
    if isinstance(keys, str): keys = [keys]
    # Rmove repeated keys
    keys = list(set(keys))

    def push_back_filelist(root, f, file_list, is_fullpath):
        f = f[:-4] if ignore_suffix else f
        if is_fullpath:
            file_list.append(os.path.join(root, f))
        else:
            file_list.append(f)

    for i, (root, dirs, files) in enumerate(os.walk(path)):
        # print(root, dirs, files)
        if not recursive:
            if i > 0: break

        if get_dirs:
            files = dirs
            
        for j, f in enumerate(files):
            if keys:
                for key in keys:
                    if key in f:
                        push_back_filelist(root, f, file_list, return_fullpath)
            else:
                push_back_filelist(root, f, file_list, return_fullpath)

    if file_list:
        if sort: file_list.sort(key=sorting_key)
    else:
        f = 'dir' if get_dirs else 'file'
        if keys: 
            logging.warning(f'No {f} exist with key {keys}.') 
        else: 
            logging.warning(f'No {f} exist.') 
    return file_list


def load_itk_image(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def xml2mask(xml_file):
    header, annos = parse(xml_file)
    seriesuid = header.series_instance_uid
    studyuid = header.study_instance_uid
    ctr_arrs = []
    meta_data = {'seriesuid': seriesuid, 'studyuid': studyuid}
    nodule_characteristics = [
        'subtlety', 'internalStructure', 'calcification',
        'sphericity', 'margin', 'lobulation',
        'spiculation', 'texture', 'malignancy',
    ]
    
    for i, reader in enumerate(annos):
        for j, nodule in enumerate(reader.nodules):
            ctr_arr = []
            for k, roi in enumerate(nodule.rois):
                z = roi.z
                for roi_xy in roi.roi_xy:
                    ctr_arr.append([z, roi_xy[1], roi_xy[0]])

            for attr in nodule_characteristics:
                if attr not in meta_data:
                    meta_data[attr] = []
                if hasattr(nodule.characteristics, attr):
                    value = getattr(nodule.characteristics, attr)
                    meta_data[attr].append(value)

            ctr_arrs.append(ctr_arr)

    for meta_key in meta_data:
        if meta_key not in ['seriesuid', 'studyuid']:
            if meta_key == 'calcification':
                meta_data[meta_key] = np.clip(meta_data[meta_key], 1, 6)
            else:
                meta_data[meta_key] = np.clip(meta_data[meta_key], 1, 5)
   
    # ctr_arrs = [total_malignancy, ctr_arrs]
    return seriesuid, ctr_arrs, meta_data


def annotation2masks(annos_dir, save_dir):
    files = find_all_files(annos_dir, '.xml')
    for f in tqdm(files, total=len(files)):
        try:
            seriesuid, masks, meta_data = xml2mask(f)
            # np.save(os.path.join(save_dir, '%s' % (seriesuid)), masks)
            os.makedirs(os.path.join(save_dir, 'metadata'), exist_ok=True)
            np.save(os.path.join(save_dir, 'metadata', '%s' % (seriesuid)), meta_data)
        except:
            print("Unexpected error:", sys.exc_info()[0])


def arr2mask(arr, reso):
    mask = np.zeros(reso)
    arr = arr.astype(np.int32)
    mask[arr[:, 0], arr[:, 1], arr[:, 2]] = 1
    
    return mask

def arrs2mask(img_dir, ctr_arr_dir, save_dir):
    # pids = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.mhd')]
    meta_dir = os.path.join(ctr_arr_dir, 'metadata')
    data_path_list = get_files(img_dir, 'mhd')
    # data_path_list = [data_path for data_path in data_path_list if 'mask' in data_path]

    cnt = 0
    consensus = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for k in consensus.keys():
        if not os.path.exists(os.path.join(save_dir, str(k))):
            os.makedirs(os.path.join(save_dir, str(k)))

    # for pid in tqdm(pids, total=len(pids)):
    keep = 0
    num_nodule = 0
    num_file = len(data_path_list)
    zero_pid = []
    row_list = []
    pd_column = [
        'seriesuid', 'studyuid',
        'subtlety', 'internalStructure', 'calcification',
        'sphericity', 'margin', 'lobulation',
        'spiculation', 'texture', 'malignancy',
    ]
    for idx, data_path in enumerate(data_path_list):
        # if idx > 10: break
        folder, filename = os.path.split(data_path)
        # _, subset = os.path.split(folder)
        pid = filename[:-4]
        print(f'{idx}/{num_file} {pid}')
        img, origin, spacing = load_itk_image(data_path)
        ctr_arrs = np.load(os.path.join(ctr_arr_dir, '%s.npy' % (pid)), allow_pickle=True)
        meta_data = np.load(os.path.join(meta_dir, '%s.npy' % (pid)), allow_pickle=True)
        meta_data = meta_data.tolist()
        
        nodule_masks = []
        # annot_malignancy = []
        annot_malignancy = ctr_arrs[0]
        ctr_arrs = ctr_arrs[1]
        cnt += (len(ctr_arrs)-1)
        for ctr_arr in ctr_arrs:
            z_origin = origin[0]
            z_spacing = spacing[0]
            ctr_arr = np.array(ctr_arr)
            ctr_arr[:, 0] = np.absolute(ctr_arr[:, 0] - z_origin) / z_spacing
            ctr_arr = ctr_arr.astype(np.int32)

            mask = np.zeros(img.shape)

            for z in np.unique(ctr_arr[:, 0]):
                ctr = ctr_arr[ctr_arr[:, 0] == z][:, [2, 1]]
                ctr = np.array([ctr], dtype=np.int32)
                mask[z] = cv2.fillPoly(mask[z], ctr, color=(1,) * 1)
            nodule_masks.append(mask)
            # annot_malignancy.append(annot_m)

        i = 0
        visited = []
        d = {}
        masks = []
        total_malignancy = []
        total_metadata = {meta_key: [] for meta_key in meta_data \
                          if meta_key not in ['seriesuid', 'studyuid'] and len(meta_data[meta_key]) > 0}
        while i < len(nodule_masks):
            # If mached before, then no need to create new mask
            if i in visited:
                i += 1
                continue
            same_nodules = []
            malignancy = []
            temp_metadata = {meta_key: [] for meta_key in meta_data \
                             if meta_key not in ['seriesuid', 'studyuid'] and len(meta_data[meta_key]) > 0}
            mask1 = nodule_masks[i]
            same_nodules.append(mask1)
            malignancy.append(annot_malignancy[i])
            for meta_key in temp_metadata:
                temp_metadata[meta_key].append(meta_data[meta_key][i])
            d[i] = {}
            d[i]['count'] = 1
            d[i]['iou'] = []

            # Find annotations pointing to the same nodule
            for j in range(i + 1, len(nodule_masks)):
                # if not overlapped with previous added nodules
                if j in visited:
                    continue
                mask2 = nodule_masks[j]
                iou = float(np.logical_and(mask1, mask2).sum()) / np.logical_or(mask1, mask2).sum()

                if iou > 0.4:
                    visited.append(j)
                    same_nodules.append(mask2)
                    malignancy.append(annot_malignancy[j])
                    for meta_key in temp_metadata:
                        temp_metadata[meta_key].append(meta_data[meta_key][j])
                    d[i]['count'] += 1
                    d[i]['iou'].append(iou)

            masks.append(same_nodules)
            total_malignancy.append(malignancy)
            for meta_key in temp_metadata:
                total_metadata[meta_key].append(temp_metadata[meta_key])
            i += 1

        for k, v in d.items():
            if v['count'] > 4:
                print('WARNING:  %s: %dth nodule, iou: %s' % (pid, k, str(v['iou'])))
                v['count'] = 4
            consensus[v['count']] += 1

        # number of consensus
        num = np.array([len(m) for m in masks])
        num[num > 4] = 4
        
        if len(num) == 0:
            zero_pid.append(pid)
            continue
        # Iterate from the nodules with most consensus
        for n in range(num.max(), 0, -1):
            if n != 3:
                continue
            mask = np.zeros(img.shape, dtype=np.uint8)
            
            final_metadata = {}
            for i, index in enumerate(np.where(num >= n)[0]):
                same_nodules = masks[index]
                nodule_malignancy = np.median(total_malignancy[index])
                for meta_key in meta_data:
                    if meta_key in total_metadata:
                        final_metadata[meta_key] = np.median(total_metadata[meta_key][index])
                    elif meta_key in ['seriesuid', 'studyuid']:
                        final_metadata[meta_key] = meta_data[meta_key]
                    else:
                        final_metadata[meta_key] = ''

                # nodule_malignancy = np.mean(total_malignancy[index])
                # if nodule_malignancy >= 3:
                #     nodule_malignancy = 2
                # else:
                #     nodule_malignancy = 1
                m = np.logical_or.reduce(same_nodules)
                mask[m] = nodule_malignancy
                # mask[m] = i + 1

            content = list(final_metadata.values())
            print(content, len(content))
            df = pd.DataFrame(content)
            row_list.append(df.T)

            # nrrd.write(os.path.join(save_dir, str(n), pid), mask)
            keep += 1
            num_nodule += len(num)
            # np.save(os.path.join(save_dir, str(n), f'{pid}.npy'), mask)
            # print(pid, np.unique(mask))

        
#         for i, same_nodules in enumerate(masks):
#             cons = len(same_nodules)
#             if cons > 4:
#                 cons = 4
#             m = np.logical_or.reduce(same_nodules)
#             mask[m] = i + 1
#             nrrd.write(os.path.join(save_dir, str(cons), pid), mask)
    df = pd.concat(row_list, axis=0, ignore_index=True)
    df.columns = pd_column
    df.to_csv(os.path.join(save_dir, 'data_samples.csv'), index=False)
    print(consensus)
    print(cnt)
    print(keep)
    print(num_nodule)


if __name__ == '__main__':
    # f3 = get_files(rf'C:\Users\test\Desktop\Leon\Datasets\masks_test\3', 'npy', return_fullpath=False)
    # f4 = get_files(rf'C:\Users\test\Desktop\Leon\Datasets\masks_test\4', 'npy', return_fullpath=False)
    # print(len(list(set(f3)-set(f4))))


    annos_dir = config['annos_dir']
    img_dir = config['data_dir']
    ctr_arr_save_dir = config['ctr_arr_save_dir']
    mask_save_dir = config['mask_save_dir']

    os.makedirs(ctr_arr_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    # annotation2masks(annos_dir, ctr_arr_save_dir)
    arrs2mask(img_dir, ctr_arr_save_dir, mask_save_dir)

    # f = rf'C:\Users\test\Desktop\Leon\Datasets\annotation\mask_test\metadata\1.3.6.1.4.1.14519.5.2.1.6279.6001.101228986346984399347858840086.npy'
    # aa = np.load(f, allow_pickle=True)
    # bb = aa.tolist()
    # print(aa)
