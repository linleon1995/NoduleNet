import nrrd
import numpy as np
import os


def save_nodule_in_nrrd(vol, mask_vol, direction, origin, spacing, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)

    vol = np.transpose(vol, (2, 1, 0))
    mask_vol = np.transpose(mask_vol, (2, 1, 0))
    spacing = np.array([spacing[1], spacing[2], spacing[0]])
    # direction = direction * np.tile(spacing[:,np.newaxis], (1, 3))
    origin = np.array([origin[1], origin[2], origin[0]])
    
    raw_path = os.path.join(save_dir, f'{filename}.nrrd')
    seg_path = os.path.join(save_dir, f'{filename}.seg.nrrd')
    raw_nrrd_write(raw_path, vol, direction, origin, spacing)
    seg_nrrd_write(seg_path, mask_vol, direction, origin, spacing)

    
def seg_nrrd_write(filename, voxels, direction, origin, spacing):
    direction = direction * np.tile(spacing[:,np.newaxis], (1, 3))
    seg_header = build_nrrd_seg_header(voxels, direction, origin)
    nrrd.write(filename, voxels, seg_header)
    print('Seg nrrd data conversion completed.')


def raw_nrrd_write(filename, voxels, direction, origin, spacing):
    direction = direction * np.tile(spacing[:,np.newaxis], (1, 3))
    nrrd_header = build_nrrd_header(voxels, direction, origin)
    nrrd.write(filename, voxels, nrrd_header)
    print('RAW nrrd data conversion completed.')


def build_nrrd_header(arr, direction, origin, space='left-posterior-superior'):
    header = {
        'type': 'unsigned char',
        'dimension': arr.ndim,
        'space': space,
        'sizes': arr.shape,
        'space directions': direction,
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space origin': origin
    }

    # TODO: not sure what is endian for
    return header


def build_nrrd_seg_header(arr, direction, origin, cmap=None, space='left-posterior-superior'):
    # TODO: slicer can decide color automatically
    arr = np.uint8(arr)
    if cmap is None:
        cmap = [(0.5, 0.7, 0), (0.7, 0.5, 0)]

    header = build_nrrd_header(arr, direction, origin, space)
    header.update(build_common_custom_field())
    
    nodule_ids = np.unique(arr)[1:]
    print(f'Nodule number {nodule_ids.size}')
    for idx, nodule_id in enumerate(nodule_ids):
        data = np.uint8(arr==nodule_id)
        color = cmap[idx%len(cmap)]
        seg_header = build_segment_custom_field(nodule_id, color, data)
        header.update(seg_header)
    return header


def build_common_custom_field():
    conversion_params = 'Collapse labelmaps|1|Merge the labelmaps \
                        into as few shared labelmaps as possible \
                        1 = created labelmaps will be shared if possible \
                        without overwriting each other.&Compute surface \
                        normals|1|Compute surface normals. 1 (default) = \
                        surface normals are computed. 0 = surface normals \
                        are not computed (slightly faster but produces less \
                        smooth surface display).&Crop to reference image \
                        geometry|0|Crop the model to the extent of reference \
                        geometry. 0 (default) = created labelmap will contain \
                        the entire model. 1 = created labelmap extent will be \
                        within reference image extent.&Decimation \
                        factor|0.0|Desired reduction in the total \
                        number of polygons. Range: 0.0 (no decimation)\
                        to 1.0 (as much simplification as possible). \
                        Value of 0.8 typically reduces data set size \
                        by 80% without losing too much details.&Fractional \
                        labelmap oversampling factor|1|Determines the \
                        oversampling of the reference image geometry. \
                        All segments are oversampled with the same value \
                        (value of 1 means no oversampling).&Joint \
                        smoothing|0|Perform joint smoothing.&Oversampling \
                        factor|1|Determines the oversampling of the \
                        reference image geometry. If it\'s a number, \
                        then all segments are oversampled with the \
                        same value (value of 1 means no oversampling). If it has the value "A", then automatic oversampling is calculated.&Reference image geometry|-0.64453125;0;0;166;0;-0.64453125;0;133.600006103516;0;0;2.49990081787109;74.0438003540039;0;0;0;1;0;511;0;511;0;130;|Image geometry description string determining the geometry of the labelmap that is created in course of conversion. Can be copied from a volume, using the button.&Smoothing factor|0.5|Smoothing factor. Range: 0.0 (no smoothing) to 1.0 (strong smoothing).&Threshold fraction|0.5|Determines the threshold that the closed surface is created at as a fractional value between 0 and 1.&'
    
    common_custom_field = {
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|',
        'Segmentation_ConversionParameters': conversion_params,
        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_ReferenceImageExtentOffset': '0 0 0',
    }
    return common_custom_field


def build_segment_custom_field(id: int, color: tuple, data: np.array) -> dict:
    key = f'Segment{id}'
    color = f'{color[0]} {color[1]} {color[2]}'

    assert data.ndim == 3
    zs, ys, xs = np.where(data)
    extent = []
    for d in [xs, ys, zs]:
        extent.append(str(np.min(d)))
        extent.append(str(np.max(d)))

    segment_custom_field = {
        f'{key}_Color': color,
        f'{key}_ColorAutoGenerated': '0',
        f'{key}_Extent': ' '.join(extent),
        f'{key}_ID': f'ID_{id:03d}',
        f'{key}_LabelValue': id,
        f'{key}_Layer': '0',
        f'{key}_Name': f'Segment_{id:03d}',
        f'{key}_NameAutoGenerated': '0',
        f'{key}_Tags': 'Segmentation.Status:inprogress|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^272673000^Bone~^^~Anatomic codes - DICOM master list~^^~^^|',
    }
    return segment_custom_field


def main():
    # path = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\pretrained_weights\Unet3D-genesis_chest_ct\run_002\1120-maskrcnn-run_002-ckpt-best-0.1\ASUS-Malignant\test\images\1m0038\nrrd'
    # seg_data, seg_header = nrrd.read(os.path.join(path, rf'1m0038.seg.nrrd'))
    # save_path = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\pretrained_weights\Unet3D-genesis_chest_ct\run_002\1120-maskrcnn-run_002-ckpt-best-0.1\ASUS-Malignant\test\images\1m0038\nrrd'
            
    # filename = os.path.join(save_path, 'test001.seg.nrrd')
    # direction, origin = seg_header['space directions'], seg_header['space origin'], None
    # seg_nrrd_write(filename, seg_data, direction, origin)

    path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\plot\nrrd\Segmentation2.seg.nrrd'
    path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\plot\nrrd\Segmentation_multi.seg.nrrd'
    path2 = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\new_malignant\1m0024\1m0024raw\Segmentation.seg.nrrd'
    raw_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\plot\nrrd\250 No series description_2.nrrd'
    seg_path = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\pretrained_weights\Unet3D-genesis_chest_ct\run_002\1120-maskrcnn-run_002-ckpt-best-0.1\ASUS-Malignant\test\images\1m0038\nrrd\10'
    path = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\pretrained_weights\Unet3D-genesis_chest_ct\run_002\1120-maskrcnn-run_002-ckpt-best-0.1\ASUS-Malignant\test\images\1m0038\nrrd'


    # data, header = nrrd.read(os.path.join(path, rf'1m0038.seg.nrrd'))
    data, header = nrrd.read(os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\new_malignant\1m0035\1m0035raw', rf'2 No series description_2.nrrd'))
    # data, header = nrrd.read(os.path.join(seg_path, rf'1m0038_14-label_1.nrrd'))
    print(np.max(data), np.min(data))
    print(np.sum(data)==0)
    import cc3d
    data = np.transpose(data, (2, 0, 1))
    data_cc = cc3d.connected_components(data , 26)
    print(np.unique(data), np.unique(data).size)
    _dir = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\pretrained_weights\Unet3D-genesis_chest_ct\run_002\1120-maskrcnn-run_002-ckpt-best-0.1\ASUS-Malignant\test\images\1m0038\nrrd'
    filename = os.path.join(_dir, 'test_nrrd.nrrd')
    nrrd.write(filename, data, header)
    
    print(3)

if __name__ == '__main__':
    main()
