import numpy as np
import time

from post.data_postprocess import VolumePostProcessor
from post.nodule import LungNoduleStudy
from post.reduce_false_positive import NoduleClassifier


def reduce_fp(pred_study):
    total_removal = []
    remove_result = {}
    for name, remove_nodule_ids in pred_study.remove_nodule_record.items():
        other_method = pred_study.remove_nodule_record.copy()
        other_method.pop(name)
        other_method_pred = []
        for idx in other_method.values():
            other_method_pred.extend(list(idx))
        unique_pred = list(set(remove_nodule_ids)-set(other_method_pred))
        unique_pred_num = len(unique_pred)
        common_pred_num = len(remove_nodule_ids) - unique_pred_num
        if name not in remove_result:
            remove_result[name] = {}

        if 'unique' in remove_result[name] and 'common' in remove_result[name]:
            remove_result[name]['unique'] = remove_result[name]['unique'] + unique_pred_num
            remove_result[name]['common'] = remove_result[name]['common'] + common_pred_num
        else:
            remove_result[name].update({'unique': unique_pred_num, 'common': common_pred_num})
        total_removal.extend(remove_nodule_ids)

        # tp, fp = 0, 0
        # for n_id in remove_nodule_ids:
        #     if 'tp' in pred_study.nodule_evals[n_id]:
        #         tp += 1
        #     elif 'fp' in pred_study.nodule_evals[n_id]:
        #         fp += 1
        # print(f'-- {name} TP {tp} FP {fp}')
        # if 'TP' in remove_result[name] and 'FP' in remove_result[name]:
        #     remove_result[name]['TP'] = remove_result[name]['TP'] + tp
        #     remove_result[name]['FP'] = remove_result[name]['FP'] + fp
        # else:
        #     remove_result[name].update({'TP': tp, 'FP': fp})

    total_removal = list(set(total_removal))
    # post_nodule_num = len(pred_study.nodule_evals)-len(total_removal)
    

    pred_vol_category = pred_study.category_volume
    tp, fp = 0, 0
    for remove_nodule_id in total_removal:
        # if 'tp' in pred_study.nodule_evals[remove_nodule_id]:
        #     tp += 1
        # elif 'fp' in pred_study.nodule_evals[remove_nodule_id]:
        #     fp += 1
        pred_vol_category[pred_vol_category==remove_nodule_id] = 0
    pred_study.category_volume = pred_vol_category
    # if 'total' in remove_result:
    #     remove_result['total']['TP'] = remove_result['total']['TP'] + tp
    #     remove_result['total']['FP'] = remove_result['total']['FP'] + fp
    # else:
    #     remove_result['total'] = {'TP': tp, 'FP': fp}
    return pred_study


def get_nodule_id(pred_vol_category):
    return [pred_nodule_id for pred_nodule_id in np.unique(pred_vol_category)[1:]]


def remove_unusual_nodule_by_lung_size(pred_study, lung_mask_vol, min_lung_ration=0.5):
    lung_mask_pxiel_sum = np.sum(lung_mask_vol, axis=(1,2))
    ratio = lung_mask_pxiel_sum / np.max(lung_mask_pxiel_sum)
    remove_mask = np.where(ratio>=min_lung_ration, 0, 1)
    remove_mask = np.reshape(remove_mask, [remove_mask.size, 1, 1])

    pred_vol_category = pred_study.category_volume * remove_mask

    remove_nodule_ids = get_nodule_id(pred_vol_category)
    pred_study.record_nodule_removal(name='RUNLS', nodules_ids=remove_nodule_ids)
    return pred_study


def under_slice_removal(pred_study, slice_threshold=1):
    remove_nodule_ids = []
    for nodule_id, nodule in pred_study.nodule_instances.items():
        max_z = nodule.nodule_range['index']['max']
        min_z = nodule.nodule_range['index']['min']
        if max_z-min_z < slice_threshold:
            remove_nodule_ids.append(nodule_id)

    pred_study.record_nodule_removal(name='_1SR', nodules_ids=remove_nodule_ids)
    return pred_study


def simple_post_processor(vol, 
                          mask_vol, 
                          pred_vol,
                          lung_mask_vol,
                          pid,
                          FP_reducer_checkpoint,
                          _1SR=True,
                          RUNLS=True,
                          nodule_cls=True,
                          raw_vol=None,
                          connectivity=26,
                          area_threshold=8,
                          lung_size_threshold=0.2,
                          nodule_cls_prob=0.5,
                          crop_range=(32,64,64)
                          ):
    if raw_vol is None:
        raw_vol = vol
    start = time.time()
    # Data post-processing
    # post_processer = VolumePostProcessor(connectivity, area_threshold)
    # pred_vol_category = post_processer(pred_vol)
    pred_vol_category = pred_vol
    # target_vol_category = post_processer(mask_vol)
    num_pred_nodule = np.unique(pred_vol_category).size-1
    # target_study = LungNoduleStudy(pid, target_vol_category, raw_volume=raw_vol)
    pred_study = LungNoduleStudy(pid, pred_vol_category, raw_volume=raw_vol)
    print(f'Predict Nodules (raw) {num_pred_nodule}')

    # False positive reducing
    if _1SR:
        pred_study = under_slice_removal(pred_study)

    if RUNLS:
        pred_study = remove_unusual_nodule_by_lung_size(pred_study, lung_mask_vol, min_lung_ration=lung_size_threshold)
    post_end = time.time()

    # Nodule classification
    if nodule_cls:
        # TODO: move object out
        crop_range = {'index': crop_range[0], 'row': crop_range[1], 'column': crop_range[2]}
        nodule_classifier = NoduleClassifier(
            crop_range, FP_reducer_checkpoint, prob_threshold=nodule_cls_prob)
        pred_study, pred_nodule_info = nodule_classifier.nodule_classify(
            vol, pred_study, mask_vol)
    pred_study = reduce_fp(pred_study)
    cls_end = time.time()

    post_time = post_end - start
    cls_time = cls_end - post_end
    pred_vol = pred_study.category_volume
    return pred_vol, post_time, cls_time
    