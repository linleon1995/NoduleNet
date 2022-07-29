import numpy as np
import torch

from post.ResNet_3d import build_3d_resnet

# TODO: torch, numpy problem
# TODO: class design

class NoduleClassifier():
    def __init__(self, crop_range, checkpint_path, model_depth=50, num_class=2, prob_threshold=0.5):
        self.crop_range = crop_range
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.classifier = self.build_classifier(checkpint_path, model_depth, num_class)
        self.prob_threshold = prob_threshold

    def nodule_classify(self, raw_volume, pred_study, target_volume):
        pred_study, pred_nodules = self.reduce_false_positive(raw_volume, pred_study)
        for nodule_id in pred_nodules:
            crop_target_volume = crop_volume(target_volume, self.crop_range, pred_nodules[nodule_id]['Center'])
            target_class = np.where(np.sum(crop_target_volume)>0, 1, 0)
            pred_class = pred_nodules[nodule_id]['Nodule_pred_class']
            result = self.eval(target_class, pred_class)
            pred_nodules[nodule_id]['eval'] = result
        # print(np.unique(pred_volume_category))
        return pred_study, pred_nodules

    def reduce_false_positive(self, raw_volume, pred_study):
        pred_volume_category = pred_study.category_volume
        pred_nodules = self.get_nodule_center(pred_volume_category)
        remove_nodule_ids = []
        for nodule_id in list(pred_nodules):
            crop_raw_volume = crop_volume(raw_volume, self.crop_range, pred_nodules[nodule_id]['Center'])
            # crop_raw_volume = np.swapaxes(np.expand_dims(crop_raw_volume, (0, 1)), 1, 5)[...,0]
            crop_raw_volume = np.expand_dims(crop_raw_volume, (0, 1))
            crop_raw_volume = np.tile(crop_raw_volume, (1, 3, 1, 1, 1))
            crop_raw_volume = torch.from_numpy(crop_raw_volume).float().to(self.device)

            pred_class, pred_prob = self.inference(crop_raw_volume)
            # pred_class = pred_class.item()
            if pred_class == 0:
                remove_nodule_ids.append(nodule_id)
                pred_nodules.pop(nodule_id)
            else:
                pred_nodules[nodule_id]['Nodule_pred_class'] = pred_class
                pred_nodules[nodule_id]['Nodule_pred_prob'] = pred_prob.cpu().detach().numpy()[0]

            pred_study.record_nodule_removal(name='NC', nodules_ids=remove_nodule_ids)
        return pred_study, pred_nodules

    def inference(self, input_volume):
        logits = self.classifier(input_volume)
        prob = torch.nn.functional.softmax(logits, dim=1)
        # pred = torch.argmax(prob)
        if prob[0,1] > self.prob_threshold:
            pred = 1
        else:
            pred = 0
        return pred, prob

    def build_classifier(self, checkpint_path, model_depth, num_class):
        state_key = torch.load(checkpint_path, map_location=self.device)
        model = build_3d_resnet(model_depth, n_classes=num_class, conv1_t_size=7, conv1_t_stride=2)
        model.load_state_dict(state_key['net'])
        model = model.to(self.device)
        return model

    @classmethod
    def get_nodule_center(cls, pred_volume):
        total_nodule_center = {}
        for nodule_id in np.unique(pred_volume)[1:]:
            zs, ys, xs = np.where(pred_volume==nodule_id)
            center_index, center_row, center_column = np.mean(zs), np.mean(ys), np.mean(xs)
            total_nodule_center[nodule_id] = {'Center': {'index': np.mean(center_index).astype('int32'), 'row': np.mean(center_row).astype('int32'), 'column': np.mean(center_column).astype('int32')}}

        return total_nodule_center

    def eval(self, target, pred):
        if target == 0 and pred == 0:
            return 'tn'
        elif target == 0 and pred == 1:
            return 'fp'
        elif target == 1 and pred == 0:
            return 'fn'
        elif target == 1 and pred == 1:
            return 'tp'
        

def crop_volume(volume, crop_range, crop_center):
    def get_interval(crop_range_dim, center, size_dim):
        begin = center - crop_range_dim//2
        end = center + crop_range_dim//2
        if begin < 0:
            begin, end = 0, end-begin
        elif end > size_dim:
            modify_distance = end - size_dim + 1
            begin, end = begin-modify_distance, size_dim-1
        # print(crop_range_dim, center, size_dim, begin, end)
        assert end-begin == crop_range_dim, f'Actual cropping range {end-begin} not fit the required cropping range {crop_range_dim}'
        return (begin, end)

    index_interval = get_interval(crop_range['index'], crop_center['index'], volume.shape[0])
    row_interval = get_interval(crop_range['row'], crop_center['row'], volume.shape[1])
    column_interval = get_interval(crop_range['column'], crop_center['column'], volume.shape[2])

    return volume[index_interval[0]:index_interval[1], 
                    row_interval[0]:row_interval[1], 
                    column_interval[0]:column_interval[1]]

