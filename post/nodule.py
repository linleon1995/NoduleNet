
import numpy as np


class LungNoduleStudy():
    def __init__(self, study_id, category_volume, raw_volume=None):
        self.study_id = study_id
        self.category_volume = category_volume
        self.raw_volume = raw_volume
        self.nodule_ids = np.unique(category_volume).tolist()
        self.nodule_ids.remove(0)
        self.nodule_num = len(self.nodule_ids)
        self.volume_shape = category_volume.shape
        self.study_evals = {}
        self.nodule_evals = {}
        self.remove_nodule_record = {}
        self.nodule_instances = self.build_nodule_instance()

    def build_nodule_instance(self):
        nodule_mapping = {}
        for id in self.nodule_ids:
            nodule_volume = np.uint8(np.where(self.category_volume==id, 1, 0))
            # TODO: caused by test with different image size
            # TODO: bug when plot scatter but hu=None
            # hu = None
            if self.raw_volume is not None:
                hu = np.mean(nodule_volume*self.raw_volume)
                # hu = np.mean(nodule_volume*self.raw_volume[...,0])
            else:
                hu = None
            nodule_mapping[id] = Nodule(self.study_id, id, nodule_volume, hu)
        return nodule_mapping

    def record_nodule_removal(self, name, nodules_ids):
        self.remove_nodule_record[name] = nodules_ids

    def get_binary_volume(self):
        return np.where(self.category_volume>0, 1, 0)
    
    def set_score(self, score_name, score):
        self.study_evals[score_name] = score

    def get_score(self, score_name):
        if score_name in self.study_evals:
            return self.study_evals[score_name]
        else:
            return None

class Nodule():
    def __init__(self, study_id, id, nodule_volume, hu=None):
        # TODO: Potnetially memory problem caused by large number nodule. Consider nodule_volume recording.
        self.study_id = study_id
        self.id = id
        self.hu = hu
        self.nodule_score = {}
        self.nodule_volume = nodule_volume
        self.nodule_size = np.sum(nodule_volume)
        self.nodule_range, self.nodule_center = self.get_nodule_position(nodule_volume)

    def get_nodule_position(self, nodule_volume):
        zs, ys, xs = np.where(nodule_volume)
        nodule_range = {
            'index': {'min': np.min(zs), 'max': np.max(zs)},
            'row': {'min': np.min(ys), 'max': np.max(ys)},
            'column': {'min': np.min(xs), 'max': np.max(xs)},
        }
        nodule_center = {
            'index': np.mean(zs),
            'row': np.mean(ys),
            'column': np.mean(xs),
        }
        return nodule_range, nodule_center

    def set_score(self, score_name, score):
        self.nodule_score[score_name] = score

    def get_score(self, score_name):
        if score_name in self.nodule_score:
            return self.nodule_score[score_name]
        else:
            return None
