import numpy as np
import cc3d


class VolumePostProcessor():
    def __init__(self, connectivity=26, area_threshold=8):
        self.connectivity = connectivity
        self.area_threshold = area_threshold

    def __call__(self, binary_volume):
        category_volume = self.connect_components(binary_volume, self.connectivity)
        category_volume = self.remove_small_area(category_volume, self.area_threshold)
        return category_volume

    @classmethod
    def connect_components(cls, array, connectivity):
        connected_conponents_label = cc3d.connected_components(array, connectivity=connectivity)
        return connected_conponents_label

    @classmethod
    def remove_small_area(cls, category_volume, area_threshold):
        # Remove connected area smaller than area_threshold in category_volume
        category = np.unique(category_volume).tolist()
        category.remove(0)
        for label in category:
            area_mask = category_volume==label
            area_size = np.sum(area_mask)
            if area_size < area_threshold:
                category_volume = category_volume * (1-area_mask)
        return category_volume
