from pylung.annotation import parse
from utils.LIDC.cvrt_annos_to_npy import load_itk_image


def xml_parse(xml_file):
    # get metadata (slicer thickness)
    header, annos = parse(xml_file)
    series_uid = header.series_instance_uid
    study_uid = header.study_instance_uid
    for i, reader in enumerate(annos):
        for j, nodule in enumerate(reader.nodules):
            ctr_arr = []
            for k, roi in enumerate(nodule.rois):
                z = roi.z
                for roi_xy in roi.roi_xy:
                    ctr_arr.append([z, roi_xy[1], roi_xy[0]])
            malignancy = nodule.characteristics.malignancy
            # id = nodule.id
            # if id in total_malignancy:
            #     total_malignancy[id].append(malignancy)
            # else:
            #     total_malignancy[id] = []
            # ctr_arr.insert(0, [malignancy])
            total_malignancy.append(malignancy)
            ctr_arrs.append(ctr_arr)

def get_lidc_metadata():
    f = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
    a = load_itk_image(f)
    # metadata = xml_parse(xml_file)

    # get pid

    # get spacing

    # get diameter

    # save data frame


def main():
    get_lidc_metadata()

if __name__ == "__main__":
    main()