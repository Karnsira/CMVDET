from .builder import DATASETS
from .custom import CustomDataset
import mmcv
import numpy as np
import pandas as pd


@DATASETS.register_module()
class CMVDataset(CustomDataset):
    
    CLASSES = ('CMV',)

    def load_annotations(self, ann_file):
        
        cat2label = {'CMV': 0}
        image_list = mmcv.list_from_file(self.ann_file)

        data_infos = []

        anno = pd.read_csv(self.img_prefix+'/annotations.csv')

        for image_id in image_list:
            filename = image_id
            height, width = 256, 256

            data_info = dict(filename=image_id, width=width, height=height)

            gt_bboxes = []
            gt_labels = []
            bbox_names = []
            bboxes = []

            anno_file = anno[anno['filename']==filename]

            for i in range(len(anno_file)):
                gt_labels.append(cat2label[anno_file['class'].values[i]])
                gt_bboxes.append(anno_file[['x_min', 'y_min', 'x_max', 'y_max']].values[i])

            data_anno = dict(bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                            labels=np.array(gt_labels, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos