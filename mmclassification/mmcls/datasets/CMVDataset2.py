import mmcv
import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset

@DATASETS.register_module()
class CMVDataset2(BaseDataset):
    
    CLASSES = ('Melanoma', 'Ganglion', 'CMV')
    
    def load_annotations(self):
        
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(self.data_prefix, folder_to_idx, extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples
        
        image_list = mmcv.list_from_file(self.ann_file)

        data_infos = []

        for filename in image_list:

            if filename[:3] == 'cmv':
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(2, dtype=np.int64)
            elif filename[:3] == 'gan':
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(1, dtype=np.int64)
            elif filename[:3] == 'mel':
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(0, dtype=np.int64)

            data_infos.append(info)

        return data_infos