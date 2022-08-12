import pickle
from mmcls.apis import init_model, inference_model
from tqdm import tqdm
import openslide
import cv2
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file path')
parser.add_argument('--checkpoint', help='checkpoint file path')
parser.add_argument('--input', help='input file path from detection stage in pickle format')
parser.add_argument('--output', help='output file in pickle format')
parser.add_argument('--wsi', help='WSI folder path', default='/data/WSI/')
parser.add_argument('--gpu-id', help='ids of gpus to use')

args = parser.parse_args()

input_path = args.input
output_path = args.output
WSI_path = args.wsi

config = args.config
checkpoint = args.checkpoint
device = 'cuda:' + str(args.gpu_id)
img_size = 224


def inference(config:str, checkpoint:str, device:str, input_pkl:str, output_pkl:str, img_size:int, WSI_path:str):
    
    model = init_model(config, checkpoint, device)
    
    with open(input_pkl, 'rb') as pkl:
        det_res = pickle.load(pkl)

    cls_res = {}

    for slidename in tqdm(det_res):

        det_res_slide = det_res[slidename]
        cls_res[slidename] = []

        slide =  openslide.OpenSlide(WSI_path+slidename)

        for i, det_pred in enumerate(det_res_slide):

            x_min, y_min, x_max, y_max = det_pred[:4]

            anno_width = x_max - x_min
            anno_height = y_max - y_min

            size = int(max(anno_width, anno_height)) + int(max(anno_width, anno_height)/2)

            x_center = x_min + anno_width/2
            y_center = y_min + anno_height/2

            patch_x_origin = int(x_center - size/2)
            patch_y_origin = int(y_center - size/2)

            x_min_rescaled = int(size/2 - anno_width/2)
            x_max_rescaled = int(size/2 + anno_width/2)
            y_min_rescaled = int(size/2 - anno_height/2)
            y_max_rescaled = int(size/2 + anno_height/2)

            # change from wsi coor to patch coor
            x_change = x_min-x_min_rescaled
            y_change = y_min-y_min_rescaled

            patch = slide.read_region((patch_x_origin, patch_y_origin), 0, (size, size)).convert('RGB')
            patch = patch.resize((img_size, img_size), Image.BICUBIC)

            result = inference_model(model, cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR))

            pred_score = abs(1-result['pred_label']-result['pred_score'])
            cls_res[slidename].append(np.array([x_min, y_min, x_max, y_max, pred_score]))
    
    cls_res = { slidename : np.array(cls_res[slidename]) for slidename in cls_res }
    
    with open(output_pkl, 'wb') as pkl:
        pickle.dump(cls_res, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    
    return cls_res


inference(config=config, 
          checkpoint=checkpoint, 
          device=device, 
          input_pkl=input_path, 
          output_pkl=output_path, 
          img_size=img_size, 
          WSI_path=WSI_path)