#!/usr/bin/env python
import gc
import sys
import cv2
import mmcv
import json
import time
import pickle
import warnings
import openslide
import numpy as np
import torch, torchvision
from torchvision.ops import nms as torch_nms
from pathlib import Path
import concurrent.futures
from lib.utils import *
from lib.nms_WSI import *
from lib.objectDetectionHelper import convertBoxesToOriginalCoor
from datetime import datetime
from mmdet.apis import init_detector, inference_detector

warnings.filterwarnings('ignore')

with open("detection/inference/config.json") as json_data_file:
    config = json.load(json_data_file)

print("--------------- Configuration ---------------")
for i in config : 
    print(f'{i} : {config[i]}')
print("------------------ Inference ------------------")



def load_img(xy_index, sliding_step, patch_size) :   
    xy_origin = tuple(map(lambda p : p*sliding_step, xy_index))
    img = slide.read_region(xy_origin, 0, (patch_size, patch_size)).convert('RGB')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return [img, xy_index]

def inference(model, batch_img, xy_index_list, boxes_dict) :
    model_boxes = inference_detector(model, imgs = batch_img)
    for idx, each_img in enumerate(model_boxes):
        pred_boxes = each_img[0]
        if pred_boxes.size != 0 :
            boxes_dict[xy_index_list[idx]] = np.array([ box for box in pred_boxes])
    return boxes_dict

def display_time(seconds, granularity=2):
    intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

test_slide_filenames = config['WSI_slides']


print('Test slides : ', test_slide_filenames )
print()
print('Total slides : ', len(test_slide_filenames))


WSI_path = config['WSI_path']
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
boxes_path = f'{config["result"]}/{dt_string}/pred_boxes.pkl' #output/dateTime/pred_boxes.pkl
boxes_nms_path = f'{config["result"]}/{dt_string}/pred_boxes_nms_{config["nms_thres"]}.pkl' #output/dateTime/pred_boxes_nms.pkl

checkpoint_file = config['model']
config_file = config['model_config']

model = init_detector(config_file, checkpoint_file, device=config['gpu'] if torch.cuda.is_available() == True  else 'cpu')

slide_step = config['slide_step']
patch_size = config['patch_size']
batch_size = config['batch_size']

nms_thres = config['nms_thres']

img_num = 0
chunk_size = batch_size*10

start_time = time.time()

try :
    result_boxes = load_object(boxes_path)
except :
    print('Creating new detection dict (pkl)')
    result_boxes = dict()
    
try : 
    print('Creating folder for pred_boxes')
    Path(f"{config['result']}/{dt_string}").mkdir(parents=True, exist_ok=True)
except :
    print('Creating folder for pred_boxes failed (pkl)')
    sys.exit()
    
for idx , test_file_name in enumerate(test_slide_filenames) :
    if test_file_name in result_boxes :
        print(f'{test_file_name} has already inferenced')
        continue

    print("\r", f"WSI(CMV) Current Slide : {test_file_name} || {idx+1}/{len(test_slide_filenames)} " , end="")

    boxes_dict = dict()
    try :
        slide = openslide.OpenSlide(WSI_path + test_file_name)
    except :
        print(f'Slide : {test_file_name} not found')
        continue
    
    slide_width, slide_height = slide.dimensions[0], slide.dimensions[1]

    
    x_round = int(slide_width/slide_step) + 1
    y_round = int(slide_height/slide_step) + 1
    
    xi_range, yi_range = range(x_round) , range(y_round)
    
    print('Total images = ',x_round*y_round)
    start = time.time()
    xyi_list = [(xi, yi) for xi in xi_range for yi in yi_range]
    chunks = [xyi_list[i:i + chunk_size] for i in range(0, len(xyi_list), chunk_size)]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for ic, chunk in enumerate(chunks) :
            results = [executor.submit(load_img, xyi, slide_step, patch_size) for xyi in chunk]

            batch_img = list()
            xy_index_list = list()
            chunk_sTime = time.time()

            for f in concurrent.futures.as_completed(results): 

                img, xy_index = f.result()
                batch_img.append(img)
                xy_index_list.append(xy_index)

                if len(batch_img) == batch_size :
                    boxes_dict = inference(model, batch_img, xy_index_list, boxes_dict)     
                    batch_img = list()
                    xy_index_list = list()
                    img_num += batch_size
                    

            if len(batch_img) != 0 :
                boxes_dict = inference(model, batch_img, xy_index_list, boxes_dict)     
                img_num += len(batch_img)
            
            chunk_eTime = time.time()
            chunk_time = chunk_eTime - chunk_sTime
            
    boxes_arr = convertBoxesToOriginalCoor(boxes_dict, slide_step = slide_step)

    result_boxes[test_file_name] = boxes_arr

    save_object(boxes_path, objects = result_boxes)
    save_object(boxes_nms_path, objects = nms(result_boxes, iou_thres = nms_thres)) #nms_boxes

    convert_pkl_to_geoJson(result_boxes, path = config['result'], format='.svs')
    
    try :
        del results
        del batch_img
    except :
        print('Del Large List Failed')
    gc.collect()
    end = time.time()
    print('Time : ',display_time(end - start))

end_time = time.time() 
print('Total time  : ', display_time(end_time - start_time))   
print('Done...')