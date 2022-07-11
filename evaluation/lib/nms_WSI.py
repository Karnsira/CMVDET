import torch, torchvision
import numpy as np
from torchvision.ops import nms as torch_nms

def nms(result_boxes, det_thres=None, iou_thres = 0.5):
    
    for filekey in result_boxes.keys():
        arr = np.array(result_boxes[filekey])
        arr = result_boxes[filekey]
        if arr is not None and isinstance(arr, np.ndarray) and (arr.shape[0] == 0):
            continue
        if (arr.shape[0]>0):
                arr = non_max_suppression(arr, arr[:,-1], det_thres, iou_thres)

        result_boxes[filekey] = arr
    
    return result_boxes

def non_max_suppression(boxes, scores, det_thres=None, iou_thres = 0.5):
    if (det_thres is not None): # perform thresholding
        to_keep = scores>det_thres
        boxes = boxes[to_keep]
        scores = scores[to_keep]
    
    keep_boxes = torch_nms(torch.tensor(boxes[:,:-1]), torch.tensor(scores)  , iou_thres) # det,score
    boxes = boxes[keep_boxes.tolist()]
    return boxes