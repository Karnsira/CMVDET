import numpy as np
import torch, torchvision
import torchvision.ops.boxes as bops

def cal_iou(gt, det) :
    box1 = torch.tensor([gt], dtype=torch.float)
    box2 = torch.tensor([det], dtype=torch.float)
    iou = bops.box_iou(box1, box2) # tensor([[0.1382]])
    return iou.item()

def convertBoxesToOriginalCoor(boxes_dict : dict, slide_step = 256) :
    boxes_list = list()
    
    for k in boxes_dict :
        
        x_ori = k[0]*slide_step
        y_ori = k[1]*slide_step
    
        tmp_box = list()
        
        for box in boxes_dict[k] :
            
            x1, y1, x2, y2, score = box

            x1 = x_ori + x1
            y1 = y_ori + y1
            x2 = x_ori + x2
            y2 = y_ori + y2

            pred_box = [x1, y1, x2, y2, score]
            
            tmp_box.append(np.array(pred_box))
            boxes_list.append(np.array(pred_box))
        
        boxes_dict[k] = np.array(tmp_box)
        
    return np.array(boxes_list)


def getMaxDetBoxDiagLength(det_bboxes_list, show_info = False) :

    max_width = 0
    max_height = 0
    max_area = 0
    max_width_coor = tuple()
    max_height_coor = tuple()
    max_area_coor = tuple()

    for item in det_bboxes_list:
        x1, y1, x2, y2 = item
        width = x2-x1
        height = y2-y1
        area = width*height
        if width > max_width : 
            max_width = width
            max_width_coor  = (width,height)
        if height > max_height : 
            max_height = height
            max_height_coor  = (width,height)
        if area > max_area : 
            max_area = area
            max_area_coor = (width,height)

    max_wh_coor = (max_width,max_height)

    get_diag_length = lambda xy : np.linalg.norm( np.array(xy) - np.array((xy[0]/2, xy[1]/2)) )

    diag_width_center  = get_diag_length(max_width_coor)
    diag_height_center = get_diag_length(max_height_coor)
    max_diag_det_length = max(diag_width_center,diag_height_center)

    if show_info :
        print('Max Area : ', max_area_coor )
        print('Max (width,height) : ',max_wh_coor)
        print('-------------------------------------------------')
        print('Max width BBoxes : ',max_width_coor)
        print('Max height BBoxes : ',max_height_coor)
        print('-------------------------------------------------')
        print('Max diag width : ', diag_width_center)
        print('Max diag height : ', diag_height_center)
        print('-------------------------------------------------')
        print('Max diag length : ', max_diag_det_length)
        print('\n')
        
    return max_diag_det_length


def getMinDetBoxDiagLength(det_bboxes_list, show_info = False) :

    min_width = float('inf')
    min_height = float('inf')
    min_area = float('inf')
    min_width_coor = tuple()
    min_height_coor = tuple()
    min_area_coor = tuple()

    for item in det_bboxes_list:
        x1, y1, x2, y2 = item
        width = x2-x1
        height = y2-y1
        area = width*height
        if width < min_width : 
            min_width = width
            min_width_coor  = (width,height)
        if height < min_height : 
            min_height = height
            min_height_coor  = (width,height)
        if area < min_area : 
            min_area = area
            min_area_coor = (width,height)

    min_wh_coor = (min_width,min_height)

    get_diag_length = lambda xy : np.linalg.norm( np.array(xy) - np.array((xy[0]/2, xy[1]/2)) )

    diag_width_center  = get_diag_length(min_width_coor)
    diag_height_center = get_diag_length(min_height_coor)
    min_diag_det_length = min(diag_width_center,diag_height_center)

    if show_info :
        print('Min Area : ', min_area_coor )
        print('Min (width,height) : ',min_wh_coor)
        print('-------------------------------------------------')
        print('Min width BBoxes : ',min_width_coor)
        print('Min height BBoxes : ',min_height_coor)
        print('-------------------------------------------------')
        print('Min diag width : ', diag_width_center)
        print('Min diag height : ', diag_height_center)
        print('-------------------------------------------------')
        print('Min diag length : ', min_diag_det_length)
        print('\n')
        
    return min_diag_det_length
