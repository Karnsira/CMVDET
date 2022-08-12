import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from .nms_WSI import nms
from .utils_eval import load_database, get_anno_boxes
from .objectDetectionHelper import getMaxDetBoxDiagLength


#  ----------------------- Calculate Evaluation Metrics ----------------------- #

def get_metric(tp : int, fp : int , fn : int, beta = None) -> dict:
        
        try : 
            recall = tp/(tp + fn)
        except ZeroDivisionError :
            recall = 0
        try : 
            precision = tp/(tp + fp)
        except ZeroDivisionError :
            precision = 0
        if beta is not None :
            try : 
                f_score = ((1+(beta**2)) * ( (precision*recall))/ ( (beta**2)*precision + recall ) )
            except ZeroDivisionError :
                f_score = 0
        else :
            try : 
                f_score = 2*tp/(2*tp + fp + fn)
            except ZeroDivisionError :
                f_score = 0

        return {'f_score' : f_score , 'recall' : recall, 'precision' : precision}

def f_score(tp : int, fp : int , fn : int, beta = None) :
    return get_metric(tp, fp, fn, beta)['f_score']

def precision(tp : int, fp : int ) :
    return get_metric(tp, fp, 0)['precision']

def recall(tp : int , fn : int) :
    return get_metric(tp, 0, fn)['recall']




#  ----------------------- Evaluation Function For Detection Result ----------------------- #

def get_eval(databasefile, resfile=None, conf_thres=0.5, nms_thres=0.5, fold=[1,2], show_info = False):
    if (resfile is None):
        raise ValueError('At least one of resfile must be given')
    elif isinstance(resfile,dict) :
        result_boxes = resfile
    else : 
        with open(resfile, 'rb') as file :
            result_boxes = pickle.load(file) 
        if isinstance(result_boxes[list(result_boxes.keys())[0]], list) :
            result_boxes = {k : np.array(result_boxes[k], dtype = np.ndarray) for k in result_boxes}
    
    sTP, sFN, sFP = 0,0,0
    metric_dict = dict()
    sub_metric_dict = dict()
    
    result_boxes = nms(result_boxes, conf_thres, iou_thres=nms_thres)
    
    if show_info : print('Evaluating test set of %d files' % len(result_boxes))
    
    slide_arr = list(result_boxes.keys())
    anno_df = load_database(databasefile, slide_arr, fold=fold)

    for wsi in result_boxes :
        boxes = np.array(result_boxes[wsi])
        anno_boxes = get_anno_boxes(anno_df[anno_df['filename'] == wsi]) #[ [x1 y1 x2 y2 annoID],[...]
        TP, FP, FN = 0,0,0
        if boxes.shape[0]>0:
            score = boxes[:,-1]
            anno_ID = anno_boxes[:,-1]
            TP,FP,FN = eval_core(anno_boxes[:,:-1], boxes, score, conf_thres, anno_ID =anno_ID )
            sub_metric_dict[wsi] = [TP, FP, FN]
        else :
            anno_ID = anno_boxes[:,-1]
            TP,FP,FN = 0,0 ,len(anno_ID)
            sub_metric_dict[wsi] = [TP, FP, FN]
        sTP+=TP
        sFP+=FP
        sFN+=FN

    metric_dict['indiv'] = sub_metric_dict
    metric_dict['overall'] = [sTP, sFP, sFN]
    if show_info : 
        sF1 = f_score(sTP,sFP,sFN)
        print('Overall: ')
        print('TP:', sTP, 'FP:', sFP,'FN: ',sFN,'F1:',sF1)

    return metric_dict
 
def eval_core(anno_boxes,boxes, score , conf_thres = 0.5 ,  anno_ID = None) :
    np2list = lambda arr : [ list(i) for i in arr]

    anno_boxes_size = len(anno_boxes)
    to_keep = score>=conf_thres

    boxes_withScore = boxes[to_keep]
    boxes = boxes[to_keep][:,:-1]

    annoDet_coor = np.vstack((anno_boxes, boxes))

    x_anno_center = anno_boxes[:, 0] + (anno_boxes[:, 2] - anno_boxes[:, 0]) / 2
    y_anno_center = anno_boxes[:, 1] + (anno_boxes[:, 3] - anno_boxes[:, 1]) / 2

    x_det_center = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
    y_det_center = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

    A = np.dstack((x_anno_center,y_anno_center))[0]
    D = np.dstack((x_det_center,y_det_center))[0]
    X = np.vstack((A,D))

    used_box = set()
    
    tp = 0
    for anno_index in range(anno_boxes_size) :

        gt_box = annoDet_coor[anno_index]
        gx1, gy1, gx2 ,gy2 = gt_box
        if gx2 - gx1 < 5 or gy2 - gy1 < 5 : 
            print('Unusual Ground Truth Box Size (W,H) : ', (gx2-gx1,gy2-gy1))
            continue
            
        radius  = getMaxDetBoxDiagLength([annoDet_coor[anno_index]], show_info = False)

        try:
            tree = KDTree(X)
        except:
            print('Shapes of X: ',X.shape)

        ind, dist = tree.query_radius(X, r=radius,sort_results = True, return_distance = True)
        ind_anno = ind[:anno_boxes_size]
        
        anno_nn_idx = ind_anno[anno_index]
        anno_nn_idx = anno_nn_idx[anno_nn_idx >= anno_boxes_size] # Det boxes filter
        anno_nn_idx = [ idx-anno_boxes_size for idx in anno_nn_idx] # get Det boxes index
        anno_nn_idx = np.array([idx for idx in anno_nn_idx if idx not in used_box ]) # filter out used box

        if anno_nn_idx.size != 0 : #Have pred box around with radius
            det_boxes = boxes_withScore[anno_nn_idx] # get det bboxes around gt 
            tmp_dboxes = list(det_boxes)

            det_boxes = np.array(sorted(det_boxes, key = lambda x : x[-1], reverse=True)) #Sort by conf
            det_index = [ np2list(det_boxes).index(list(v)) for v in tmp_dboxes] # get sorted det index
            det_index = anno_nn_idx[det_index] 
            
            used_box.add(det_index[0])
            tp += 1

    fp = len(boxes) - tp
    fn = len(anno_boxes) - tp
    
    return tp ,fp, fn



#  ----------------------- Optimization Of Metrics ----------------------- #

def optimize_threshold(databasefile, resfile=None, fold = [1,2],
                        minthres=0.5,conf_thres=0.5, nms_thres=0.5,step =0.01,
                        metric = 'f_score', beta = 1, display=True):
    mode = {'f_score','precision','recall'}
    if metric not in mode :
        raise ValueError('Please select metric in following string : ', mode)

    if (resfile is None):
        raise ValueError('At least one of resfile must be given')
    elif isinstance(resfile,dict) :
        result_boxes = resfile
    else : 
        with open(resfile, 'rb') as file :
            result_boxes = pickle.load(file)
    
    
    MIN_THR = minthres
    result_boxes = nms(result_boxes, conf_thres, iou_thres=nms_thres)
    TPd, FPd, FNd, MVd = dict(), dict(), dict(), dict()
    thresholds = np.arange(MIN_THR,0.99,step)

    slide_arr = list(result_boxes.keys())
    anno_df = load_database(databasefile, slide_arr, fold=fold)
    
    print('Optimizing threshold for test set of %d files: '%len(result_boxes.keys()))

    for wsi in result_boxes:
        boxes = np.array(result_boxes[wsi])

        TP, FP, FN = 0,0,0
        TPd[wsi] = list()
        FPd[wsi] = list()
        FNd[wsi] = list()
        MVd[wsi] = list()

        if (boxes.shape[0]>0):
            score = boxes[:,-1]
         
            anno_boxes = get_anno_boxes(anno_df) #[ [x1 y1 x2 y2 annoID],[...]]
            anno_ID = anno_boxes[:,-1]

            for conf_thres in thresholds:
                TP,FP,FN = eval_core(anno_boxes[:,:-1], boxes, score,conf_thres, anno_ID =anno_ID )
                metric_value = get_metric(tp = TP, fp = FP, fn =FN, beta=beta)[metric]
                
                TPd[wsi] += [TP]
                FPd[wsi] += [FP]
                FNd[wsi] += [FN]
                MVd[wsi] += [metric_value]
        else:
            for conf_thres in thresholds:
                TPd[wsi] += [0]
                FPd[wsi] += [0]
                FNd[wsi] += [0]
                MVd[wsi] += [0]
        

    allTP = np.zeros(len(thresholds))
    allFP = np.zeros(len(thresholds))
    allFN = np.zeros(len(thresholds))
    allMV = np.zeros(len(thresholds))
    allF1M = np.zeros(len(thresholds))

    for k in range(len(thresholds)):
        allTP[k] = np.sum([TPd[x][k] for x in result_boxes])
        allFP[k] = np.sum([FPd[x][k] for x in result_boxes])
        allFN[k] = np.sum([FNd[x][k] for x in result_boxes])
        allMV[k] = get_metric(allTP[k], allFP[k], allFN[k], beta)[metric] #2*allTP[k] / (2*allTP[k] + allFP[k] + allFN[k])
        allF1M[k] = np.mean([MVd[x][k] for x in result_boxes])
        
    max_idx = np.argmax(allMV)
    
    if display :
        from matplotlib import pyplot as plt 
        plt.plot(thresholds, allMV)
        plt.xlabel('cutoff value for CMV detection')
        plt.ylabel(f'f{beta} score' if metric == 'f_score' else metric)
        #plt.title('Optimization of detection threshold (test set)')
        plt.plot([minthres, 1.0], np.max(np.array(allMV))*np.array([1,1]),'r--')
        plt.plot();
        
    return allMV[max_idx] , thresholds[max_idx], allMV, thresholds


#  ----------------------- Export Boxes To 2nd Stage ----------------------- #

def categorize_pred_boxes(databasefile, resfile=None, conf_thres=0.5, nms_thres=0.5, fold=[1,2], mode = 'actual'):
    if mode not in {'actual', 'eval'} :
        raise ValueError('Please select mode in following string : ', {'actual', 'eval'})
    
    if (resfile is None):
        raise ValueError('At least one of resfile must be given')
    elif isinstance(resfile,dict) :
        result_boxes = resfile
    else : 
        with open(resfile, 'rb') as file :
            result_boxes = pickle.load(file)

    categorized_boxes = dict()
    result_boxes = nms(result_boxes, conf_thres, iou_thres=nms_thres)

    slide_arr = list(result_boxes.keys())
    anno_df = load_database(databasefile, slide_arr, fold=fold)

    for wsi in result_boxes :
        boxes = np.array(result_boxes[wsi])
        anno_boxes = get_anno_boxes(anno_df[anno_df['filename'] == wsi]) #[ [x1 y1 x2 y2 annoID],[...]]
        if boxes.shape[0]>0:
            score = boxes[:,-1]
            cat_box = categorize_core(anno_boxes[:,:-1], boxes, score, conf_thres, mode)
            categorized_boxes[wsi] = cat_box
        else :
            cat_box = {'tp' : np.array([]), 'fp' : np.array([]), 'fn': anno_boxes[:,:-1]}
            categorized_boxes[wsi] = cat_box
    return categorized_boxes
 

def categorize_core(anno_boxes, boxes, score , conf_thres = 0.5, mode = 'actual') :
    np2list = lambda arr : [ list(i) for i in arr]

    anno_boxes_size = len(anno_boxes)
    to_keep = score>=conf_thres

    boxes_withScore = boxes[to_keep]
    boxes = boxes_withScore[:,:-1]

    annoDet_coor = np.vstack((anno_boxes, boxes))

    x_anno_center = anno_boxes[:, 0] + (anno_boxes[:, 2] - anno_boxes[:, 0]) / 2
    y_anno_center = anno_boxes[:, 1] + (anno_boxes[:, 3] - anno_boxes[:, 1]) / 2

    x_det_center = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
    y_det_center = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

    A = np.dstack((x_anno_center,y_anno_center))[0]
    D = np.dstack((x_det_center,y_det_center))[0]
    X = np.vstack((A,D))

    sol_box = set()
    used_box = set()
    exclude_box = set()
    
    tp = 0
    for anno_index in range(anno_boxes_size) :

        gt_box = annoDet_coor[anno_index]
        gx1, gy1, gx2 ,gy2 = gt_box
        if gx2 - gx1 < 5 or gy2 - gy1 < 5 : 
            print('Unusual Ground Truth Box Size (W,H) : ', (gx2-gx1,gy2-gy1))
            continue
            
        radius  = getMaxDetBoxDiagLength([annoDet_coor[anno_index]], show_info = False)

        try:
            tree = KDTree(X)
        except:
            print('Shapes of X: ',X.shape)

        ind, dist = tree.query_radius(X, r=radius,sort_results = True, return_distance = True)
        ind_anno = ind[:anno_boxes_size]
        
        anno_nn_idx = ind_anno[anno_index]
        anno_nn_idx = anno_nn_idx[anno_nn_idx >= anno_boxes_size] # Det boxes filter
        anno_nn_idx = [ idx-anno_boxes_size for idx in anno_nn_idx] # get Det boxes index
        anno_nn_idx = np.array([idx for idx in anno_nn_idx if idx not in used_box ]) # filter out used box

        if anno_nn_idx.size != 0 : #Have pred box around with radius
            det_boxes = boxes_withScore[anno_nn_idx] # get det bboxes around gt 
            tmp_dboxes = list(det_boxes)

            det_boxes = np.array(sorted(det_boxes, key = lambda x : x[-1], reverse=True)) #Sort by conf
            det_index = [ np2list(det_boxes).index(list(v)) for v in tmp_dboxes] # get sorted det index
            det_index = anno_nn_idx[det_index] 

            sol_box.add(anno_index)
            used_box.add(det_index[0])
            exclude_box = exclude_box.union(set(det_index))
            tp += 1

    if mode == 'eval' :
        exclude_box = used_box
            
    tp_boxes = np.array([boxes_withScore[i] for i in used_box])
    fp_boxes = np.array([boxes_withScore[i] for i in range(len(boxes_withScore)) if i not in exclude_box ])
    fn_boxes = np.array([ anno_boxes[i] for i in range(anno_boxes_size) if i not in sol_box])

    result_boxes = {'tp' : tp_boxes, 'fp' : fp_boxes, 'fn' : fn_boxes}
   
    return result_boxes

#  ----------------------- Show Evaluation Result Function ----------------------- #


def show_eval_results(databasefile : str, fold = [1,2],mode = 'all', top_k = 10, resfile = None, conf_thres= 0, nms_thres=0.5) :
    
    def hitl(metric, actual) : 
        tp = metric[0]
        if tp > 0 or (tp == 0 and actual == 0) : return 1
        return 0
    
    def full_ai(metric, actual) : 
        tp = metric[0]
        fp = metric[1]
        if tp > 0 or (tp==0 and actual == 0 and fp == 0) : return 1
        return 0

    metric_dict = get_eval(databasefile=databasefile, resfile=resfile, fold=fold,
                           conf_thres=conf_thres, nms_thres=nms_thres, show_info=False)
    metric_dict = { k : metric_dict[k] for k in metric_dict}
    metric_df = pd.DataFrame([k for k in metric_dict['indiv']], columns = ['Slide Name'] )


    metric_df['TP'] = [int(metric_dict['indiv'][k][0]) for k in  metric_dict['indiv']]
    metric_df['FP'] = [int(metric_dict['indiv'][k][1]) for k in  metric_dict['indiv']]
    metric_df['FN'] = [int(metric_dict['indiv'][k][2]) for k in  metric_dict['indiv']]
    metric_df['F1'] = f_score(metric_df['TP'],metric_df['FP'],metric_df['FN'])
    metric_df['F2'] = f_score(metric_df['TP'],metric_df['FP'],metric_df['FN'], beta=2)
    
    metric_df['Recall'] = recall(metric_df['TP'],metric_df['FN']).round(2)
    metric_df['Precision'] = recall(metric_df['TP'],metric_df['FP']).round(2)
    metric_df['F1'] = metric_df['F1'].round(2)*100
    metric_df['F2'] = metric_df['F2'].round(2)*100

    metric_df.insert(loc = 1 , column =  'Actual' , value =  metric_df['TP']+metric_df['FN'] )

    sumSlide = 'All Slide'
    sumRTP = sum(metric_df['Actual'].to_numpy())
    sumTP = metric_dict['overall'][0]
    sumFP = metric_dict['overall'][1]
    sumFN = metric_dict['overall'][2]
    sumF1 = round(f_score(sumTP, sumFP, sumFN),2 )*100
    sumF2 = round(f_score(sumTP, sumFP, sumFN, beta=2),2 )*100
    sumRecall = round(recall(sumTP, sumFN), 2 )*100
    sumPrecision = round(precision(sumTP, sumFP), 2)*100

    extra_column = list()
    extra_sum = list()

    if resfile is not None :
        if isinstance(resfile, str)  :
            with open(resfile, 'rb') as file :
                pred_boxes = pickle.load(file)
        elif isinstance(resfile, dict) :
            pred_boxes = resfile
        else :
            raise ValueError('Please give resfile in form of dictionary or path(string) ')
        
        sort_by_conf = lambda arr, k : np.array(sorted(arr[k], key = lambda x : x[-1], reverse=True))
        pred_boxes = { k : sort_by_conf(pred_boxes,k)[:top_k] for k in pred_boxes}
        metric_top_k = get_eval(databasefile, pred_boxes, conf_thres=conf_thres , nms_thres=nms_thres)
        metric_df['HITL'] = [hitl(metric = metric_top_k['indiv'][k], actual = metric_df['Actual'][i]) for i,k in enumerate(pred_boxes)]
        metric_df['Full-AI'] = [full_ai(metric = metric_top_k['indiv'][k], actual = metric_df['Actual'][i]) for i,k in enumerate(pred_boxes)]
        
        extra_column = ['HITL', 'Full-AI']
        slide_num = len(pred_boxes)
        extra_sum = [ f'{sum(metric_df["HITL"])} / {slide_num}', f'{sum(metric_df["Full-AI"])} / {slide_num}']
    
    sum_df = pd.DataFrame([[sumSlide, sumRTP, sumTP, sumFP, sumFN, sumF1, sumF2, sumRecall, sumPrecision]+extra_sum] ,
                      columns = ['Slide Name', 'Actual', 'TP',  'FP', 'FN', 'F1', 'F2', 'Recall', 'Precision']+extra_column)
    
    if mode == 'sum' : return sum_df
    
    final_sum_df = metric_df.copy(deep = True)
    space_df = pd.DataFrame([['' for _ in final_sum_df.columns]], columns = final_sum_df.columns )
    final_sum_df = final_sum_df.append(space_df, ignore_index=True)
    final_sum_df = final_sum_df.append(sum_df, ignore_index=True)
    final_sum_df
    
    return final_sum_df


#  ----------------------- Generate tp/fp/fn  ----------------------- #


def generate_boxes(databasefile, metric = 'fp', resfile=None, conf_thres=0.5, nms_thres=0.5, fold=[1,2], mode='actual') :
    if metric not in {'tp', 'fp', 'fn'} :
        raise ValueError('Please select metric in following string : ',  {'tp', 'fp', 'fn'})

    cat_boxes = categorize_pred_boxes(databasefile, resfile=resfile, conf_thres=conf_thres, nms_thres=nms_thres, fold=fold, mode =mode)
    cat_boxes = { wsi : cat_boxes[wsi][metric]  for wsi in cat_boxes}
    return cat_boxes
