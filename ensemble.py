import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--det', help='result from detection stage in pickle format')
parser.add_argument('--cls', help='result from classification stage in pickle format')
parser.add_argument('--output', help='output file in pickle format')
parser.add_argument('--det-weight', help='detection weight (classification weight = 1 - detection weight')

args = parser.parse_args()

det_path = args.det
cls_path = args.cls
ens_path = args.output

det_weight = float(args.det_weight)
cls_weight = 1-det_weight

def ensemble(det_pkl:str, cls_pkl:str, ens_pkl:str, det_weight:float, cls_weight:float):
    
    with open(det_pkl, 'rb') as pkl:
        det_res = pickle.load(pkl)
        
    with open(cls_pkl, 'rb') as pkl:
        cls_res = pickle.load(pkl)
        
    ens_res = {}
    
    for slidename in tqdm(det_res):           
        det_res_slide = det_res[slidename]
        cls_res_slide = cls_res[slidename]
        ens_res_slide = det_res_slide
        
        for i in range(len(det_res_slide)):
            ens_res_slide[i][4] = det_weight*det_res_slide[i][4] + cls_weight*cls_res_slide[i][4]
        
        ens_res[slidename] = ens_res_slide
        
    with open(ens_pkl, 'wb') as pkl:
        pickle.dump(ens_res, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        
    return ens_res


ensemble(det_pkl=det_path, 
         cls_pkl=cls_path, 
         ens_pkl=ens_path, 
         det_weight=det_weight, 
         cls_weight=cls_weight)