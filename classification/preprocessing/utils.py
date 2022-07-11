from SlideRunner.dataAccess.database import Database
import sqlite3
import openslide
import pandas as pd
import numpy as np
from random import randrange, shuffle, uniform
from tqdm import tqdm
from PIL import Image, ImageEnhance
import pickle
import sys
sys.path.append('')
from evaluation.lib.evaluation import generate_boxes


def load_database(path, slide_list:list, class_list:list=[4], fold=[1,2], manual=False):

    database = Database()
    database.open(path)
    
    query = '''SELECT ac.slide, s.width, s.height, s.filename, ac.annoId, a.agreedClass, c.name, ac.coordinateX, ac.coordinateY 
             FROM Annotations_coordinates ac LEFT JOIN Annotations a ON ac.annoId = a.uid 
             LEFT JOIN Classes c ON a.agreedClass = c.uid
             LEFT JOIN Slides s ON ac.slide = s.uid
             WHERE a.deleted = 0 and (a.agreedClass = 4 or a.agreedClass = 5 or a.agreedClass = 6)'''

    anno_df = pd.DataFrame(database.execute(query).fetchall(), 
                        columns=['slide', 'width', 'height', 'filename', 'annoID', 'class', 'classname', 'coordinateX', 'coordinateY'])

    anno_df['x_max'] = anno_df.groupby('annoID')['coordinateX'].transform('last')
    anno_df['y_max'] = anno_df.groupby('annoID')['coordinateY'].transform('last')
    anno_df['count'] = anno_df.groupby('annoID')['slide'].transform(len)
    anno_df = anno_df[anno_df['count']==2].copy()
    anno_df.drop('count', axis=1, inplace=True)
    anno_df = anno_df.groupby('annoID').head(1)
    anno_df.columns = ['slide', 'width', 'height', 'filename', 'annoID', 'class', 'classname', 'x_min', 'y_min', 'x_max', 'y_max']
    anno_df[['x_min', 'y_min','x_max','y_max']] = anno_df[['x_min', 'y_min','x_max','y_max']].astype(int)
    anno_df = anno_df[anno_df['filename'].isin(slide_list)]
    anno_df.reset_index(inplace=True)
    anno_df.drop('index', axis=1, inplace=True)
    
    new_anno = pd.read_csv('/data/database/reviewed.csv')
    new_anno_cmv = new_anno[(new_anno['classname']=='CMV') & (new_anno['filename'].isin(slide_list))]
    anno_df = anno_df.append(new_anno_cmv[['filename', 'x_min', 'y_min', 'x_max', 'y_max', 'classname']])
    anno_df.loc[anno_df['class'].isna(), 'class'] = 4
    
    new_anno_neg = new_anno[(new_anno['classname']=='Negative') & (new_anno['filename'].isin(slide_list))]
    
    if manual == False:
        new_anno_neg = new_anno_neg[new_anno_neg['timestamp'].str[:10]!='2022-03-20']
        
    anno_df = anno_df.append(new_anno_neg[['filename', 'x_min', 'y_min', 'x_max', 'y_max', 'classname']])
    anno_df.loc[anno_df['class'].isna(), 'class'] = 0
    
    slide_info = load_slides_info(path=path, slide_list=slide_list)
    for slidename in slide_list:
        anno_df.loc[(anno_df['slide'].isna()) & (anno_df['filename']==slidename), ['slide', 'width', 'height']] = slide_info[slide_info['filename']==slidename][['slide', 'width', 'height']].values[0]
    
    #filter out wrong annotations
    anno_df['ratio'] = (anno_df['x_max']-anno_df['x_min']) / (anno_df['y_max']-anno_df['y_min'])
    anno_df = anno_df[(anno_df['ratio']!=0) & (anno_df['ratio']!=np.inf)]
    anno_df.drop('ratio', axis=1, inplace=True)
    anno_df.reset_index(inplace=True)
    anno_df.drop('index', axis=1, inplace=True)
    anno_df = anno_df[anno_df['class'].isin(class_list)]
    anno_df[['slide', 'width', 'height', 'annoID','class']] = anno_df[['slide', 'width', 'height', 'annoID','class']].astype(pd.Int64Dtype())
    
    anno_df['q1'] = anno_df['height']*0.25
    anno_df['q2'] = anno_df['height']*0.5
    anno_df['q3'] = anno_df['height']*0.75
    
    anno_df.loc[(anno_df['y_max']<anno_df['q1']) | ((anno_df['y_max']>anno_df['q2']) & (anno_df['y_max']<anno_df['q3'])), 'fold'] = 1
    anno_df.loc[(anno_df['y_max']>anno_df['q3']) | ((anno_df['y_max']<anno_df['q2']) & (anno_df['y_max']>anno_df['q1'])), 'fold'] = 2
    anno_df['fold'] = anno_df['fold'].astype(pd.Int64Dtype())
    anno_df = anno_df[anno_df['fold'].isin(fold)]

    return anno_df


def load_slides_info(path, slide_list:list):

    database = Database()
    database.open(path)

    slides_df = pd.DataFrame(database.execute('''SELECT uid, filename, width, height
                                                 FROM Slides''').fetchall(),
                           columns=['slide', 'filename', 'width', 'height'])
    slides_df = slides_df[slides_df['filename'].isin(slide_list)]

    return slides_df


def generate_patches_cmv(size:int, db_path:str, WSI_path:str, slide_list:list, save_patches_to:str, save_patches_anno_to:str, fold=[1,2], repeat=10, fit=True):
    
    anno_df = load_database(db_path, slide_list, fold=fold)

    class_df = anno_df[anno_df['classname']=='CMV']

    patches_data = []
    cnt = 0
    real_size = size
    for filename in tqdm(class_df['filename'].unique(), desc='Generating CMV patches'):

        slide_df = class_df[class_df['filename']==filename]

        slide =  openslide.OpenSlide(WSI_path+filename)

        for i in range(len(slide_df)):

            x_min, y_min, x_max, y_max = slide_df[['x_min','y_min','x_max','y_max']].values[i]

            anno_width = x_max - x_min
            anno_height = y_max - y_min
            
            if fit:
                size = int(max(anno_width, anno_height)) + int(max(anno_width, anno_height)/2)

            x_center = x_min + anno_width/2
            y_center = y_min + anno_height/2

            patch_x_origin = int(x_center - size/2)
            patch_y_origin = int(y_center - size/2)

            x_min_rescaled = int(size/2 - anno_width/2)
            x_max_rescaled = int(size/2 + anno_width/2)
            y_min_rescaled = int(size/2 - anno_height/2)
            y_max_rescaled = int(size/2 + anno_height/2)
            
            x_move = 0
            y_move = 0

            # change from wsi coor to patch coor
            x_change = x_min-(x_min_rescaled+x_move)
            y_change = y_min-(y_min_rescaled+y_move)

            for n in range(repeat):
                cnt += 1
                patchname = 'cmv'+str(cnt)+'.png'
                patch = slide.read_region((patch_x_origin-x_move, patch_y_origin-y_move), 0, (size, size)).convert('RGB')
                patch = patch.resize((real_size, real_size), Image.BICUBIC)

                br = uniform(0.6,1.4)
                cl = uniform(0.6,1.4)
                patch = ImageEnhance.Color(ImageEnhance.Brightness(patch).enhance(br)).enhance(cl)
                
                patch.save(save_patches_to+'/'+patchname)
                patches_data.append([filename, patchname, slide_df['annoID'].values[i], 'CMV'])

    patches_df = pd.DataFrame(patches_data, columns=['slidename','filename', 'annoID', 'class'])
    patches_df.to_csv(save_patches_anno_to+'/annotations.csv', index=False)
    #print('***Total',cnt,'patches***')
    return cnt
    
    
def generate_patches_bg(size:int, db_path:str, WSI_path:str, slide_list:list, save_patches_to:str, number_per_wsi:int, color_thres:int, start_idx:int=0, fold=[1,2], repeat=10):
  
    anno_df = load_database(db_path, slide_list)
    class_df = anno_df[anno_df['classname']=='CMV']

    cnt = start_idx
    for filename in tqdm(anno_df['filename'].unique(), desc='Generating Background patches'):
        
        slide_df = class_df[class_df['filename']==filename]

        width = anno_df[anno_df['filename']==filename]['width'].values[0]
        height = anno_df[anno_df['filename']==filename]['height'].values[0]
        
        if fold == [1]:
            foldr1 = range(0, int(anno_df['q1'].values[0]))
            foldr2 = range(int(anno_df['q2'].values[0]), int(anno_df['q3'].values[0]))
        elif fold == [2]:
            foldr1 = range(int(anno_df['q1'].values[0]), int(anno_df['q2'].values[0]))
            foldr2 = range(int(anno_df['q3'].values[0]), height)
        else:
            foldr1 = range(0, height)
            foldr2 = range(0, height)

        slide =  openslide.OpenSlide(WSI_path+filename)

        n_patches_per_wsi = 0
        while n_patches_per_wsi != number_per_wsi:
            while True:
                patch_x_origin = randrange(0, width-size)
                patch_y_origin = randrange(0, height-size)
                if (patch_y_origin in foldr1) | (patch_y_origin in foldr2):
                    break

            patch = slide.read_region((patch_x_origin, patch_y_origin), 0, (size, size)).convert('RGB')
            
            if np.array(patch).mean() > color_thres:    # check if white background -> skip
                continue

            patch_data = find_anno_in_patch(slide_df, filename, 'bg', size, (patch_x_origin, patch_y_origin), (0,0), (0,0))

            if len(patch_data) != 0:    # if has annotations -> skip
                continue
            
            for i in range(repeat):
                cnt += 1
                patchname = 'bg'+str(cnt)+'.png'

                br = uniform(0.6,1.4)
                cl = uniform(0.6,1.4)
                patch = ImageEnhance.Color(ImageEnhance.Brightness(patch).enhance(br)).enhance(cl)

                patch.save(save_patches_to+'/'+patchname)
                n_patches_per_wsi += 1
            
    #print('***Total',cnt,'patches***')
    return cnt
    
    
def generate_patches_hard(size:int, db_path:str, WSI_path:str, slide_list:list, save_patches_to:str, start_idx:int=0, fold=[1,2], repeat=10, manual=True, class_list=[0,5,6], fit=True):
    
    # class: 0 -> negative (not specific type) | 5 -> ganglion cells | 6 -> melanoma cells
    anno_df = load_database(db_path, slide_list, class_list=class_list, fold=fold, manual=manual)

    cnt = start_idx
    real_size = size
    for filename in tqdm(anno_df['filename'].unique(), desc='Generating Hard Example patches'):
        
        slide_df = anno_df[anno_df['filename']==filename]

        slide =  openslide.OpenSlide(WSI_path+filename)

        for i in range(len(slide_df)):

            x_min, y_min, x_max, y_max = slide_df[['x_min','y_min','x_max','y_max']].values[i]

            anno_width = x_max - x_min
            anno_height = y_max - y_min
            
            if fit:
                size = int(max(anno_width, anno_height)) + int(max(anno_width, anno_height)/2)

            x_center = x_min + anno_width/2
            y_center = y_min + anno_height/2

            patch_x_origin = int(x_center - size/2)
            patch_y_origin = int(y_center - size/2)

            x_min_rescaled = int(size/2 - anno_width/2)
            x_max_rescaled = int(size/2 + anno_width/2)
            y_min_rescaled = int(size/2 - anno_height/2)
            y_max_rescaled = int(size/2 + anno_height/2)
            
            x_move = 0
            y_move = 0

            # change from wsi coor to patch coor
            x_change = x_min-(x_min_rescaled+x_move)
            y_change = y_min-(y_min_rescaled+y_move)
                
            for i in range(repeat):
                cnt += 1
                patchname = 'bg'+str(cnt)+'.png'
                patch = slide.read_region((patch_x_origin-x_move, patch_y_origin-y_move), 0, (size, size)).convert('RGB')
                patch = patch.resize((real_size, real_size), Image.BICUBIC)

                br = uniform(0.6,1.4)
                cl = uniform(0.6,1.4)
                patch = ImageEnhance.Color(ImageEnhance.Brightness(patch).enhance(br)).enhance(cl)
                
                patch.save(save_patches_to+'/'+patchname)

    #print('***Total',cnt,'patches***')
    return cnt
    
    
def generate_patches_fp(size:int, res_path:str, WSI_path:str, db_path:str, save_patches_to:str , slide_list:list, repeat=10, fit=True, start_idx:int=0, conf_thres:int=0.5, det_thres:int=0, nms_thres:int=0.5):
    
    fp_dict = generate_boxes(db_path, metric = 'fp', resfile=res_path, conf_thres=det_thres, nms_thres=nms_thres, fold=[1,2], mode='actual')
    
    low_conf_fp = {}
    for slide in fp_dict:
        if slide not in slide_list:
            continue
        for i in fp_dict[slide]:
            if i[4] < conf_thres:
                if slide not in low_conf_fp:
                    low_conf_fp[slide]=[]
                low_conf_fp[slide].append(i)
    fp_dict = low_conf_fp
   
    cnt = start_idx
    real_size = size
    for slidename in tqdm(fp_dict, desc='Generating False Postive patches'):

        fp_coor = fp_dict[slidename]
        
        slide =  openslide.OpenSlide(WSI_path+slidename)
        
        for i, coor in enumerate(fp_coor):
            
            x_min, y_min, x_max, y_max = coor[:4]
            
            anno_width = x_max - x_min
            anno_height = y_max - y_min
            
            if fit:
                size = int(max(anno_width, anno_height)) + int(max(anno_width, anno_height)/2)

            x_center = x_min + anno_width/2
            y_center = y_min + anno_height/2

            patch_x_origin = int(x_center - size/2)
            patch_y_origin = int(y_center - size/2)

            x_min_rescaled = int(size/2 - anno_width/2)
            x_max_rescaled = int(size/2 + anno_width/2)
            y_min_rescaled = int(size/2 - anno_height/2)
            y_max_rescaled = int(size/2 + anno_height/2)

            x_move = 0
            y_move = 0

            # change from wsi coor to patch coor
            x_change = x_min-(x_min_rescaled+x_move)
            y_change = y_min-(y_min_rescaled+y_move)
            
            for i in range(repeat):
                cnt += 1
                patchname = 'bg'+str(cnt)+'.png'
                patch = slide.read_region((patch_x_origin-x_move, patch_y_origin-y_move), 0, (size, size)).convert('RGB')
                patch = patch.resize((real_size, real_size), Image.BICUBIC)

                br = uniform(0.6,1.4)
                cl = uniform(0.6,1.4)
                patch = ImageEnhance.Color(ImageEnhance.Brightness(patch).enhance(br)).enhance(cl)
                
                patch.save(save_patches_to+'/'+patchname)

    #print('***Total',patch_cnt,'patches***')
    return cnt
    
    
def find_anno_in_patch(slide_df, filename, patchname, size, patch_origin, patch_move, coor_change):

    patch_x_origin, patch_y_origin = patch_origin
    x_move, y_move = patch_move
    x_change, y_change = coor_change
    s = filename

    patch_data = []
    for i in range(len(slide_df)):

        x_min_cand, y_min_cand, x_max_cand, y_max_cand = slide_df[['x_min','y_min','x_max','y_max']].values[i]

        anno_width_cand = x_max_cand - x_min_cand
        anno_height_cand = y_max_cand - y_min_cand

        min_area = 0.5
        anno_width_cand_pct = (1-min_area) * anno_width_cand
        anno_height_cand_pct = (1-min_area) * anno_height_cand

        patch_x_min = patch_x_origin - x_move - anno_width_cand_pct
        patch_y_min = patch_y_origin - y_move - anno_height_cand_pct
        patch_x_max = patch_x_origin - x_move + size + anno_width_cand_pct
        patch_y_max = patch_y_origin - y_move + size + anno_height_cand_pct

        if (x_min_cand > patch_x_min and x_max_cand < patch_x_max and y_min_cand > patch_y_min and y_max_cand < patch_y_max):

            x_min_cand_rescaled = x_min_cand-x_change
            y_min_cand_rescaled = y_min_cand-y_change
            x_max_cand_rescaled = (x_min_cand-x_change)+anno_width_cand
            y_max_cand_rescaled = (y_min_cand-y_change)+anno_height_cand

            row = [s, patchname, slide_df['annoID'].values[i], size, size, x_min_cand_rescaled, y_min_cand_rescaled, x_max_cand_rescaled, y_max_cand_rescaled,'CMV']
            patch_data.append(row)

    return patch_data
    
    
def train_val_split(annotations_path:str, slide_val:list, num_train:int, num_val:int, neg_per_pos:int, save_to:str):
    
    df = pd.read_csv(annotations_path)
    
    trainlist = df[~df['slidename'].isin(slide_val)]['filename'].unique()[:num_train]
    vallist = df[df['slidename'].isin(slide_val)]['filename'].unique()[:num_val]
    
    bglist = ['bg'+str(i+1)+'.png' for i in range(neg_per_pos*(num_train+num_val))]
    shuffle(bglist)

    trainlist = list(trainlist) + list(bglist[:neg_per_pos*num_train])
    vallist = list(vallist) + list(bglist[neg_per_pos*num_train:])
    shuffle(trainlist)
    shuffle(vallist)

    print('train:',len(trainlist), 'val:', len(vallist))

    with open(save_to+'_train_.txt', 'w') as f:
        for i in trainlist:
            f.write(i+'\n')

    with open(save_to+'_val_.txt', 'w') as f:
        for i in vallist:
            f.write(i+'\n')