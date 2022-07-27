import pickle
import warnings
import numpy as np
import pandas as pd
from SlideRunner.dataAccess.database import Database

def ignore_warning(): return warnings.filterwarnings('ignore')

def load_slides_info(path, slide_list:list):

    database = Database()
    database.open(path)

    slides_df = pd.DataFrame(database.execute('''SELECT uid, filename, width, height
                                                 FROM Slides''').fetchall(),
                           columns=['slide', 'filename', 'width', 'height'])
    slides_df = slides_df[slides_df['filename'].isin(slide_list)]

    return slides_df

def load_database(path, slide_list:list, class_list:list=[4], fold=[1,2]): #Fold

    database = Database()
    database.open(path)
    
    query = '''SELECT ac.slide, s.width, s.height, s.filename, ac.annoId, a.agreedClass, c.name, ac.coordinateX, ac.coordinateY 
             FROM Annotations_coordinates ac LEFT JOIN Annotations a ON ac.annoId = a.uid 
             LEFT JOIN Classes c ON a.agreedClass = c.uid
             LEFT JOIN Slides s ON ac.slide = s.uid
             WHERE a.deleted = 0 and (a.agreedClass = 4 or a.agreedClass = 5)'''

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
    anno_df = anno_df.append(new_anno_neg[['filename', 'x_min', 'y_min', 'x_max', 'y_max', 'classname']])
    anno_df.loc[anno_df['class'].isna(), 'class'] = 5
    
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
    

def get_anno_boxes(anno_df) :
    coor_df = anno_df[['x_min', 'y_min', 'x_max', 'y_max','classname','annoID']].reset_index()
    coor_df = coor_df.loc[coor_df['classname'] == 'CMV']
    coor_df.drop(columns = ['index','classname'], inplace = True)
    anno_boxes = coor_df.to_numpy(copy = True) 
    return anno_boxes #[ [x1 y1 x2 y2 annoID],[...]]

def save_object(path, objects) :
    with open(path, 'wb') as file :
        pickle.dump(objects, file)

def load_object(path) :
    with open(path, 'rb') as file :
        objects = pickle.load(file)
        return objects
