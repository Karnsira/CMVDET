import pickle
import numpy as np
import pandas as pd
import sqlite3
import sys
from SlideRunner.dataAccess.database import Database
from lib.evaluation import generate_boxes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file path from detection stage in pickle format')
parser.add_argument('--output', help='output file path in sqlite format')
parser.add_argument('--db', help='database file path', default='/data/database/CU_DB.sqlite')
parser.add_argument('--top', help='top-k false positive for each WSI', default=10)
parser.add_argument('--conf-thres', help='confidence threshold', default=0.5)

args = parser.parse_args()

input_path = args.input
output_path = args.output
db_path = args.db
conf_thres = float(args.conf_thres)
top = int(args.top)

fp = generate_boxes(db_path, metric = 'fp', resfile=input_path, conf_thres=conf_thres, nms_thres=1, fold=[1,2], mode='actual')

fp_sorted = {k:np.array(sorted(fp[k], key=lambda x:x[-1], reverse=True)) for k in fp}
fp_filtered = {k:fp_sorted[k][fp_sorted[k][:,-1] >= conf_thres] if len(fp_sorted[k])>0 else [] for k in fp_sorted}
fp_top = {k:fp_filtered[k][:top] if len(fp_sorted[k])>0 else [] for k in fp_filtered}

def load_slides_info(path, slide_list:list):
    database = Database()
    database.open(path)
    slides_df = pd.DataFrame(database.execute('''SELECT uid, filename, width, height
                                                 FROM Slides''').fetchall(),
                             columns=['slide', 'filename', 'width', 'height'])
    slides_df = slides_df[slides_df['filename'].isin(slide_list)]
    return slides_df

slideinfo = load_slides_info(db_path, fp_top.keys())

cnt = 0
annotations = []
annotations_coordinates = []
annotations_label = []
classes = []
log = []
person = []
slides = []

for i in fp_top:
    
    slideid = slideinfo[slideinfo['filename']==i]['slide'].values[0]
    width = slideinfo[slideinfo['filename']==i]['width'].values[0]
    height = slideinfo[slideinfo['filename']==i]['height'].values[0]
    
    for j in fp_top[i]:
        cnt += 1
        agreedClass = 1
        annotations.append([slideid, 2, agreedClass])
        
        x_min, y_min, x_max, y_max = j[0:-1]
        annotations_coordinates.append([int(x_min), int(y_min), slideid, cnt, 1])
        annotations_coordinates.append([int(x_max), int(y_max), slideid, cnt, 2])
        
        annotations_label.append([1, agreedClass, cnt])
        
        log.append([pd.Timestamp.now(), cnt])
    
    slides.append([slideid, i, width, height, ''])
        
classes.append(['FP', '#5998e1'])
person.append(['AI'])

annotations = pd.DataFrame(annotations, columns=['slide', 'type', 'agreedClass'])
annotations_coordinates = pd.DataFrame(annotations_coordinates, columns=['coordinateX', 'coordinateY', 'slide', 'annoId', 'orderIdx'])
annotations_label = pd.DataFrame(annotations_label, columns=['person', 'class', 'annoId'])
classes = pd.DataFrame(classes, columns=['name', 'color'])
log = pd.DataFrame(log, columns=['dateTime', 'labelId'])
log['dateTime'] = log['dateTime'].dt.strftime('%Y%m%d').astype(float)
person = pd.DataFrame(person, columns=['name'])
slides = pd.DataFrame(slides, columns=['uid', 'filename', 'width', 'height', 'directory'])

conn = sqlite3.connect(output_path)

conn.execute('CREATE TABLE Annotations (uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, slide INTEGER, guid TEXT, lastModified REAL DEFAULT 1604222545.4657702, deleted INTEGER DEFAULT 0, type INTEGER, agreedClass INTEGER, description TEXT, clickable INTEGER DEFAULT 1)')
conn.execute('CREATE TABLE Annotations_coordinates (uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, coordinateX INTEGER, coordinateY INTEGER, coordinateZ INTEGER DEFAULT 0, slide INTEGER, annoId INTEGER, orderIdx INTEGER)')
conn.execute('CREATE TABLE Annotations_label (uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, person INTEGER, class INTEGER, exact_id INTEGER, annoId INTEGER)')
conn.execute('CREATE TABLE Classes (uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, name TEXT, color TEXT)')
conn.execute('CREATE TABLE Log (uid INTEGER PRIMARY KEY AUTOINCREMENT, dateTime FLOAT, labelId INTEGER)')
conn.execute('CREATE TABLE Persons (uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, name TEXT, isExactUser INTEGER DEFAULT 0)')
conn.execute('CREATE TABLE Slides (uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, filename TEXT, width INTEGER, height INTEGER, directory TEXT, uuid TEXT, exactImageID TEXT, EXACTUSER INTEGER DEFAULT 0)')

annotations.to_sql('Annotations', conn, if_exists='append', index=False)
annotations_coordinates.to_sql('Annotations_coordinates', conn, if_exists='append', index=False)
annotations_label.to_sql('Annotations_label', conn, if_exists='append', index=False)
classes.to_sql('Classes', conn, if_exists='append', index=False)
log.to_sql('Log', conn, if_exists='append', index=False)
person.to_sql('Persons', conn, if_exists='append', index=False)
slides.to_sql('Slides', conn, if_exists='append', index=False)