import os
from lib.utils_det import load_database, generate_patches_cmv, generate_patches_hard, generate_patches_bg, train_val_split
import argparse
from lib.utils import DatabaseConfig

parser = argparse.ArgumentParser()
parser.add_argument('--output', help='output folder path')
parser.add_argument('--wsi', help='WSI folder path', default='/data/WSI/')
parser.add_argument('--db', help='database file path', default='/data/database/CU_DB.sqlite')
parser.add_argument('--img-size', help='patch size', default=224)
parser.add_argument('--img-repeat', help='number of repeated patches', default=10)
parser.add_argument('--reviewed', help='number of repeated patches', default="/data/database/reviewed.csv")

args = parser.parse_args()

train_slide = ['S20-32081 B1.svs']#, 'SP63-22003 A.svs']
val_slide = ['S20-32081 D1.svs']
ganglion_slide = ['SP65-2246 B.svs']
negative_slide = []

db_path = args.db
reviewed = args.reviewed
WSI_path = args.wsi

save_to = args.output
repeat = int(args.img_repeat)
size = int(args.img_size)

generate_patches_cmv(size=size, 
                     WSI_path=WSI_path, 
                     db_config=DatabaseConfig(path=db_path, slide_list=train_slide+val_slide, reviewed=reviewed), 
                     save_patches_to=save_to, 
                     save_patches_anno_to=save_to, 
                     repeat=repeat)

cnt = generate_patches_hard(size=size,
                            WSI_path=WSI_path,
                            db_config=DatabaseConfig(path=db_path, slide_list=ganglion_slide, manual=True, class_list=[0,5,6], reviewed=reviewed), 
                            save_patches_to=save_to,
                            start_idx=0,
                            repeat=repeat,
                            )

cnt = generate_patches_bg(size=size,
                          WSI_path=WSI_path,
                          db_config=DatabaseConfig(path=db_path, slide_list=train_slide, reviewed=reviewed), 
                          save_patches_to=save_to,
                          number_per_wsi=10,
                          color_thres=230,
                          start_idx=cnt)

train_val_split(annotations_path=save_to+'annotations.csv', 
                slide_val=val_slide, 
                num_train=(len(load_database(db_config = DatabaseConfig(path=db_path, slide_list=train_slide, reviewed=reviewed)))*repeat)-(len(load_database(db_config=DatabaseConfig(path=db_path, slide_list=val_slide, reviewed=reviewed)))*repeat), 
                num_val=len(load_database(DatabaseConfig(path=db_path, slide_list=val_slide, reviewed=reviewed)))*repeat,
                neg_per_pos=1,
                save_to=save_to)