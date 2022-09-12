import pickle
import geojson


class DatabaseConfig : 
    def __init__(self, path=None, slide_list=None, class_list=[4], fold=[1,2], manual=None, reviewed=None) -> None:
        self.path = path
        self.slide_list = slide_list
        self.class_list = class_list
        self.fold = fold
        self.manual = manual
        self.reviewed = reviewed

    def __str__(self) -> str:
        string = ("path : %s \n"
                  "slide_list : %s \n"
                  "class_list : %s \n"
                  "fold : %s \n"
                  "manual : %s \n"
                  "reviewed : %s \n") % (self.path, self.slide_list, self.class_list, self.fold, self.manual, self.reviewed)
        return string



def save_object(path, objects) :
    with open(path, 'wb') as file :
        pickle.dump(objects, file)

def load_object(path) :
    with open(path, 'rb') as file :
        objects = pickle.load(file)
        return objects

def nested_np_to_list(arr, printing = False) :
    new_list = list(map(lambda x : list(x), arr))
    if printing : print('new_list : ', new_list)
    return new_list


def generate_geoJson(coordinates : list or np.ndarray, type = 'Polygon', 
                    object_type = 'annotation', isLocked = True, 
                    cls_name = "CMV", colorRGB = 204) -> list : 
    
    json = list()

    for idx, coor in enumerate(coordinates) :
        x1, y1, x2, y2, confidence = coor
        ordered_coor = [[[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]]
        sub_json = {
            "type": "Feature",
            "id": idx+1,
            "geometry": {
                "type": type,
                "coordinates": ordered_coor
                },
            "properties":
                {
                    "object_type" : object_type,
                    "isLocked" :isLocked,
                    "classification" : {
                        "name" : cls_name,
                        "colorRGB" : colorRGB
                    }
                }
            }
        json.append(sub_json)

    return json

def convert_pkl_to_geoJson(pkl_file : dict or str, path ='output/', format = '.svs') :

    if isinstance(pkl_file, str) :
        pred_boxes = load_object(pkl_file)
    elif isinstance(pkl_file, dict) :
        pred_boxes = pkl_file
    
    for wsi, coor in pred_boxes.items() :
        anno = generate_geoJson(coor)
        fname = f'{path}{wsi[:-len(format)]}.json'
        with open(fname, 'w') as outfile:
            geojson.dump(anno, outfile)