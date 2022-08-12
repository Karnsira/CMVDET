import pickle


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


