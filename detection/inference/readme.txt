Description :
    There is a 'infer_script.py' that can infer the whole-slide images (WSI).
    You can set your own configuration in the config.json file
    
Input : 
    WSI_path : Path of folder that contains whole-slide images files
    WSI_slides : List of WSI files
    model: Your model file path
    model_config: Your model configuration file path
    gpu: You can select your cuda machine here, default is "cuda:0". If gpu not available this argument automatically select "cpu".
    batch_size: Number of images that parallelly infered, default = 8. If you want to infer only one image, select this to 1.
    patch_size: patches size in pixel, default = 256
    slide_step: Sliding step in pixel, default = 128
    result : Predicted bounding boxes folder path, default = "output/"

Output :
    Predicted bounding boxes in 'pkl' format
