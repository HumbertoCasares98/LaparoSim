from transfer import Transfer
from preprocess import Preprocess
from cut import Cut
from suture import Suture
import os

preprocess_obj = Preprocess()
transfer_obj = Transfer()
cut_obj = Cut()
suture_obj = Suture()


async def inference_maps(csv_file_path: str, exercise: str):
    inference = None
    if os.path.exists(csv_file_path):
        if exercise=="1":
            x,y,z,x2,y2,z2=preprocess_obj.read_file(csv_file_path)
            maps_values = preprocess_obj.maps_2(csv_file_path, x, y, z, x2, y2, z2)
            inference = transfer_obj.classify(maps_values)
        elif exercise=="2":
            x,y,z,x2,y2,z2=preprocess_obj.read_file(csv_file_path)
            maps_values = preprocess_obj.maps_1(csv_file_path, x, y, z, x2, y2, z2)
            inference = cut_obj.classify(maps_values)
        elif exercise=="3":
            x,y,z,x2,y2,z2=preprocess_obj.read_file(csv_file_path)
            maps_values = preprocess_obj.maps_2(csv_file_path, x, y, z, x2, y2, z2)
            inference = suture_obj.classify(maps_values)
        return {"status_code": 200, "status_message": inference}
    else:
        return {"status_code": 404, "status_message": "File not found"}
