import json
import os

class json_iter:
    def __init__(self,dir_path) -> None:
        self.dir_path = dir_path
    
    def __iter__(self):
        for path, dirs, files in os.walk(self.dir_path):
            for file in files:
                if file.endswith(".json") and file.startswith("CVE-"):
                    with open(os.path.join(path, file)) as f:
                        yield json.load(f)
    

if __name__ == "__main__":
    dir_path = os.path.join(os.getcwd(),"data","cvelistV5-main")
    for json_obj in json_iter(dir_path):
        print(json_obj)
        # HTML files:json_obj['containers']['cna']['references']
        print('---')