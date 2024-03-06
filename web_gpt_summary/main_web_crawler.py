from util.json_loader import json_iter
from util.url_summary import handle_summarize
import os

dir_path = os.path.join(os.getcwd(),"data","cvelistV5-main")
for json_obj in json_iter(dir_path):
    for row in json_obj['containers']['cna']['references']:
        url = row['url']
        summary = handle_summarize(url)
        print(url)
        print(summary)
        print("--------------------------------------------------")