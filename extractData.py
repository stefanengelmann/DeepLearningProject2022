import os 
import numpy as np
import json
import gzip

def isnan(x):
    return x==x
SDR = []
SIR = []
SAR = []
folderpath = "./final_eval/results/test/"
for path in os.listdir("./final_eval/results/test/"):
    with gzip.open("./final_eval/results/test/"+path, 'r') as file:
        json_bytes = file.read()             
    json_str = json_bytes.decode('utf-8')     
    tmp_data = json.loads(json_str)           
    for name in [1,0]:
        for num in range(len(tmp_data["targets"][name]["frames"])):
            if isnan(tmp_data["targets"][name]["frames"][num]["metrics"]["SDR"]): 
                SDR.append(tmp_data["targets"][name]["frames"][num]["metrics"]["SDR"] )
            if isnan(tmp_data["targets"][name]["frames"][num]["metrics"]["SIR"]): 
                SIR.append(tmp_data["targets"][name]["frames"][num]["metrics"]["SIR"] )
            if isnan(tmp_data["targets"][name]["frames"][num]["metrics"]["SAR"]): 
                SAR.append(tmp_data["targets"][name]["frames"][num]["metrics"]["SAR"] )
SDR = np.mean(SDR)
SIR = np.mean(SIR)
SAR = np.mean(SAR)
print("SDR: " , SDR)
print("SIR: " , SIR)
print("SAR: " , SAR)
