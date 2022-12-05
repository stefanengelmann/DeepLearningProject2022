import os 
import numpy as np
import json
import gzip

SDR = []
SIR = []
SAR = []
for path in os.listdir("./test/"):
    with gzip.open("./test/"+path, 'r') as file:
        json_bytes = file.read()             
    json_str = json_bytes.decode('utf-8')     
    tmp_data = json.loads(json_str)           
    for name in [1,0]:
        for num in range(len(tmp_data["targets"][name]["frames"])):
            SDR.append(tmp_data["targets"][name]["frames"][num]["metrics"]["SDR"] )
            SIR.append(tmp_data["targets"][name]["frames"][num]["metrics"]["SIR"] )
            SAR.append(tmp_data["targets"][name]["frames"][num]["metrics"]["SAR"] )
SDR = np.mean(SDR)
SIR = np.mean(SIR)
SAR = np.mean(SAR)
print("SDR: " , SDR)
print("SIR: " , SIR)
print("SAR: " , SAR)
