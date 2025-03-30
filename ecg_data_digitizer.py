from ecgtizer.ecgtizer import ECGtizer
import os
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
import torch
warnings.filterwarnings("ignore")

# Specify the directory path
directory = "./data/ecg_data"
pdf_path = os.path.join(directory, "pdf")
save_path = os.path.join(directory,"datasets")
os.makedirs(pdf_path,exist_ok=True)
os.makedirs(save_path,exist_ok=True)

model_path = "./ecgtizer/model/Model_Completion.pth"
device = torch.device('cpu')

# List all the files in the directory
files = os.listdir(pdf_path)

filenames = []
labels = []

# Print the list of files
for file in tqdm(files):
    try:
        ecg_extracted = ECGtizer (pdf_path + f"/{file}", 500, extraction_method="fragmented", verbose = False, DEBUG = False)
        ecg_extracted.completion(model_path, device)
        lead = ecg_extracted.get_extracted_lead(completion = True)
        keys_order = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        combined_matrix = np.stack([lead[key] for key in keys_order], axis=0)
        filename =  f"{file[:-4]}.npy"
        np.save(os.path.join(save_path , filename), combined_matrix)
        filenames.append(filename)
        labels.append("N")
    except Exception as e:
        pass

metadata = pd.DataFrame({
    "filename": filenames,
    "label": labels
})
metadata.to_csv(save_path + "/metadata.csv", index=False)