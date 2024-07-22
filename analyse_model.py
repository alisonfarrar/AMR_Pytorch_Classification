# From model, load classifications and correlate with properties

import torch

MODEL_PATH = r"C:/Users/farrara/Desktop/AMR_Pytorch_Classification/models/AntibioticClassification_L48480fresh_1X_240315_1202/AMRClassification_[Ciprofloxacin-Cy3]_231118_1123"

# device
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)

