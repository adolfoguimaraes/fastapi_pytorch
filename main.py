from fastapi import FastAPI, UploadFile
from models.ConvolutionalModel import ConvolutionalModel

import sys 
import os 

from PIL import Image
import torch
import torchvision.transforms as T

app = FastAPI(
    title="Machine Learning Classify Model",
    description="A model to classify imagens on CIFAR10 classes.",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

@app.on_event("startup")
async def startup_event():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConvolutionalModel()
    model.load_state_dict(torch.load("models/localmodel_dict.pt", map_location=device))
    categories = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    app.package = {
        "model": model,
        "device": device,
        "categories": categories
    }

@app.post("/api/v1/predict")
async def create_upload_file(file: UploadFile):
    img = Image.open(file.file)

    prep_transforms = T.Compose(
        [T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
    ])
    
    img_tensor = prep_transforms(img)
    batch = img_tensor.unsqueeze(0).to(app.package["device"])
    output = app.package["model"](batch)
    output_index = output.argmax().item()
    category_output = app.package["categories"][output_index]

    logits = torch.nn.functional.softmax(output, dim=1)
    prob_dict = {}
    for i, classname in enumerate(app.package["categories"]):
        prob = logits[0][i].item()
        prob_dict[classname] = prob
     

    return {"model": ConvolutionalModel.__name__, "filename": file.filename, "class": output_index, "class_description": category_output, "class_probability": prob_dict}

@app.get('/about')
def show_about():

    return {
        'api.verion': app.version,
        'sys.version': sys.version,
        'torch.__version__': torch.__version__,
        'torch.cuda.is_available': torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        'local.device': app.package['device']
    }
