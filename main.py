from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
import urllib.request
import shutil
import os

app = FastAPI()

# Add the domain which will be accessing this API. Otherwise you will get CORS error
origins = ["http://localhost:3000"]
# origins = ["https://sushmitha-93.github.io"]

# origins = ["*"]
# methods = ["*"]
# headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading Trained ML model
print("****************loading model*****************")
# print("helloo")
# print(os.getcwd())  // in Docker, current directory is code.
model_dir = "app/distracted_driving_resnet_model.h5" #in Docker, current directory is code, Hence using app/<filename>
resnet_model = load_model(model_dir)
print("**********************************************")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
   
   # Saving imgage as "destination.png"
    with open("destination.png", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread("destination.png")  
    img = img[50:,120:-50]
    img = cv2.resize(img,(224,224)) 
    img = np.array(img).reshape(-1,224,224,3)
    print(img.shape)
    # Model predict
    img_pred = resnet_model.predict(img)

    # image prediction class
    img_pred_class = np.argmax(img_pred, axis=1)
    pred_class = img_pred_class[0]
    print(pred_class)
   
    # image prediction percentage
    pred_prob = "{:.4f}".format(img_pred[0][img_pred_class[0]]*100)
    print(pred_prob)

    return {"class": str(pred_class), "probability":pred_prob}

@app.post("/imageLink/")
async def getPredForImageLink(image_link: str = ""):
    if image_link=="":
        return "No Image Link provided"

    
    urllib.request.urlretrieve(image_link,"img.png")

    img = cv2.imread("img.png")    
    img = img[50:,120:-50]
    img = cv2.resize(img,(224,224)) 
    img = np.array(img).reshape(-1,224,224,3)
    print(img.shape)

    # Model predict
    img_pred = resnet_model.predict(img)

    # image prediction class
    img_pred_class = np.argmax(img_pred, axis=1)
    pred_class = img_pred_class[0]
    print(pred_class)
   
    # image prediction percentage
    pred_prob = "{:.4f}".format(img_pred[0][img_pred_class[0]]*100)
    print(pred_prob)

    return {"class": str(pred_class), "probability":pred_prob}
    