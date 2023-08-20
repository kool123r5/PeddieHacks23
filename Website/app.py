from flask import Flask, render_template, url_for, redirect

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import time


import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
import torchvision
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import tensorflow as tf



img_path = "rgb1.jpg"
img_url = 'C:\\Users\\ashwi\\OneDrive\\Documents\\hackathon\\flask\\static\\' + img_path
image = img_url.replace("rgb", '')
ximage = Image.open(image)

#Use GPU if available to speed up the process








# Preprocess the RGB images
rgb_image = ""
#DISPLAY THE IMAGE ABOVE

# Preprocess the Infrared images

#USE THIS IMAGE FOR THE THE MODEL



#Transform the data to the required size and format to train under the ResNet model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

preprocessed_image = transform(ximage)
preprocessed_image = preprocessed_image.unsqueeze(0)



#Loading the ResNet-50 model
model = models.resnet50(pretrained=True)
#Changing the last fully connected layer to suit our needs of binary classification
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)


#Load the model
model.load_state_dict(torch.load("C:\\Users\\ashwi\\OneDrive\\Documents\\hackathon\\flask\\RESNET-50_DETECTINGWILDFIRE", map_location=torch.device('cpu'))["model_state_dict"])

model.eval()  # Set the model to evaluation mode

# Pass the preprocessed image through the model
with torch.no_grad():
    outputs = model(preprocessed_image)
    _, predicted = torch.max(outputs, dim=1)  # Get predicted class


detection_wildfire = predicted.item()
if(round(detection_wildfire) == 1):
     isFire = True
elif(round(detection_wildfire) == 0):
     isFire = False



model1 = tf.keras.models.load_model('firePredict.h5')

def predict_class(image_path):
    img = Image.open(image_path)
    img = img.resize((350, 350))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255
    img_array = tf.expand_dims(img_array, 0)
    prediction = model1.predict(img_array)
    class_label = 'Likely' if prediction[0][0] >= 0.5 else 'Unlikely'
    print(prediction[0])
    return (class_label, round(prediction[0][0] * 100, 1))



app = Flask(__name__)

if(not isFire):
   text = "Currently no fire"
   prediction, frac = predict_class(img_url)
   predict = "Prediction for future: " + str(prediction) + " (" + str(frac) + "%)"
else:
   text = ""
   predict = "There is a fire right now"


@app.route("/")
def home():
       img_url = url_for('static', filename=img_path)
       return render_template("home.html", img_url = img_url, predict = predict, text = text)

       
   #  x = predict_class("C:\\Users\\ashwi\\OneDrive\\Documents\\hackathon\\flask\\wildfire.jpg")
   #  return render_template("home.html", x = x)


if __name__ == '__main__':
    app.run(debug=True)
