import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('path/to/model/here.h5')

# Function to predict class using the loaded model
def predict_class(image_path):
    img = Image.open(image_path)
    img = img.resize((350, 350))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    class_label = 'Wildfire' if prediction[0][0] >= 0.5 else 'No Wildfire'
    print(prediction[0])
    return class_label


# count = 0
# total = 0
# for file in os.listdir('same/image/to/test'):
#     if total == 1000:
#         break
#     total = total + 1
#     predicted_class = predict_class(os.path.join('heres/an/image/to/test', file))
#     if predicted_class == 'Wildfire':
#         count = count + 1

# print("Percentage is:", count/total*100)

predicted_class = predict_class('/Users/kushalb/Downloads/plswork.jpeg')
print(predicted_class)
    
