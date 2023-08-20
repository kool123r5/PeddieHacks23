import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

train_dir = '/Users/kushalb/Downloads/fireData/train'
validation_dir = '/Users/kushalb/Downloads/fireData/valid'
test_dir = '/Users/kushalb/Downloads/fireData/test'

# image dimensions and other params
img_width, img_height = 350, 350
batch_size = 64
epochs = 1

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,  # Rescale pixel values between 0 and 1
    rotation_range=20,  # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill mode for new pixels
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

sample_images, sample_labels = next(train_generator)

#Displaying the images in the batch
plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    plt.subplot(8, 8, i+1)
    plt.imshow(sample_images[i])
    plt.title("Label: " + str(sample_labels[i]))
    plt.axis('off')
plt.show()

#Building the model....
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid > softmax here
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dir, validation_data=validation_dir, epochs=epochs)

# Make predictions on test
def predict_wildfire(image_path):
    img = Image.open(image_path)
    img = img.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    if prediction[0][0] >= 0.5:
        return "Wildfire"
    else:
        return "No Wildfire"

new_image_path = '/Users/kushalb/Downloads/fireData/train/wildfire/-79.70618,48.94454.jpg'
prediction = predict_wildfire(new_image_path)
print(f"The image is classified as: {prediction}")

image_path_agn = '/Users/kushalb/Downloads/fireData/test/nowildfire/-79.744402,43.689693.jpg'
prediction = predict_wildfire(image_path_agn)
print(f"The image is classified as: {prediction}")

model.save('firePredict.h5')
