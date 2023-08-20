import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.load_model('path/to/the/predicting/fire/model.h5')
tfjs.converters.save_keras_model(model, 'path/to/my/folder')

