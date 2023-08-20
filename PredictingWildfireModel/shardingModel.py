import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.load_model('/Users/kushalb/Documents/VSCode/PeddieHacks2023/firePredict.h5')
tfjs.converters.save_keras_model(model, '/Users/kushalb/Downloads/predictingFireModel')

