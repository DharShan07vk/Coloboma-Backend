import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("model/coloboma_detector.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open("coloboma_detector.tflite", "wb") as f:
    f.write(tflite_model)
