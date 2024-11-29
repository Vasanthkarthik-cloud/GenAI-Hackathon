from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image_path = "/home/scaret/Downloads/modelgenai/dirty_64.jpg"  # Update with the actual image path
image = Image.open(image_path).convert("RGB")

# Resize the image to be 224x224
ize = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Convert image to numpy array
image_array = np.asarray(image)

# Normalize the image (if model expects normalization to [-1, 1])
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict the model
prediction = model.predict(data)

# Get the class with the highest confidence
index = np.argmax(prediction)
class_name = class_names[index].strip()  # Strip any leading/trailing whitespace
confidence_score = prediction[0][index]

# Print prediction and confidence score
print(f"Class: {class_name}, Confidence Score: {confidence_score}")

