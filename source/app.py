
#streamlit app
from matplotlib.pylab import norm
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

# Load the model
model = load_model(r"C:\Users\jacok\OneDrive\Desktop\net\netscript\source\keras_model.h5", compile=False)

# Load the labels
class_names = open(r"C:\Users\jacok\OneDrive\Desktop\net\netscript\source\labels.txt", "r").readlines()

# Streamlit app
st.title("Fruit Classifier App")
st.header("Michael Muthomi")
st.header("Labels: 0 - rotten bananas. 1- rotten apples. 2- rotten oranges. 3- fresh bananas. 4- fresh apples. 5- fresh oranges")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    #Resize the image
    resized_image_array = zoom(normalized_image_array, (224/normalized_image_array.shape[0], 224/normalized_image_array.shape[1], 1))

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = resized_image_array

    # Make a prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()

    # Display the prediction
    st.write("Prediction: ", class_name)