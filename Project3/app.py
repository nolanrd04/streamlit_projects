import streamlit as st
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
from dataProcessor import preprocess_image, IMG_SIZE, CLASS_NAMES
from CNN import load_data, build_model, train_and_evaluate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(__file__)
accuracy_plot = os.path.join(BASE_DIR, "accuracy_plot.png")
loss_plot = os.path.join(BASE_DIR, "loss_plot.png")

st.set_page_config(page_title="Convolutional Neural Network", page_icon="🧩", layout="centered")
st.title("Convolutional Neural Network - Vehicle Classifier")
st.markdown("The kaggle dataset strictly utilizes images of three vehicles: cars, motorbikes, and airplanes.")
st.markdown("The CNN model is trained to classify these vehicles.")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("vehicle_classifier_model.keras")

model = load_model()

# Tabs
tab1, tab2, tab3 = st.tabs(["Test Model", "Train Model", "Project Requirements"])

# ---------------- Test Model Tab ----------------
with tab1:
    st.title("Test Model")
    st.caption("Upload an image or dataset to test the model.")

    # Upload a single image
    uploaded_img = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        img_array = preprocess_image(uploaded_img)  # already grayscale + normalized
        img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,1)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        st.image(Image.open(uploaded_img), caption=f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.2%})", width='stretch')


# ---------------- Train Model Tab ----------------
with tab2:
    st.title("Train Model")
    st.subheader("IMPORTANT:\nTraining the model is not necessary for the app to work. Our model is already trained and ready to go.")
    # Controls
    dataset_file = st.file_uploader("Upload dataset (.npz)", type=["npz"])
    if dataset_file:
        data = np.load(dataset_file)

    epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50, step=10)
    batch_size = st.number_input("Batch Size", min_value=16, max_value=128, value=32, step=16)
    val_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05)

    if st.button("🚀 Train Model"):
        st.write("Loading dataset...")
        try:
            if dataset_file:
                X, y = data["X"], data["y"]
                st.write("✅ Dataset loaded:", data["X"].shape, data["y"].shape)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.write("Training model...")
            model = build_model()
            history, (test_loss, test_acc) = train_and_evaluate(
                model, X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size, validation_split=val_split
            )

            st.success(f"✅ Training complete. Test Accuracy: {test_acc*100:.2f}%, Test Loss: {test_loss:.4f}")

            # Plot Accuracy
            fig_acc, ax = plt.subplots()
            ax.plot(history.history['accuracy'], label='Train Accuracy')
            ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax.set_title("Model Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
            st.pyplot(fig_acc)

            # Plot Loss
            fig_loss, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Train Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title("Model Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            st.pyplot(fig_loss)

            # Save model
            if st.button("💾 Save Model"):
                model.save("vehicle_classifier_model.keras")
                st.success("✅ Model saved successfully.")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")

# ---------------- Project Requirements Tab ----------------
with tab3:
    st.title("Project Requirements")
    st.markdown("## 1 and 2. Dataset")
    st.markdown('''Our dataset are the Car, Airplane, and Motorbike folders from the Kaggle dataset Natural Images:
    https://www.kaggle.com/datasets/prasunroy/natural-images''')

    st.markdown("## 3. Description")
    st.markdown('''
                Our dataset consists of different images of cars, planes, and motorbikes. These images are on the
                older side, so using images of newer models of cars or planes tends to get incorrect results.
                For example, a 2024 corvette (which arguably looks like a plane) will probably get classified
                as a plane.
                ''')
    st.markdown("## 4. Libraries")
    st.markdown('''
                Tensorflow, Numpy, sklearn, os, matplotlib.

                ## 5-10. Convultion Layers, Pooling, and Flattening
                ''')
    st.code('''
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=input_shape
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="relu"
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="relu"
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            ''')
    st.markdown('''
                ### Max pooling explanation:
                In this code, MaxPooling will move a 2x2 window across the data and take the most important information
                from those windows.

                ## 11. Training and evaluation
                After training and testing our model behind the scenes, it was shown to be 98.59% accurate with a loss of 6.11%.

                ## Results:
                ''')
    st.image(accuracy_plot)
    st.image(loss_plot)

    st.markdown('''
                ## 12. Summarizing the Project
                Our CNN takes images converted to 128x128 black and white np arrays as input, then gets trained on those images
                to classify other vehicles it is given. There are hundreds of images for each vehicle (car, airplane, and motorbike),
                but the images are not super recent, so using new models of cars or airplanes typically yields incorrect results.
                Some ways to improve the model would be to experiment on training with colored images and adding a bunch of recent
                vehicle images to the dataset.
                ''')
    
