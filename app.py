import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import keras


st.set_page_config(
    page_title="Dog Breed Predictor",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_dog_breed_model():
    """Load and cache the trained Keras model."""
    model_path = "model/dog_breed_model.keras"
    return keras.models.load_model(model_path)


@st.cache_data
def load_class_names():
    """Load and cache class (breed) names from labels.csv."""
    labels_all = pd.read_csv("dog_dataset/labels.csv")
    class_names = np.unique(labels_all["breed"])
    return class_names


def prepare_image(pil_image: Image.Image, target_size=(224, 224)):
    """Preprocess the uploaded image for model prediction."""
    image_resized = pil_image.convert("RGB").resize(target_size)
    img_array = np.asarray(image_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_breed(pil_image: Image.Image, top_k: int = 5):
    """Run prediction and return top-k breeds with probabilities."""
    model = load_dog_breed_model()
    class_names = load_class_names()

    img_array = prepare_image(pil_image)
    probs = model.predict(img_array, verbose=0)[0]

    top_k = min(top_k, len(probs))
    top_indices = probs.argsort()[-top_k:][::-1]

    breeds = class_names[top_indices]
    confidences = probs[top_indices]

    return breeds, confidences


def main():
    # --- Custom styling ---
    st.markdown(
        """
        <style>
            .main {
                background: radial-gradient(circle at top left, #f9fafb, #e5e7eb);
            }
            .prediction-card {
                padding: 1.25rem 1.5rem;
                border-radius: 1rem;
                background: white;
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.35);
            }
            .title-text {
                font-weight: 800;
                letter-spacing: -0.04em;
            }
            .accent-text {
                color: #2563eb;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    st.sidebar.title("About")
    st.sidebar.write(
        "Upload a dog image and this app will predict its breed using a "
        "convolutional neural network trained on the Kaggle dog breed dataset."
    )
    st.sidebar.markdown("---")
    top_k = st.sidebar.slider("Number of top breeds to show", 1, 10, 5)
    st.sidebar.info(
        "For best results, use clear images where the dog is centered and well lit."
    )

    # --- Header ---
    st.markdown(
        """
        <div>
            <h1 class="title-text">Dog Breed <span class="accent-text">Prediction</span></h1>
            <p style="font-size: 1.05rem; color: #4b5563; max-width: 640px;">
                An interactive demo powered by a transfer-learning model based on MobileNetV2.
                Simply upload an image of a dog to see the predicted breed and confidence scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacing

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        uploaded_file = st.file_uploader(
            "Upload a dog image (JPG or PNG)", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # Show the image immediately so user can see while predicting
            st.subheader("Preview")
            st.image(
                image,
                caption="Uploaded dog image",
                use_column_width=True,
            )

            predict_button = st.button("🔍 Predict Breed", type="primary")

            if predict_button:
                with st.spinner("Analyzing image and predicting breed..."):
                    breeds, confidences = predict_breed(image, top_k=top_k)

                top_breed = breeds[0]
                top_conf = confidences[0] * 100

                st.markdown("")
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div style="font-size: 0.9rem; color: #6b7280; text-transform: uppercase; letter-spacing: .15em;">
                            Top Prediction
                        </div>
                        <div style="font-size: 1.6rem; font-weight: 700; margin-top: .4rem;">
                            {top_breed}
                        </div>
                        <div style="font-size: 0.95rem; color: #4b5563; margin-top: .35rem;">
                            Confidence: <strong>{top_conf:.2f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Detailed probabilities
                st.markdown("### Top predictions")
                results_df = pd.DataFrame(
                    {
                        "Breed": breeds,
                        "Confidence (%)": (confidences * 100).round(2),
                    }
                )
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### Confidence distribution")
                chart_df = results_df.set_index("Breed")
                st.bar_chart(chart_df)
        else:
            st.info("Upload a clear photo of a dog to get started.")

    with col_right:
        st.markdown("### How it works")
        st.write(
            "- **Model**: A convolutional neural network using **MobileNetV2** as a feature extractor.\n"
            "- **Input size**: Images are resized to **224×224** pixels and normalized.\n"
            "- **Output**: The model predicts probabilities over all dog breeds from the Kaggle dataset."
        )

        st.markdown("### Tips for best predictions")
        st.write(
            "- Use high-resolution images where the dog is **clearly visible**.\n"
            "- Avoid heavy clutter in the background.\n"
            "- Try a few different photos of the same dog to see prediction stability."
        )


if __name__ == "__main__":
    main()


