"""Streamlit app for the DSC 360 antisemitic sentiment classifier."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from classify import classify_text, load_embedding_model, load_model
from config import EMBEDDING_MODEL_NAME, LABELS, MODEL_PATH


st.set_page_config(
    page_title="Antisemitic Sentiment Detection",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource
def get_model():
    """Load the sklearn model once per app session."""
    return load_model(MODEL_PATH)


@st.cache_resource
def get_embedding_model():
    """Load the embedding model once per app session."""
    return load_embedding_model(EMBEDDING_MODEL_NAME)


def main():
    st.title("🛡️ Antisemitic Sentiment Detection")

    st.markdown(
        """
        This app analyzes social media-style text and predicts whether it contains
        antisemitic sentiment using BERT sentence embeddings and a machine learning classifier.
        """
    )

    with st.sidebar:
        st.header("Model Information")
        st.write(f"Active model: `{Path(MODEL_PATH).name}`")
        st.write("Classifier: `Logistic Regression`")
        st.write("Embedding model:")
        st.code(str(EMBEDDING_MODEL_NAME), language="text")

        st.divider()

        st.caption(
            "To replace the classifier, save your new model as "
            "`models/current_model.pkl` and restart the app."
        )

    st.subheader("Enter text")

    user_text = st.text_area(
        label="Social media text",
        height=180,
        placeholder="Paste or type a social media post here...",
        label_visibility="collapsed",
        key="main_text_input",

    )

    classify_button = st.button("Classify Text", type="primary")

    if classify_button:
        if not user_text.strip():
            st.warning("Please enter text before classifying.")
            return

        try:
            model = get_model()
            embedding_model = get_embedding_model()

            result = classify_text(
                text=user_text,
                model=model,
                embedding_model=embedding_model,
                labels=LABELS,
            )

            label = result["label"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            st.divider()
            st.subheader("Result")

            if label == "Biased":
                st.error("⚠️ Antisemitic")
            else:
                st.success("✅ Not Antisemitic")

            if confidence is not None:
                st.metric(
                    label="Model confidence",
                    value=f"{confidence * 100:.1f}%",
                )
                st.progress(float(confidence))

            if probabilities is not None:
                st.subheader("Class Probabilities")

                for class_num, probability in enumerate(probabilities):
                    class_label = LABELS.get(class_num, str(class_num))
                    st.write(f"{class_label}: **{probability * 100:.1f}%**")


        except Exception as error:
            st.error("The app could not classify this text.")
            st.exception(error)


if __name__ == "__main__":
    main()

