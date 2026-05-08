# Mental Health Classifier — Streamlit Web App


import streamlit as st
import joblib
import numpy as np
import pandas as pd

import re
import string
import emoji

import nltk

try:

    nltk.data.find('corpora/stopwords')

except LookupError:
    nltk.download('stopwords')

from sklearn.base import BaseEstimator, TransformerMixin

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #A0A0A0;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333333;
        margin-top: 20px;
    }

    .prediction {
        font-size: 28px;
        font-weight: bold;
        color: #00FFAA;
    }

    .confidence {
        font-size: 20px;
        color: #FFFFFF;
    }

    .class-box {
        background-color: #1A1A1A;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid #2E2E2E;
    }

    textarea {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# STOPWORDS
# =========================

stop_words = set(stopwords.words('english'))

negations = {"no", "not", "nor", "n't", "never"}

stop_words = stop_words - negations

# =========================
# PREPROCESSING FUNCTION
# =========================


def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = emoji.demojize(text, delimiters=(" ", " "))

    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# =========================
# CUSTOM FEATURE EXTRACTOR
# =========================


class TextFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = pd.DataFrame()

        df['statement'] = X

        df['clean_text'] = df['statement'].apply(
            preprocess_text
        )

        df['statement_length'] = df['statement'].apply(len)

        df['num_words'] = df['statement'].apply(
            lambda x: len(str(x).split())
        )

        df['vocabulary_size'] = df['statement'].apply(
            lambda x: len(set(str(x).split()))
        )

        df['avg_word_length'] = np.round(
            df['statement_length'] /
            df['num_words'].replace(0, 1)
        )

        df['avg_word_length'] = (
            df['avg_word_length']
            .astype(int)
        )

        return df[
            [
                'clean_text',
                'statement_length',
                'num_words',
                'vocabulary_size',
                'avg_word_length'
            ]
        ]

# =========================
# LOAD MODEL
# =========================

pipeline = joblib.load("mental_health_pipeline.pkl")

# =========================
# HEADER
# =========================

st.markdown(
    '<div class="title">🧠 Mental Health Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP Mental Health Classification System Using ML</div>',
    unsafe_allow_html=True
)

# =========================
# AVAILABLE CLASSES
# =========================

st.subheader("📌 Available Classes")

classes = list(pipeline.classes_)

cols = st.columns(3)

for i, cls in enumerate(classes):
    cols[i % 3].info(cls)

# =========================
# TEXT INPUT
# =========================

text = st.text_area(
    "✍️ Enter your text below:",
    height=180,
    placeholder="Type how you are feeling..."
)

# =========================
# BUTTON
# =========================

if st.button("🔍 Classify Emotion", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        probs = pipeline.predict_proba([text])[0]

        classes = pipeline.classes_

        sorted_indices = np.argsort(probs)[::-1]

        top_index = sorted_indices[0]

        prediction = classes[top_index]

        confidence = probs[top_index] * 100

        # custom normal threshold
        if prediction == "Normal" and confidence < 90:

            second_index = sorted_indices[1]

            second_class = classes[second_index]

            second_confidence = probs[second_index] * 100

            prediction = f"Possibly {second_class}"

            confidence = second_confidence

        # =========================
        # MAIN RESULT
        # =========================

        st.markdown(
            f'''
            <div class="result-box">
                <div class="prediction">Prediction: {prediction}</div>
                <br>
                <div class="confidence">Confidence: {confidence:.2f}%</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # =========================
        # PROBABILITIES
        # =========================

        st.subheader("📊 Class Probabilities")

        sorted_probs = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )

        for cls, prob in sorted_probs:

            percentage = prob * 100

            st.markdown(
                f"""
                <div class="class-box">
                    <b>{cls}</b><br><br>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(prob))

            st.write(f"{percentage:.2f}%")

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "⚠️ This is an experimental NLP demonstration project and not a medical diagnosis system."
)




