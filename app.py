import os
import gradio as gr
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# ==========================
# Load Models Safely
# ==========================

print("Loading models...")

try:
    word_vect = joblib.load("models/word_tfidf.joblib")
    char_vect = joblib.load("models/char_tfidf.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    voting_model = joblib.load("models/voting_model.joblib")
    question_embeddings = joblib.load("models/question_embeddings.joblib")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Models loaded successfully.")

except Exception as e:
    print("Error loading models:", e)
    raise e


# ==========================
# Chat Function
# ==========================

def chatbot(user_input):
    if not user_input.strip():
        return "Please enter a valid question."

    try:
        # Create TF-IDF features
        word_features = word_vect.transform([user_input])
        char_features = char_vect.transform([user_input])
        combined_features = hstack([word_features, char_features])

        # Predict
        pred = voting_model.predict(combined_features)
        predicted_answer = label_encoder.inverse_transform(pred)[0]

        # Semantic similarity
        query_embedding = embedder.encode([user_input])
        scores = cosine_similarity(query_embedding, question_embeddings)[0]
        best_score = np.max(scores)

        # Confidence threshold
        if best_score > 0.60:
            return predicted_answer
        else:
            return predicted_answer  # fallback

    except Exception as e:
        return f"Error during prediction: {str(e)}"


# ==========================
# Gradio UI
# ==========================

ui = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        placeholder="Ask something about Riphah University...",
        lines=2
    ),
    outputs="text",
    title="Riphah AI Assistant",
    description="AI-powered assistant for Riphah University queries."
)


# ==========================
# IMPORTANT FOR RENDER
# ==========================

port = int(os.environ.get("PORT", 7860))

ui.launch(
    server_name="0.0.0.0",
    server_port=port
)