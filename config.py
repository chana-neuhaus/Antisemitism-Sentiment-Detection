"""App settings.

Update these values when you replace or retrain the model.
"""

MODEL_PATH = "models/current_model.pkl"

# IMPORTANT:
# This must match the SentenceTransformer model used to create the
# embeddings during training. If you trained with a different model,
# replace this value.
EMBEDDING_MODEL_NAME = r"C:\Users\chani\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

# Label mapping. Change these only if your model uses the opposite class order.
LABELS = {
    0: "Not Biased",
    1: "Biased",
}
