from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

# from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import keras_nlp
import keras

app = FastAPI()

# Charger le modèle
model = keras.models.load_model("outputs/models/prod/bert_base_en_model.keras")

# Recréer le preprocessor (même config que l'entraînement)
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_base_en_uncased", sequence_length=128
)


class Review(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"Message": "Bienvenue Steam reviews FastAPI"}


@app.get("/hello")
def say_hello():
    return {"Hello": "World3"}


@app.post("/predict")
def predict_sentiment(review: Review):

    processed = preprocessor([review.text])

    # (sigmoid output)
    prob = model.predict(processed, verbose=0).flatten()[0]

    # Calcul sentiment et confiance
    sentiment = "POSITIF" if prob > 0.5 else "NÉGATIF"
    confidence = prob if prob > 0.5 else 1 - prob

    return {
        "sentiment": sentiment,
        "confidence": round(float(confidence) * 100, 2),
        "probabilities": {
            "negative": round(float(1 - prob) * 100, 2),
            "positive": round(float(prob) * 100, 2),
        },
    }


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"message": "Pas trouvé"})
