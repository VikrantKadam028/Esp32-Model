from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
import uvicorn
import cv2
import numpy as np

app = FastAPI()

print("Loading emotion model...")
# Preload emotion model only
emotion_model = DeepFace.build_model("Emotion")
print("Model loaded!")

@app.get("/")
def home():
    return {"message": "Emotion API running!"}

@app.post("/detect")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Analyze using DeepFace
        result = DeepFace.analyze(
            img,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result["dominant_emotion"]

        return {"emotion": emotion}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
