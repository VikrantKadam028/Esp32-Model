from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
import uvicorn
import cv2
import numpy as np

app = FastAPI()

print("Loading emotion model...")
# Preload emotion model
emotion_model = DeepFace.build_model("Emotion")
print("Model loaded!")

# Load OpenCV face detector (faster pre-check)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        
        if img is None:
            return {"emotion": "error"}
        
        # STEP 1: Quick face detection with OpenCV (FAST)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # If NO face detected, return immediately
        if len(faces) == 0:
            print("⚠️ No face detected")
            return {"emotion": "no_face"}
        
        # STEP 2: Face detected, now run DeepFace
        print(f"✅ Face detected, analyzing emotion...")
        result = DeepFace.analyze(
            img,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )
        
        # Extract emotion
        emotion = result[0]["dominant_emotion"]
        confidence = result[0]["emotion"][emotion]
        
        print(f"😊 Detected: {emotion} ({confidence:.1f}%)")
        
        # Return ONLY the emotion string (same format as before)
        return {"emotion": emotion}
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"emotion": "error"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
