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

# Load multiple face detectors for better detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

@app.get("/")
def home():
    return {"message": "Emotion API running!"}

def enhance_image(img):
    """Enhance image for better face detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    return gray

def detect_face_multi(img):
    """Try multiple detection methods"""
    gray = enhance_image(img)
    
    # Method 1: Default cascade (more strict)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # More sensitive
        minNeighbors=3,     # Less strict
        minSize=(20, 20)    # Smaller minimum size
    )
    
    if len(faces) > 0:
        return True
    
    # Method 2: Alternative cascade
    faces = face_cascade_alt.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )
    
    if len(faces) > 0:
        return True
    
    # Method 3: Try even more lenient settings
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=2,
        minSize=(15, 15)
    )
    
    return len(faces) > 0

@app.post("/detect")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("❌ Invalid image")
            return {"emotion": "error"}
        
        print(f"📸 Image size: {img.shape}")
        
        # STEP 1: Try multiple detection methods
        has_face = detect_face_multi(img)
        
        if not has_face:
            print("⚠️ No face detected with OpenCV")
            # Still try DeepFace as last resort (it has its own detector)
            try:
                result = DeepFace.analyze(
                    img,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                emotion = result[0]["dominant_emotion"]
                print(f"✅ DeepFace found face: {emotion}")
                return {"emotion": emotion}
            except:
                print("❌ DeepFace also failed")
                return {"emotion": "no_face"}
        
        # STEP 2: Face detected, run DeepFace
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
        
        return {"emotion": emotion}
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        # If any error, still try to analyze
        try:
            result = DeepFace.analyze(
                img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip"
            )
            emotion = result[0]["dominant_emotion"]
            print(f"⚠️ Fallback detection: {emotion}")
            return {"emotion": emotion}
        except:
            return {"emotion": "error"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
