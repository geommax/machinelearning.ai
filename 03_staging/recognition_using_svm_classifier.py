import cv2
import torch
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC

# Load FaceNet model (embedding only)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
clf = SVC(probability=True)

# Load classifier model
classifier = joblib.load("svm_classifier.joblib")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face_embedding(image):
    image = Image.fromarray(image)
    face_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.cpu().numpy()[0]

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

# Prediction
def predict_person(embedding):
    probs = classifier.predict_proba([embedding])[0]
    max_idx = np.argmax(probs)
    name = classifier.classes_[max_idx]
    prob = probs[max_idx]
    return str(name), float(prob)


# Main loop
def run():
    cap = cv2.VideoCapture(0)
    print("[INFO] Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            try:
                embedding = extract_face_embedding(face)
                name, prob = predict_person(embedding)

                # Optional threshold
                if prob < 0.6:
                    label = "Unknown"
                else:
                    label = f"{name} ({prob:.2f})"

            except Exception:
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
