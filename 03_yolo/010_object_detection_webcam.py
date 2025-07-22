import cv2, os, time
from ultralytics import YOLOE,YOLO
# import torch


print("Searching for available cameras...")
# for i in range(10):
#     cap_test = cv2.VideoCapture(i)
#     if cap_test.isOpened():
#         print(f"✅ Camera found at index: {i}")
#         cap_test.release()
#     else:
#         print(f"❌ No camera at index: {i}")
        
model = YOLO(r"D:\Project15_Objects_Detection_YOLO (53 classes)\runs\detect\Object_detect_img_512 (50 classes) old\weights\best.pt")
# model = YOLO("yolov8s.pt")
# model = YOLOE("yoloe-11l-seg.pt")

model.fuse()


device = 'cuda' if model.device.type == 'cuda' else 'cpu'

cap = cv2.VideoCapture(0)

# Lower resolution and set FPS to improve speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 60)

actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS set to: {actual_fps}")

if not cap.isOpened():
    exit("Camera failed to open.")

os.makedirs("captured_frames", exist_ok=True)

window_name = "YOLOv8 Live"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

frame_count = 0
skip_frames = 1  # process every 3rd frame

annotated = None  # to store last annotated frame for display

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if frame_count % (skip_frames + 1) == 0:
        start = time.time()
        results = model(frame, imgsz=640, conf=0.3, device=device)
        end = time.time()

        boxes = results[0].boxes
        annotated = frame.copy()

        if boxes is not None:
            for box, score, cls in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int)
            ):
                x1, y1, x2, y2 = map(int, box)
                label = model.names.get(cls, str(cls)) if score >= 0.3 else "unknown"
                color = (57, 255, 20) if score >= 0.6 else (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Text with background
                text = f"{label} {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                bg_color = (0, 255, 0) if score >= 0.3 else (0, 0, 255)  # green if known, red if unknown
                text_color = (255, 255, 255)  # white text

                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(annotated,
                              (x1, y1 - th - baseline - 5),
                              (x1 + tw + 10, y1),
                              bg_color, -1)
                cv2.putText(annotated, text, (x1 + 5, y1 - 5),
                            font, font_scale, text_color, thickness)

        fps = 1 / (end - start)
        fps_text = f"FPS: {fps:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 10), (0, 0, 0), -1)
        cv2.putText(annotated, fps_text, (15, 10 + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # === Image Size at Top-Right ===
        h, w = annotated.shape[:2]
        size_text = f"{w}x{h}"
        (text_w, text_h), _ = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (w - text_w - 20, 10), (w - 10, 10 + text_h + 10), (0, 0, 0), -1)
        cv2.putText(annotated, size_text, (w - text_w - 15, 10 + text_h + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    else:
        # Display last annotated frame without new detection
        if annotated is None:
            annotated = frame.copy()
        fps = 0

    cv2.imshow(window_name, annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"captured_frames/snapshot_{time.strftime('%Y%m%d-%H%M%S-%f')}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"Saved {filename}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
