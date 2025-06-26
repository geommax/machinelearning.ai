### 1. Training From Scratch 
    full-training pipeline
    
#### Sample Datasets Link
https://www.kaggle.com/datasets/emmarex/plantdisease/data

### 2. Transfter Learning (Feature Extraction & Fine Tuning)

Transfer Learning (full-training pipeline or classifier pipeline)


#### Keras Applications For Transfer Learning (Tensorflow)
https://keras.io/api/applications/

#### Torch Vision Models For Transfer Learning (Pytorch)
https://docs.pytorch.org/vision/0.9/models#

#### Hugging Face
https://huggingface.co/

### 2.1 face emotion recognition (MobileNet V2 and haarcascade face detection)

Face Emotion Recognition

#### REF API:
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

https://www.tensorflow.org/api_docs/python/tf/keras/losses

https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml




### 2.2 Face Recognition (facenet with pytorch example)

| Library / Model         | Framework             | Accuracy | Pretrained? | Notes                                    |
| ----------------------- | --------------------- | -------- | ----------- | ---------------------------------------- |
| **FaceNet**             | TensorFlow / PyTorch  | ⭐⭐⭐⭐☆    | ✅ Yes       | Embedding-based, very robust             |
| **Dlib**                | C++ / Python          | ⭐⭐⭐⭐☆    | ✅ Yes       | Simple API, decent accuracy              |
| **DeepFace**            | Keras / TensorFlow    | ⭐⭐⭐⭐☆    | ✅ Yes       | Unified API for multiple models          |
| **InsightFace**         | PyTorch / MXNet       | ⭐⭐⭐⭐⭐    | ✅ Yes       | State-of-the-art (ArcFace, CosFace)      |
| **Facenet-PyTorch**     | PyTorch               | ⭐⭐⭐⭐☆    | ✅ Yes       | Easy, pretrained on VGGFace2             |
| **VGGFace / VGGFace2**  | Keras / Torch         | ⭐⭐⭐⭐☆    | ✅ Yes       | Big dataset, good baseline               |
| **OpenFace**            | Torch                 | ⭐⭐⭐☆☆    | ✅ Yes       | Lightweight, research-grade              |
| **MediaPipe Face Mesh** | TensorFlow Lite / JS  | ⭐⭐⭐☆     | ✅ Yes       | Good for landmarks, not ID               |
| **ArcFace**             | PyTorch / InsightFace | ⭐⭐⭐⭐⭐    | ✅ Yes       | Excellent accuracy for identity matching |


```
VGGface2 က InceptionResnet Model ကို Train ဖို့အတွက် သုံးတဲ့ Dataset ဖြစ်ပါတယ်။ InceptionResnet က faceimage ကို 128-d or 512-d embeddings (vectors) အဖြစ်ပြောင်းဖို့အတွက် သုံးပါတယ်။ အဲ့ embddeding ကို face comparison , clustering, classification တို့ မှာသုံးရပါတယ်။ 
```
| Component           | Role             | Typical Source                                                   |
| ------------------- | ---------------- | ---------------------------------------------------------------- |
| **MTCNN**           | Detect/crop face | `facenet-pytorch`                                                |
| **Embedding model** | Get face vector  | `InceptionResnetV1` trained on **VGGFace2** or **CASIA-WebFace** |

```bash
pip install facenet-pytorch
```

### REF: Link
https://github.com/timesler/facenet-pytorch

```bash
facenet-pytorch is designed for embedding extraction, not direct classification.
You need to:

Extract embeddings for all faces in your dataset using base_model.
Train a classifier (e.g., SVM, logistic regression, or a small neural network) on these embeddings.
```
```bash
SVM stands for Support Vector Machine.
It is a supervised machine learning algorithm used for classification and regression tasks.

How it works:
SVM finds the best boundary (hyperplane) that separates data points of different classes with the largest margin.
In face recognition:
After extracting embeddings (feature vectors) from images, SVM can be trained to classify which person each embedding belongs to.
Summary:
SVM is a popular and effective classifier, especially for high-dimensional data like face embeddings.

```

| Step | Description                                     |
| ---- | ----------------------------------------------- |
| 1    | Install dependencies                            |
| 2    | Organize face dataset in folders                |
| 3    | Use MTCNN to extract faces                      |
| 4    | Generate FaceNet embeddings                     |
| 5    | Train SVM classifier on embeddings              |
| 6    | Predict labels on new faces using FaceNet + SVM |


### 2.3 Object Regonition YOLO

```bash
C:/Users/bot/AppData/Local/Programs/Python/Python310/python.exe d:/00_google_classroom/machinelearning.ai/02_transfer_learning/03_yolo/009_train_valid_split_windows.py --datapath "D:\Project12_Yolo_Face\data" --train_pct 0.9
```

```bash
& C:/Users/bot/AppData/Local/Programs/Python/Python310/python.exe d:/00_google_classroom/machinelearning.ai/02_transfer_learning/03_yolo/009_data_yaml.py
Created config file at D:\Project12_Yolo_Face\data\data.yaml
```


#### 2.3.1 Setup testing env on mac m1 for yolo model

https://conda-forge.org/download/

```bash
install conda-forge
```

```bash
echo ". ~/miniforge3/etc/profile.d/conda.sh" >> ~/.zshrc
source ~/.zshrc
conda --version
```

#### 2.3.2 Testing with command line

```bash
yolo task=detect mode=predict model=best.pt source=0 imgsz=640 device=cpu
```

##### data.yaml: 
(Input) Describes your dataset (paths to images/labels, class names, number of classes).

##### model.yaml: 
(Input, or generated if using a pre-trained model as a base and saving its architecture) Defines the architecture of the YOLO model (number of layers, channels, etc.).

##### args.yaml (or similar, outputted by Ultralytics): 
After a training run, Ultralytics YOLO often saves an args.yaml file within the run directory. This file captures all the arguments that were used for that specific training run, making it easy to reproduce the experiment or understand exactly what settings were applied.

### 3.0 Inference Yolo Models On ...

#### 3.1 Exported Model Formats (e.g., .onnx, .tflite, .engine)
What it is: These are optimized versions of your trained model, converted from the original PyTorch format (.pt) into formats more suitable for deployment on specific hardware or software environments. This conversion often involves optimizations for inference speed and memory footprint.

##### Format: 
Varies depending on the target platform.

Location: Typically in runs/detect/export/your_run_name/ or directly in the weights/ folder if exported from there.

##### Common Export Formats:

##### 3.2.1 filename.onnx: 

ONNX (Open Neural Network Exchange) is an open standard for representing machine learning models. It allows models trained in one framework (like PyTorch) to be easily used in another (like TensorFlow, ONNX Runtime). Highly portable.

filename.tflite: TensorFlow Lite. Optimized for mobile and embedded devices.

filename.engine: NVIDIA TensorRT engine. Highly optimized for NVIDIA GPUs, offering maximum inference speed on these devices.

filename.mlmodel: Apple CoreML. For deployment on Apple devices (iOS, macOS).

filename.xml and filename.bin: OpenVINO format. For Intel hardware.

filename.pth: Sometimes models are saved in .pth which is also a PyTorch format, similar to .pt.

#####  3.2.2 Usage:

Deployment: Load these models into your application for real-time inference on various edge devices, servers, or web services.

Framework Interoperability: Use ONNX to easily move models between different deep learning frameworks.