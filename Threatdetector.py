import os
import shutil
import random
import cv2
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
import torch
import numpy as np

# Paths for COCO dataset
coco_train_img_dir = 'C:/Studies/3rd_sem_MS/Machine_Learning/Project/2_Data_Preparation/train2017/'
coco_val_img_dir = 'C:/Studies/3rd_sem_MS/Machine_Learning/Project/2_Data_Preparation/val2017/'
coco_train_annotations = 'C:/Studies/3rd_sem_MS/Machine_Learning/Project/2_Data_Preparation/annotations_trainval2017/instances_train2017.json'
coco_val_annotations = 'C:/Studies/3rd_sem_MS/Machine_Learning/Project/2_Data_Preparation/annotations_trainval2017/instances_val2017.json'

dataset_dir = 'dataset/'
os.makedirs(dataset_dir, exist_ok=True)

# Create directories for YOLO annotations
train_label_dir = 'dataset/labels/train/'
val_label_dir = 'dataset/labels/val/'
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

def convert_coco_to_yolo(coco_json_path, img_dir, output_label_dir):
    coco = COCO(coco_json_path)
    for img_id in coco.imgs:
        img_info = coco.loadImgs(img_id)[0]
        img_file = os.path.join(img_dir, img_info['file_name'])
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        img_width, img_height = img_info['width'], img_info['height']
        label_file = os.path.join(output_label_dir, f"{img_info['file_name'][:-4]}.txt")
        with open(label_file, "w") as f:
            for ann in annotations:
                category_id = ann['category_id'] - 1
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

convert_coco_to_yolo(coco_train_annotations, coco_train_img_dir, train_label_dir)
convert_coco_to_yolo(coco_val_annotations, coco_val_img_dir, val_label_dir)
print("Conversion completed. Labels saved in dataset/labels/train and dataset/labels/val.")

# Split images into train/val folders
train_img_dir = 'train/'
val_img_dir = 'dataset/images/val/'
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

for img in os.listdir(coco_train_img_dir):
    shutil.copy(os.path.join(coco_train_img_dir, img), os.path.join(train_img_dir, img))
for img in os.listdir(coco_val_img_dir):
    shutil.copy(os.path.join(coco_val_img_dir, img), os.path.join(val_img_dir, img))
print("Completed image split.")

# Data augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(640, 640),
    ToTensorV2(),
])

def augment_and_save_image(image_path, output_image_dir, output_label_dir, label_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    augmented = transform(image=image)['image']
    augmented = augmented.permute(1, 2, 0).numpy()
    augmented = (augmented * 255).astype('uint8')
    augmented = 255 - augmented
    file_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_image_dir, f"aug_{file_name}")
    cv2.imwrite(output_image_path, augmented)
    if label_path and os.path.exists(label_path):
        output_label_path = os.path.join(output_label_dir, f"aug_{os.path.basename(label_path)}")
        if not os.path.exists(output_label_path):
            shutil.copy(label_path, output_label_path)
    print(f"Augmented image saved to {output_image_path}")

aug_label_dir = 'train/'
output_image_dir = 'dataset/images/train/'
output_label_dir = 'dataset/labels/train/'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

for image_name in os.listdir(aug_label_dir):
    image_path = os.path.join(aug_label_dir, image_name)
    label_path = os.path.join(output_label_dir, image_name.replace('.jpg', '.txt'))
    augment_and_save_image(image_path, output_image_dir, output_label_dir, label_path)
print("Data Augmentation Completed")

# Visualize bounding boxes
def visualize_yolo_bboxes(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image at {img_path}")
        return
    img_height, img_width = img.shape[:2]
    with open(label_path, "r") as f:
        labels = f.readlines()
    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label.split())
        x_center = int(x_center * img_width)
        y_center = int(y_center * img_height)
        width = int(width * img_width)
        height = int(height * img_height)
        top_left_x = int(x_center - width / 2)
        top_left_y = int(y_center - height / 2)
        img = cv2.rectangle(img, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), (255, 0, 0), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    import matplotlib.pyplot as plt
    plt.imshow(img_rgb)
    plt.show()

# Create data.yaml for YOLOv5
data_yaml_path = os.path.join(os.getcwd(), 'data.yaml')
data_yaml_content = {
    'train': 'C:/Users/iampr/Machine_Learning/dataset/images/train/',
    'val': 'C:/Users/iampr/Machine_Learning/dataset/images/val/',
    'nc': 91,
    'names': {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
        9: 'trafficlight', 10: 'firehydrant', 11: 'streetsign', 12: 'stopsign', 13: 'parkingmeter', 14: 'bench', 15: 'bird',
        16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
        25: 'hat', 26: 'backpack', 27: 'umbrella', 28: 'shoe', 29: 'eyeglasses', 30: 'handbag', 31: 'tie', 32: 'suitcase',
        33: 'frisbee', 34: 'skis', 35: 'snowboard', 36: 'sportsball', 37: 'kite', 38: 'baseballbat', 39: 'baseballglove',
        40: 'skateboard', 41: 'surfboard', 42: 'tennisracket', 43: 'bottle', 44: 'plate', 45: 'wineglass', 46: 'cup',
        47: 'fork', 48: 'knife', 49: 'spoon', 50: 'bowl', 51: 'banana', 52: 'apple', 53: 'sandwich', 54: 'orange',
        55: 'broccoli', 56: 'carrot', 57: 'hotdog', 58: 'pizza', 59: 'donut', 60: 'cake', 61: 'chair', 62: 'sofa',
        63: 'pottedplant', 64: 'bed', 65: 'mirror', 66: 'diningtable', 67: 'window', 68: 'desk', 69: 'toilet', 70: 'door',
        71: 'tvmonitor', 72: 'laptop', 73: 'mouse', 74: 'remote', 75: 'keyboard', 76: 'cellphone', 77: 'microwave',
        78: 'oven', 79: 'toaster', 80: 'sink', 81: 'refrigerator', 82: 'blender', 83: 'book', 84: 'clock', 85: 'vase',
        86: 'scissors', 87: 'teddybear', 88: 'hairdrier', 89: 'toothbrush', 90: 'hairbrush'
    }
}
with open(data_yaml_path, 'w') as file:
    yaml.dump(data_yaml_content, file, default_flow_style=False)
print("data.yaml file created")

# --- YOLOv5 Training and Inference (run these commands in terminal or notebook) ---

# Clone YOLOv5 repository (run in terminal)
# git clone https://github.com/ultralytics/yolov5.git

# Change directory and install requirements (run in terminal)
# cd yolov5
# pip install -r requirements.txt

# Train YOLOv5 (run in terminal)
# python train.py --img 640 --batch 16 --epochs 10 --data ../data.yaml --weights yolov5m.pt --project runs/train --name exp --device 0

# Detect objects in an image (run in terminal)
# python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source ../indoor.jpg --data ../data.yaml

# Detect objects in another image (run in terminal)
# python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source ../outdoor.jpg --data ../data.yaml

# --- Optional: Monitor Training Progress with TensorBoard (run in notebook) ---
# %load_ext tensorboard
# %tensorboard --logdir runs/train --port 6006

# --- Optional: Check CUDA Availability ---
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# --- Optional: Display Output Images (requires IPython/Jupyter) ---
def display_detected_image(image_path):
    from IPython.display import Image, display
    display(Image(filename=image_path, width=600))

# Example usage:
# display_detected_image('runs/detect/exp/indoor.jpg')
# display_detected_image('runs/detect/exp/outdoor.jpg')



# Load YOLOv5 model (make sure weights path is correct)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True)
model.conf = 0.25  # confidence threshold

def preprocess(frame, size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img

def postprocess(results, orig_shape, input_shape):
    boxes = results.xyxy[0].cpu().numpy()
    h_ratio = orig_shape[0] / input_shape[0]
    w_ratio = orig_shape[1] / input_shape[1]
    processed = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1 = int(x1 * w_ratio)
        x2 = int(x2 * w_ratio)
        y1 = int(y1 * h_ratio)
        y2 = int(y2 * h_ratio)
        processed.append((x1, y1, x2, y2, conf, int(cls)))
    return processed

def run_inference_on_video(input_path, output_path, fps=30):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_img = preprocess(frame)
        results = model(input_img)
        detections = postprocess(results, frame.shape[:2], input_img.shape[:2])

        # Draw boxes
        for x1, y1, x2, y2, conf, cls in detections:
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Inference complete. Output saved to {output_path}")

# Example usage:
# run_inference_on_video('input.mp4', 'output.mp4', fps=30)