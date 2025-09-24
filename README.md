# Threat-Detector: Object Detection with YOLOv5 and COCO Dataset

This project prepares the COCO dataset, converts annotations to YOLO format, performs data augmentation, trains a YOLOv5 model for object detection, and runs real-time video inference.

## Project Structure

- `ObjectDetection.ipynb`: Main notebook for data preparation, augmentation, training, and inference.
- `Threatdetector.py`: End-to-end pipeline including dataset prep, augmentation, training config, and video inference.
- `data.yaml`: Configuration file for YOLOv5 training (paths, class names).
- `dataset/`: Contains images and labels in YOLO format.

## Setup

1. **Install Dependencies**

Run the following in your notebook or terminal:
```sh
pip install pycocotools opencv-python albumentations tqdm torch torchvision torchaudio
```

2. **Prepare COCO Dataset**

Download COCO 2017 Train/Val images and annotations from [COCO Dataset](http://cocodataset.org/#download). Update paths in the notebook or script if needed.

3. **Convert COCO Annotations to YOLO Format**

The code converts COCO JSON annotations to YOLO `.txt` files using:
```python
convert_coco_to_yolo(coco_train_annotations, coco_train_img_dir, train_label_dir)
convert_coco_to_yolo(coco_val_annotations, coco_val_img_dir, val_label_dir)
```

4. **Split Images**

Training and validation images are copied to:
- `train/`
- `dataset/images/val/`

5. **Data Augmentation**

Augmentation is performed using [Albumentations](https://albumentations.ai/), including flips, brightness/contrast, resizing, and color inversion.

6. **Create `data.yaml`**

The script generates `data.yaml` with 91 COCO classes and image paths.

## Training YOLOv5

1. **Clone YOLOv5**
```sh
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

2. **Train**
```sh
python train.py --img 640 --batch 16 --epochs 10 --data ../data.yaml --weights yolov5m.pt --project runs/train --name exp --device 0
```

## Inference on Images

Detect objects in images:
```sh
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source ../indoor.jpg --data ../data.yaml
```

## **Video Inference**

Run real-time object detection on video files at 30 FPS using OpenCV and YOLOv5:

```python
from Threatdetector import run_inference_on_video

# Example usage:
run_inference_on_video('input.mp4', 'output.mp4', fps=30)
```
- This will process `input.mp4`, draw bounding boxes, and save the result to `output.mp4`.

## Monitoring

Monitor training progress with TensorBoard:
```sh
%load_ext tensorboard
%tensorboard --logdir runs/train --port 6006
```

## Visualization

Visualize bounding boxes using the provided function in the notebook or script.

## License

See [YOLOv5 License](https://github.com/ultralytics/yolov5/blob/master/LICENSE).

**Note:** Update paths in `data.yaml` and the scripts as needed for your environment.
