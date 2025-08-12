# Staff Tracking Using YOLOv5 with RKNN device
Using yolov5 converted model from here https://github.com/airockchip/rknn-toolkit2

## Instruction to run
### Make sure that your Orange Pi 5 series RK3588 using rknpu > 0.9.8
Run
```bash
pip install -r requirements.txt
```
To install the deps.

Run 
```bash
python test.py
```
To test with test video `test.mp4`. Change your video or use RTSP to realtime staff checking

## Demo
See the video `output.mp4`

## Requirements

* **python**
* **opencv**
* **YOLO**
* **RKNN-Toolkit**

