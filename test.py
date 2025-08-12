"""This code was written by JimmyNguyen09-AI, give me the stars or forks if you fell it good:))
Any question please send me by the contact already linked on my github account. Thanks"""

import cv2
import time
import numpy as np
from rknn.api import RKNN

PERSON_ID = 0
CHAIR_ID = 56

def compute_iou(boxA, boxB)->float:
    """
    :param boxA:
    :param boxB:
    :return: float
    Calculate the iou score by the formula: IoU = interArea / (areaA + areaB - interArea)
    You can also calculate the coordinates of the central point of the person's bounding
    box then consider it in the bounding box of the chair
    But apply Iou score gives better performance :>
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    """Find 2 edges of overlap rectangle"""
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    """
    Calculate S of 2 areas, 1 is the value of 1 pixel, remove it is fine!
    """
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(areaA + areaB - interArea)
    return iou

# RKNN model initialization
def init_rknn_model(model_path, target_platform='rk3588'):
    # Create RKNN object
    rknn = RKNN(verbose=True)
    
    # Load RKNN model
    print('Loading RKNN model: ' + model_path)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    
    # Init runtime environment
    print('Init runtime environment')
    ret = rknn.init_runtime(target=target_platform)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    
    return rknn

# YOLOv5 post-processing
def yolov5_post_process(input_data, anchors, stride, num_classes=80, conf_thres=0.4):
    # Reshape input data
    outputs = []
    for i, feat in enumerate(input_data):
        # Get grid and anchor grid
        batch_size, n_ch, grid_h, grid_w = feat.shape
        n_anchors = len(anchors[i])
        
        # Reshape output
        feat = feat.reshape(batch_size, n_anchors, 5 + num_classes, grid_h, grid_w)
        feat = feat.transpose(0, 1, 3, 4, 2)  # (batch_size, n_anchors, grid_h, grid_w, 5+num_classes)
        
        # Create grid
        grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
        grid = np.stack((grid_x, grid_y), axis=2).reshape(1, 1, grid_h, grid_w, 2)
        
        # Box prediction
        box_xy = (feat[..., :2] * 2 - 0.5 + grid) * stride[i]
        box_wh = (feat[..., 2:4] * 2) ** 2 * np.array(anchors[i]).reshape(1, n_anchors, 1, 1, 2)
        
        # Combine predictions
        box = np.concatenate((box_xy, box_wh), axis=-1)
        conf = feat[..., 4:5] * feat[..., 5:]
        cls = np.argmax(conf, axis=-1, keepdims=True)
        conf = np.max(conf, axis=-1, keepdims=True)
        pred = np.concatenate((box, conf, cls.astype(np.float32)), axis=-1)
        
        # Filter by confidence threshold
        mask = pred[..., 4] > conf_thres
        pred = pred[mask]
        
        if len(pred) > 0:
            outputs.append(pred)
    
    # Combine all predictions
    if len(outputs) > 0:
        outputs = np.concatenate(outputs, axis=0)
        # Convert to [x1, y1, x2, y2, conf, cls]
        boxes = outputs[:, :4].copy()
        boxes[:, 0] = outputs[:, 0] - outputs[:, 2] / 2  # x1 = x - w/2
        boxes[:, 1] = outputs[:, 1] - outputs[:, 3] / 2  # y1 = y - h/2
        boxes[:, 2] = outputs[:, 0] + outputs[:, 2] / 2  # x2 = x + w/2
        boxes[:, 3] = outputs[:, 1] + outputs[:, 3] / 2  # y2 = y + h/2
        
        # Final output format: [x1, y1, x2, y2, conf, cls]
        outputs = np.concatenate((boxes, outputs[:, 4:6]), axis=1)
        return outputs
    else:
        return np.array([])

# Main function
def main():
    # RKNN model path - update this to your model path
    rknn_model = '../yolov5/yolov5m.rknn'  # Change to your RKNN model path
    target_platform = 'rk3588'  # Change to your target platform
    
    # Initialize RKNN model
    rknn = init_rknn_model(rknn_model, target_platform)
    
    # YOLOv5 anchors (make sure these match your model)
    anchors = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]]
    ]
    stride = [8, 16, 32]  # Strides for each feature map
    
    # Video path
    video_path = '../test.mp4'  # Change the input video here
    cap = cv2.VideoCapture(video_path)
    
    # Initialize chair tracking variables
    chair_timers = []
    chair_elapsed = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 640
    height = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('../yolov5/output.mp4', fourcc, fps, (width, height))  # Output video path
    
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        
        # Preprocess image for RKNN
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Inference with RKNN
        outputs = rknn.inference(inputs=[img])
        
        # Post-process outputs
        detections = yolov5_post_process(outputs, anchors, stride, conf_thres=0.4)
        
        persons = []
        chairs = []
        
        # Process detections
        if len(detections) > 0:
            for detec in detections:
                x1, y1, x2, y2, conf, cls = detec
                cls = int(cls)
                box = [int(x1), int(y1), int(x2), int(y2)]
                if cls == PERSON_ID:
                    persons.append(box)
                elif cls == CHAIR_ID:
                    chairs.append(box)
        
        # Draw persons
        for person in persons:
            cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), (0, 255, 0), 2)
            cv2.putText(frame, "", (person[0], person[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Initialize chair timers if needed
        if len(chair_timers) != len(chairs):
            chair_timers = [None] * len(chairs)
            chair_elapsed = [0] * len(chairs)
        
        # Check if chairs have persons
        chair_has_person = [False] * len(chairs)
        for i, chair in enumerate(chairs):
            for p in persons:
                iou = compute_iou(chair, p)
                if iou > 0.05:
                    chair_has_person[i] = True
                    break
        
        # Update chair timers and draw chairs
        for i, c in enumerate(chairs):
            if chair_has_person[i]:
                chair_timers[i] = None
                chair_elapsed[i] = 0
            else:
                if chair_timers[i] is None:
                    chair_timers[i] = time.time()
                chair_elapsed[i] = time.time() - chair_timers[i]
                cv2.rectangle(frame, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 2)
                cv2.putText(frame, f"Empty Chair", (c[0], c[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Away: {int(chair_elapsed[i])}s", (c[0], c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
        
        # Display and save frame
        cv2.imshow('Staff Tracking', frame)
        out.write(frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main()
