import os
import cv2
import time
import numpy as np
from rknn.api import RKNN

# Constants
PERSON_ID = 0
CHAIR_ID = 56  # Chair class ID in COCO dataset
OBJ_THRESH = 0.4  # Object detection threshold
NMS_THRESH = 0.45  # Non-maximum suppression threshold
IMG_SIZE = (640, 640)  # (width, height)

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

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:,:2,:,:]*2 - 0.5
    box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

    return xyxy

def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []
    # 1*255*h*w -> 3*85*h*w
    input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:,:4,:,:], anchors[i]))
        scores.append(input_data[i][:,4:5,:,:])
        classes_conf.append(input_data[i][:,5:,:,:])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

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

def main():
    # RKNN model path - update this to your model path
    rknn_model = './yolov5/yolov5.rknn'  # Change to your RKNN model path
    target_platform = 'rk3588'  # Change to your target platform
    
    # Load anchors from file
    anchors_file = './anchors_yolov5.txt'
    with open(anchors_file, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3,-1,2).tolist()
    print(f"Using anchors from '{anchors_file}', which is {anchors}")
    
    # Initialize RKNN model
    rknn = init_rknn_model(rknn_model, target_platform)
    
    # Video path
    video_path = 'test.mp4'  # Change the input video here
    cap = cv2.VideoCapture(video_path)
    
    # Initialize chair tracking variables
    chair_timers = []
    chair_elapsed = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 640
    height = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))  # Output video path
    
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        
        # Resize frame for display and output
        display_frame = cv2.resize(frame, (width, height))
        
        # Preprocess image for RKNN - resize to 640x640 with letterboxing
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Create a square canvas of 640x640
        canvas = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        
        # Calculate scaling factor to fit the image within 640x640 while preserving aspect ratio
        scale = min(IMG_SIZE[0]/w, IMG_SIZE[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize the image
        resized_img = cv2.resize(img, (new_w, new_h))
        
        # Calculate padding
        dw, dh = (IMG_SIZE[0] - new_w) // 2, (IMG_SIZE[1] - new_h) // 2
        
        # Place the resized image on the canvas
        canvas[dh:dh+new_h, dw:dw+new_w, :] = resized_img
        
        # Inference with RKNN
        outputs = rknn.inference(inputs=[canvas])
        
        # Post-process outputs
        boxes, classes, scores = post_process(outputs, anchors)
        
        persons = []
        chairs = []
        
        # Process detections and scale boxes back to original frame size
        if boxes is not None:
            for box, score, cl in zip(boxes, scores, classes):
                # Scale boxes from 640x640 to original frame size
                x1, y1, x2, y2 = [int(_b) for _b in box]
                
                # Remove padding offset
                x1 = (x1 - dw) / scale
                y1 = (y1 - dh) / scale
                x2 = (x2 - dw) / scale
                y2 = (y2 - dh) / scale
                
                # Clip to frame boundaries
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                # Scale to display frame size
                x1, y1 = int(x1 * width / w), int(y1 * height / h)
                x2, y2 = int(x2 * width / w), int(y2 * height / h)
                
                cls = int(cl)
                box_coords = [x1, y1, x2, y2]
                if cls == PERSON_ID:
                    persons.append(box_coords)
                elif cls == CHAIR_ID:
                    chairs.append(box_coords)
        
        # Draw persons
        for person in persons:
            cv2.rectangle(display_frame, (person[0], person[1]), (person[2], person[3]), (0, 255, 0), 2)
            cv2.putText(display_frame, "", (person[0], person[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
                cv2.rectangle(display_frame, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Empty Chair", (c[0], c[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Away: {int(chair_elapsed[i])}s", (c[0], c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
        
        # Display and save frame
        cv2.imshow('Staff Tracking', display_frame)
        out.write(display_frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main()
