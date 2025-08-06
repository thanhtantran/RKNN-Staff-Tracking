"""This code was written by JimmyNguyen09-AI, give me the stars or forks if you fell it good:))
Any question please send me by the contact already linked on my github account. Thanks"""

import torch
import cv2
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/yolov5m.pt')  #Change the checkpoint path
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


video_path = '../test.mp4' #Change the input video here
cap = cv2.VideoCapture(video_path)

chair_timers = []
chair_elapsed = []
fps = cap.get(cv2.CAP_PROP_FPS)
width = 640
height = 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../yolov5/output.mp4', fourcc, fps, (width, height)) #Output video path
while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        break
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]
    detections = detections[detections[:, 4] > 0.4]

    persons = []
    chairs = []

    for detec in detections:
        x1, y1, x2, y2, conf, cls = detec
        cls = int(cls.item())
        box = [int(x1), int(y1), int(x2), int(y2)]
        if cls == PERSON_ID:
            persons.append(box)
        elif cls == CHAIR_ID:
            chairs.append(box)
    for person in persons:
        cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), (0, 255, 0), 2)
        cv2.putText(frame, "", (person[0], person[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if len(chair_timers) != len(chairs):
        chair_timers = [None] * len(chairs)
        chair_elapsed = [0] * len(chairs)

    chair_has_person = [False] * len(chairs)
    for i, chair in enumerate(chairs):
        for p in persons:
            iou = compute_iou(chair, p)
            if iou > 0.05:
                chair_has_person[i] = True
                break
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

    cv2.imshow('Staff Tracking', frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
