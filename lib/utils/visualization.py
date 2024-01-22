import cv2
from copy import copy

"""                
    "Shoulder Left",
    "Shoulder Right",
    "Elbow Left",
    "Hand Left",
    "Elbow Right",
    "Hand Right",
    "Foot Left",
    "Foot Right",
    "Hip",
    "Neck"
"""

left_color = (255, 0, 219)
center_color = (0, 228, 255)
right_color = (0, 0, 255)

colors = [
    left_color,
    right_color,
    left_color,
    left_color,
    right_color,
    right_color,
    left_color,
    right_color,
    center_color,
    center_color
]


def draw_skeleton(frame, pred, threshold=0.5, skeleton=[[0,2],[9,8],[8,6],[9,1],[4,5],[9,0],[2,3],[1,4],[8,7]]):
    frame = copy(frame)
    centers = [(int(point[0]), int(point[1])) for point in pred]
    scores = [point[2] for point in pred]
    for i, point in enumerate(centers):
        if scores[i] > threshold:
            frame = cv2.circle(frame, point, 2, colors[i])
            # frame = cv2.putText(frame, f"{int(scores[i]*100)}%", point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    for p1, p2 in skeleton:
        if scores[p1] > threshold and scores[p2] > threshold:
            color = [0, 0, 0]
        else:
            color = [255, 255, 255]
        frame = cv2.line(frame, centers[p1], centers[p2], color)
    return frame


def draw_rectangle(frame, pred):
    x, y, w, h, = pred
    start_point = (int(x - w//2), int(y - h//2))
    end_point = (int(x + w//2), int(y + h//2))
    # start_point = (10, 10)
    # end_point = (200, 200)
    return cv2.rectangle(frame, start_point, end_point, 1, 2) 
