import cv2
import numpy as np   
from tracker import Tracker  # Implement a person tracker or use an existing one

# Load your video streams from two cameras
camera1 = cv2.VideoCapture("camera1.mp4")
camera2 = cv2.VideoCapture("camera2.mp4")  
# Initialize your tracker
tracker = Tracker() 


# Initialize an empty dictionary to store person information
persons = {}

# Camera information (you need to fill in these values)
camera1_info = {
    'position': (0, 0, 0),  # x, y, z coordinates
    'view_area': (1920, 1080)  # Total area of the camera's view
}

camera2_info = {
    'position': (5, 0, 0),  # x, y, z coordinates
    'view_area': (1920, 1080)  # Total area of the camera's view
}

def calculate_distance(bbox, camera_info):
    # Calculate the center of the bounding box
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    # Calculate the Euclidean distance
    distance = np.sqrt((bbox_center_x - camera_info['position'][0]) ** 2 +
                       (bbox_center_y - camera_info['position'][1]) ** 2)

    return distance

def calculate_visibility(bbox, camera_info):
    # Calculate the area of the person's bounding box
    person_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    # Calculate the intersection area with the camera's view
    intersection_x1 = max(bbox[0], 0)
    intersection_y1 = max(bbox[1], 0)
    intersection_x2 = min(bbox[2], camera_info['view_area'][0])
    intersection_y2 = min(bbox[3], camera_info['view_area'][1])

    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

    # Calculate visibility ratio
    visibility = intersection_area / person_area

    return visibility 
def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height, frame_width = frame.shape[:2]

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detections = [] 
    print("indices", indices)
    for i in indices: 
        x, y, w, h = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        detections.append((x, y, w, h, confidence, class_id))

    return detections



def draw_bboxes_and_ids(frame, persons):
    for person_id, person_info in persons.items():
        bbox1 = person_info['bbox1']
        bbox2 = person_info['bbox2']
        best_view = person_info['best_view']

        # Draw bounding boxes
        cv2.rectangle(frame, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), (0, 0, 255), 2)

        # Draw person ID and best view
        cv2.putText(frame, f"Person {person_id}", (bbox1[0], bbox1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Person {person_id}", (bbox2[0], bbox2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Best View: {best_view}", (bbox1[0], bbox1[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Load YOLO model and its configuration and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define the YOLO output layers
output_layers = net.getUnconnectedOutLayersNames()

# Confidence threshold and NMS threshold
confidence_threshold = 0.5
nms_threshold = 0.4

# Define the update_persons_dictionary function
def update_persons_dictionary(detections, camera_key, persons_dict):
    for detection in detections:
        x, y, w, h, confidence,class_id = detection

        person_id = len(persons_dict) + 1
        if person_id not in persons_dict:
            persons_dict[person_id] = {'bbox1': None, 'bbox2': None}
        
        persons_dict[person_id][camera_key] = (x, y, x + w, y + h)

while True:
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    if not ret1 or not ret2:
        break

    # Detection on camera 1
    blob1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob1)
    outs1 = net.forward(output_layers)
    detections1 = post_process(frame1, outs1, confidence_threshold, nms_threshold)
    update_persons_dictionary(detections1, 'bbox1', persons)

    # Detection on camera 2
    blob2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob2)
    outs2 = net.forward(output_layers)
    detections2 = post_process(frame2, outs2, confidence_threshold, nms_threshold)
    update_persons_dictionary(detections2, 'bbox2', persons)

    # Update the tracker with the detected persons
    tracker.update(persons, frame1, frame2)

    # Calculate distances and visibilities for each person
    for person_id, person_info in persons.items():
        person_info['distance1'] = calculate_distance(person_info['bbox1'], camera1_info)
        person_info['distance2'] = calculate_distance(person_info['bbox2'], camera2_info)
        person_info['visibility1'] = calculate_visibility(person_info['bbox1'], camera1_info)
        person_info['visibility2'] = calculate_visibility(person_info['bbox2'], camera2_info)

       # Determine the best view for each person based on distance and visibility
        if person_info['distance1'] < person_info['distance2']:
            if person_info['visibility1'] > person_info['visibility2']:
                person_info['best_view'] = 'Camera 1'
            else:
                person_info['best_view'] = 'Camera 2'
        else:
            if person_info['visibility2'] > person_info['visibility1']:
                person_info['best_view'] = 'Camera 2'
            else:
                person_info['best_view'] = 'Camera 1'


    # Draw bounding boxes, IDs, and best views on the frames
    draw_bboxes_and_ids(frame1, persons)
    draw_bboxes_and_ids(frame2, persons)

    # Display the frames
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    # Exit the loop if a key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video streams and close windows
camera1.release()
camera2.release()
cv2.destroyAllWindows()
