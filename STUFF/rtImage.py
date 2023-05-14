import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('yolov4.h5')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

class_names = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck', 'traffic light', 'stop sign', 'parking meter']

class_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0), (128, 128, 128)]

def detect_objects(image, model):
    image = cv2.resize(image, (416, 416))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    outputs = model.predict(image)

    boxes, scores, classes = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(outputs[0][:, :, :, :4], (tf.shape(outputs[0])[0], -1, 1, 4)),
        scores=tf.reshape(outputs[0][:, :, :, 4:], (tf.shape(outputs[0])[0], -1, tf.shape(outputs[0])[3])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.5,
        score_threshold=0.5
    )


    height, width, _ = image.shape
    boxes = boxes * np.array([width, height, width, height])

    
    for i in range(len(boxes)): # Draw the boxes on the image
        x_min, y_min, x_max, y_max = boxes[i]
        class_id = int(classes[i])
        class_name = class_names[class_id]
        color = class_colors[class_id]
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(image, f'{class_name} {scores[i]:.2f}', (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return image

# Start the video capture and detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    output = detect_objects(frame, model)
    cv2.imshow('Object Detection', output)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release() # Release the video capture and close the window
cv2.destroyAllWindows()
