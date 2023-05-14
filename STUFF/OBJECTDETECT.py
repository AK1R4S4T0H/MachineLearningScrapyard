import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = 'path/to/frozen_inference_graph.pb'
PATH_TO_LABELS = 'path/to/label_map.pbtxt'
NUM_CLASSES = 90

# Load the frozen inference graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Load the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Open a video stream
cap = cv2.VideoCapture(0)

# Run the object detection model in real-time
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      # Read a frame from the video stream
      ret, image_np = cap.read()

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

      # Get handles to input and output tensors
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in detection_graph.get_operations():
          tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)

      # Run inference
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

      # Visualize the results
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(output_dict['detection_boxes']),
          np.squeeze(output_dict['detection_classes']).astype(np.int32),
          np.squeeze(output_dict['detection_scores']),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      # Display the resulting image
      cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

      # Press 'q' to quit
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
