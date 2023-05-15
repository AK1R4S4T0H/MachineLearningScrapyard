# 
import tensorflow as tf
from tensorflow import keras
from keras import Layers
from keras import backend as K
 

input_shape = (224, 224, 3)
num_classes = 21841
batch_size = 16

def yolo_v3(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Backbone
    x = keras.Layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Neck
    x = keras.layers.Conv2D(512, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    neck_output = keras.layers.Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same')(x)
    
    # Head
    x = keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(neck_output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), activation='softmax', padding='same')(x)
    head_output = keras.layers.GlobalAveragePooling2D()(x)


    num_anchors = 3
    anchors = [(10, 13), (16, 30), (33, 23)]
    num_classes = num_classes
    
    x = keras.layers.Conv2D(num_anchors * (5 + num_classes), kernel_size=(1, 1), padding='same')(neck_output)
    x = keras.layers.Reshape((input_shape[0]//32, input_shape[1]//32, num_anchors, 5 + num_classes))(x)
    grid_size = keras.backend.shape(x)[1:3]
    x = keras.layers.Lambda(lambda x: x / tf.cast(grid_size[::-1], tf.float32))(x)
    
    box_xy, box_wh, objectness, class_probs = tf.split(x, (2, 2, 1, num_classes), axis=-1)
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_boxes = tf.concat([box_xy, box_wh], axis=-1)
    
    grid_y = tf.range(grid_size[0])
    grid_x = tf.range(grid_size[1])
    a, b = tf.meshgrid(grid_x, grid_y)
    grid = tf.stack([a, b], axis=-1)
    grid = tf.cast(grid, tf.float32)
    

    box_xy = (box_xy + grid) * 32
    box_wh = tf.exp(box_wh) * anchors
    box_wh = box_wh * 32
    

    pred_boxes = tf.concat([box_xy, box_wh], axis=-1)
    output_tensor = tf.concat([pred_boxes, objectness, class_probs], axis=-1)
    
    return keras.Model(inputs, output_tensor)




def yolov3_head(inputs):
    num_anchors = 3
    anchors = [(10, 13), (16, 30), (33, 23)]
    
    x = keras.layers.Conv2D(num_anchors * (5 + num_classes), kernel_size=(1, 1), padding='same')(inputs)
    
    x = keras.layers.Reshape((input_shape[0]//32, input_shape[1]//32, num_anchors, 5 + num_classes))(x)
    
    grid_size = keras.backend.shape(x)[1:3]
    
    x = keras.layers.Lambda(lambda x: x / tf.cast(grid_size[::-1], tf.float32))(x)
    
    
    box_xy, box_wh, objectness, class_probs = tf.split(x, (2, 2, 1, num_classes), axis=-1)
    
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    
    class_probs = tf.sigmoid(class_probs)
    
    pred_boxes = tf.concat([box_xy, box_wh], axis=-1)
    
    grid_y = tf.range(grid_size[0])
    grid_x = tf.range(grid_size[1])
    a, b = tf.meshgrid(grid_x, grid_y)
    grid = tf.stack([a, b], axis=-1)
    grid = tf.cast(grid, tf.float32)
    
    pred_boxes = tf.concat([pred_boxes[..., :2] + grid, pred_boxes[..., 2:]], axis=-1)
    
    anchors = tf.cast(tf.constant(anchors), tf.float32)
    anchor_xy = (anchors[..., ::-1] / 2.) / tf.cast(grid_size[::-1], tf.float32)
    anchor_wh = anchors[..., ::-1] / tf.cast(input_shape[::-1], tf.float32)
    anchors = tf.concat([anchor_xy, anchor_wh], axis=-1)
    anchors = tf.tile(anchors, [tf.shape(x)[0], 1, 1, 1, 1])
    
    
    pred_boxes = tf.expand_dims(pred_boxes, 4)
    anchor_boxes = tf.expand_dims(anchors, 0)
    pred_boxes_xy = pred_boxes[..., :2]
    pred_boxes_wh = pred_boxes[..., 2:4]

        
    best_anchor = tf.argmax(iou, axis=-1)
    best_anchor = tf.cast(best_anchor, tf.float32)
    best_anchor = tf.expand_dims(best_anchor, axis=-1)
    
    mask = tf.one_hot(tf.cast(best_anchor, tf.int32), depth=num_anchors)
    no_object_mask = 1.0 - objectness * mask
    
    objectness = objectness + 1e-7
    
    box_loss_scale = 2.0 - inputs[..., 2:3] * inputs[..., 3:4]
    xy_loss = tf.reduce_sum(tf.square(inputs[..., :2] - box_xy) * mask * box_loss_scale) / batch_size
    wh_loss = tf.reduce_sum(tf.square(inputs[..., 2:4] - box_wh) * mask * box_loss_scale) / batch_size
    obj_loss = tf.reduce_sum(-1.0 * inputs[..., 4:5] * tf.math.log(objectness) * mask) / batch_size
    no_obj_loss = tf.reduce_sum(-1.0 * (1.0 - inputs[..., 4:5]) * tf.math.log(1.0 - objectness) * no_object_mask) / batch_size
    class_loss = tf.reduce_sum(-1.0 * inputs[..., 5:] * tf.math.log(class_probs) * mask) / batch_size
    
    loss = xy_loss + wh_loss + obj_loss + no_obj_loss + class_loss
    
    # Return the predicted bounding boxes and the loss
    return pred_boxes, loss

##     Define the input shape of your model. This should be the size of the images you want to input to the model.
  ##  Instantiate the YOLOv3 model using the yolov3_head function I provided. Pass the input shape as an argument.
   #$ Compile the model using an appropriate loss function and optimizer.
    #  Train the model on a dataset of labeled images using the fit method of the model.
  ##  To use the model for inference, load the saved weights of the trained model into the YOLOv3 model and pass an input image through the model. The output of the model will be the predicted bounding boxes and class probabilities for each object in the image. #