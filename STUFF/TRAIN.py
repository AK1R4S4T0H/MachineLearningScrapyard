# 
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Set up pipeline config
config_path = 'path/to/your/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(config_path)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(config_path, 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Set up training parameters
num_steps = 100000
checkpoint_every_n = 10000

# Set up training data
train_record_path = 'path/to/your/train.record'
label_map_path = 'path/to/your/label_map.pbtxt'

# Set up validation data
val_record_path = 'path/to/your/val.record'

# Set up output directory
output_directory = 'path/to/your/output_directory'

# Set up model checkpoint directory
model_checkpoint_directory = 'path/to/your/model_checkpoint_directory'

# Start training
pipeline_config.train_config.fine_tune_checkpoint = 'path/to/pretrained/model/checkpoint'
pipeline_config.train_config.num_steps = num_steps
pipeline_config.train_input_reader.label_map_path = label_map_path
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [train_record_path]
pipeline_config.eval_input_reader[0].label_map_path = label_map_path
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [val_record_path]
pipeline_config.model.ssd.num_classes = 80  # change this to match the number of classes in your dataset
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = 0.2
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = num_steps
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = 0.006666666666666667
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = 500
pipeline_config.train_config.gradient_clipping_by_norm = 10.0
pipeline_config.train_config.checkpoint_every_n = checkpoint_every_n
pipeline_config.train_config.max_number_of_boxes = 100
pipeline_config.train_config.unpad_groundtruth_tensors = False
pipeline_config.train_config.use_bfloat16 = False
pipeline_config.train_config.num_layers = 6

# Write updated pipeline config to file
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(config_path, 'wb') as f:
    f.write(config_text)

# Run the training job
!python object_detection/model_main_tf2.py --model_dir={model_checkpoint_directory} --pipeline_config_path={config_path} --num_train_steps={num_steps} --num_eval_steps=10 --alsologtostderr

# Export the trained model
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [train_record_path]
pipeline_config.eval_config.max_evals = 1
pipeline_config.eval_config.use_moving_averages = False
pipeline_config.eval_config.metrics_set.extend(['coco_detection_metrics'])
pipeline_config.eval_config.export_path = os.path.join(output_directory, 'exported_model')
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(config_path, 'wb') as f:
    f.write(config_text)
!python object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path {config_path} --trained_checkpoint_dir {model_checkpoint_directory} --output_directory {pipeline_config.eval_config.export_path}

