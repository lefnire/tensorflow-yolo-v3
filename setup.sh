wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights

# Unclear if NCHW should be used for GPU ckpt/pb files (incompat w/ CPU). Maybe better performance? Seems NHCW is
# compatible w/ both, so using same file for both CPU/GPU.
#python ./convert_weights.py --tiny --weights_file yolov3-tiny.weights --data_format NCHW --ckpt_file ./saved_model/yolov3_tiny_GPU.ckpt

python ./convert_weights.py --tiny --weights_file yolov3-tiny.weights --data_format NHWC --ckpt_file ./saved_model/yolov3_tiny.ckpt
python ./convert_weights_pb.py --tiny --weights_file yolov3-tiny.weights --data_format NHWC --output_graph yolov3_tiny.pb

python ./convert_weights.py --weights_file yolov3.weights --data_format NHWC --ckpt_file ./saved_model/yolov3.ckpt
python ./convert_weights_pb.py --weights_file yolov3.weights --data_format NHWC --output_graph yolov3.pb