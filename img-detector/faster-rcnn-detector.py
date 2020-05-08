# adapted from: https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/

import numpy as np
import argparse
import random
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--faster-rcnn", default = '../models/faster-rcnn',
	help="base path to faster-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["faster_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["faster_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["faster_rcnn"],
	"graph.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Faster R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
boxes = net.forward("detection_out_final")
end = time.time()

# show timing information and volume information on Mask R-CNN
print("[INFO] Faster R-CNN took {:.6f} seconds".format(end - start))

# loop over the number of detected objects
for i in range(0, boxes.shape[2]):
	# extract the class ID of the detection along with the confidence
	# (i.e., probability) associated with the prediction
	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]

	# filter out weak predictions by ensuring the detected probability
	# is greater than the minimum probability
	if confidence > args["confidence"]:
		# clone our original image so we can draw on it
		# clone = image.copy()

		# scale the bounding box coordinates back relative to the
		# size of the image and then compute the width and the height
		# of the bounding box
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
		boxW = endX - startX
		boxH = endY - startY

		if(LABELS[classID] == 'car' or LABELS[classID] == 'bus' or LABELS[classID] == 'motorcycle' or LABELS[classID] == 'truck'):
			cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)

			# draw the predicted label and associated probability of the
			# instance segmentation on the image
			text = "{}: {:.4f}%".format("Vehicle: ", confidence*100)
			cv2.putText(image, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
	   
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)