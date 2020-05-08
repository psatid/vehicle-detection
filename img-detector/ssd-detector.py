# adapted from: https://jeanvitor.com/tensorflow-object-detecion-opencv/

import cv2
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--ssd", default = '../models/ssd',
	help="base path to ssd directory")

args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["ssd"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

print("[INFO] loading SSD from disk...")
# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('../models/ssd/frozen_inference_graph.pb', '../models/ssd/graph.pbtxt')
 
# Input image
img = cv2.imread(args['image'])
rows, cols, channels = img.shape
 
# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

start = time.time()
networkOutput = tensorflowNet.forward()
end = time.time()

print("[INFO] SSD took {:.6f} seconds".format(end - start)) 
 
# Loop on the outputs
for detection in networkOutput[0,0]:
    
    score = float(detection[2])
    id = int(detection[1])
    classID = id -1

    # print(LABELS[classID])

    if score > 0.5:
    	
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows

        # print(LABELS[classID])

        # print((int(left), int(top)), (int(right), int(bottom)))
 
        if(LABELS[classID] == 'car' or LABELS[classID] == 'bus' or LABELS[classID] == 'motorcycle' or LABELS[classID] == 'truck'):
            #draw a red rectangle around detected objects
            text = "{}: {:.4f}".format(LABELS[classID], score)
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=2)

            cv2.putText(img, text, (int(left), int(top) - 5),
		        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
 
# Show the image with a rectagle surrounding the detected objects 
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()