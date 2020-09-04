#!/usr/bin/python3

#
# ****************************************************************************
# Detect and annotate objects on a LIVE camera feed
# using the Google Coral USB Stick.
#
# Works with both Raspberry Pi Camera and USB Camera
# (see ARGS for how to switch bewtween these).
#
# Version 2: Fixed percentage display and assume that all files are under the
# same directory.
#
# ****************************************************************************
#

import os
import cv2
import sys
import numpy
import ntpath
import argparse
import time

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import edgetpu.detection.engine
from edgetpu.utils import image_processing

from imutils.video import FPS

# Variable to store command line arguments
ARGS = None

# Read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

# Annotate and display video
def annotate_and_display ( image, inferenceResults, elapsedMs, labels, font ):

    # Iterate through result list. Note that results are already sorted by
    # confidence score (highest to lowest) and records with a lower score
    # than the threshold are already removed.
    result_size = len(inferenceResults)
    for idx, obj in enumerate(inferenceResults):
        # TODO: don't annotate and draw boxes if not displaying the image
        
        # Prepare image for drawing
        draw = PIL.ImageDraw.Draw( image )

        # Prepare boundary box
        box = obj.bounding_box.flatten().tolist()

        # Draw rectangle to desired thickness
        for x in range( 0, 4 ):
            draw.rectangle(box, outline=(255, 255, 0) )

        # Annotate image with label and confidence score
        display_str = labels[obj.label_id] + ": " + str(round(obj.score*100, 2)) + "%"
        draw.text( (box[0], box[1]), display_str, font=font )

        # Log the current result to terminal
        if ARGS.print:
            print("Object (" + str(idx+1) + " of " + str(result_size) + "): "
                + labels[obj.label_id] + " (" + str(obj.label_id) + ")"
                + ", Confidence:" + str(obj.score)
                + ", Elapsed:" + str(elapsedMs*1000.0) + "ms"
                + ", Box:" + str(box))

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ or ARGS.display:
        displayImage = numpy.asarray( image )
        cv2.imshow( 'NCS Improved live inference', displayImage )

# Main flow
def main():

    # Store labels for matching with inference results
    labels = ReadLabelFile(ARGS.labels) if ARGS.labels else None

    # Specify font for labels
    font = PIL.ImageFont.load_default()

    # Use Google Corals own DetectionEngine for handling
    # communication with the Coral
    inferenceEngine = edgetpu.detection.engine.DetectionEngine(ARGS.model)

    camera = cv2.VideoCapture( 0 )
    camera.set( cv2.CAP_PROP_FRAME_WIDTH, 620 )
    camera.set( cv2.CAP_PROP_FRAME_HEIGHT, 480 )

    time.sleep(1)

    # Use imutils to count Frames Per Second (FPS)
    fps = FPS().start()

    # Capture live stream & send frames for preprocessing, inference and annotation
    while True:
        try:

            # Read frame from video and prepare for inference
            _, screenshot = camera.read()

            # Prepare screenshot for annotation by reading it into a PIL IMAGE object
            image = PIL.Image.fromarray( screenshot )

            # Perform inference and note time taken
            startMs = time.time()
            inferenceResults = inferenceEngine.detect_with_image(image, threshold=ARGS.confidence, keep_aspect_ratio=True, relative_coord=False, top_k=ARGS.maxobjects)
            elapsedMs = time.time() - startMs

            # Annotate and display
            annotate_and_display( image, inferenceResults, elapsedMs, labels, font )

            # Display the frame for 5ms, and close the window so that the next
            # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
            if( cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) ):
                fps.stop()
                break

            fps.update()

        # Allows graceful exit using ctrl-c (handy for headless mode).
        except KeyboardInterrupt:
            fps.stop()
            break

    print("Elapsed time: " + str(fps.elapsed()))
    print("Approx FPS: :" + str(fps.fps()))

    cv2.destroyAllWindows()
    # vs.stop()
    time.sleep(2)

def str2bool(v):
  """Utility function for parsing arguments to Booleans (https://stackoverflow.com/a/43357954/1632704)"""
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


# Define 'main' function as the entry point for this script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Detect objects on a LIVE camera feed using \
                         Google Coral USB." )

    parser.add_argument( '--model', type=str,
                         default='mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite',
                         help="Path to the neural network graph file." )

    parser.add_argument( '--labels', type=str,
                         default='coco_labels.txt',
                         help="Path to labels file." )

    parser.add_argument( '--maxobjects', type=int,
                         default=3,
                         help="Maximum objects to infer in each frame of video." )

    parser.add_argument( '--confidence', type=float,
                         default=0.60,
                         help="Minimum confidence threshold to tag objects." )

    parser.add_argument( '-d', "--display", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Display annotated image (default False)")

    parser.add_argument( '-p', "--print", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Print results continuously (default True)")

    ARGS = parser.parse_args()

    # Load the labels file
    labels =[ line.rstrip('\n') for line in
              open( ARGS.labels ) if line != 'classes\n']


    main()

# ==== End of file ===========================================================
