# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects in a given image."""

import argparse
import time
import numpy as np

from PIL import Image
from PIL import ImageDraw

import detect
import tflite_runtime.interpreter as tflite
import platform

import cv2

# In order to fix Mac OS issue with matplotlib (https://stackoverflow.com/a/53014308/1632704)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
from imutils.video import FPS

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter():
  try:
    model_file, *device = args.model.split('@')
    delegates = [tflite.load_delegate( EDGETPU_SHARED_LIB, {'device': device[0]} if device else {} )] if args.edgetpu else []
    print( "Using Edge TPU" if args.edgetpu else "Not using Edge TPU - running on local CPU." )
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=delegates)
  except ValueError as ex:
    description = str( ex ).replace( "\n", "" )
    print( f"⚠️  Unable to initialize interpreter ({description}). Is the Edge TPU connected?\n\n")


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')

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

def main():
  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter()
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Check the type of the input tensor and height, width
  # is_floating_model = input_details[0]['dtype'] == np.float32
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  # Open camera
  camera = cv2.VideoCapture( 0 )
  camera.set( cv2.CAP_PROP_FRAME_WIDTH, 620 )
  camera.set( cv2.CAP_PROP_FRAME_HEIGHT, 480 )
  time.sleep(1) # Allow time to open camera

  fps = FPS().start()

  while True:
    try:
      # Read frame from video and prepare for inference
      _, frame = camera.read()
      #frame = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )

      # Prepare screenshot for annotation by reading it into a PIL IMAGE object
      image = Image.fromarray( frame )

      scale = detect.set_input(interpreter, image.size,
                              lambda size: image.resize(size, Image.ANTIALIAS))
      interpreter.invoke()
      objs = detect.get_output( interpreter, args.threshold, scale )

      if not objs:
        print('No objects detected')

      # for obj in objs:
      #   print( labels.get(obj.id, obj.id) )
      #   print('  id:    ', obj.id)
      #   print('  score: ', obj.score)
      #   print('  bbox:  ', obj.bbox)

      draw_objects( ImageDraw.Draw(image), objs, labels )

      frame = np.array( image )

      fps.update()

      cv2.imshow( "Capture", frame )
      if( cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) ):
          fps.stop()
          break
    
    except KeyboardInterrupt:
        fps.stop()
        break

  print("Approx FPS: :" + str(fps.fps()))



if __name__ == '__main__':
  # Parse args in global scope
  parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter )
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-v', '--video', default=0,
                      help='Video index to open (default 0).')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects.')
  parser.add_argument( '-e', "--edgetpu", type=str2bool, nargs='?',
                        const=True, required=True,
                        help="Use Edge PTU" )
  args = parser.parse_args()

  main()
