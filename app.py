#importing necessary modules
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry
import re

model_type="vit_h"

# Load the vit_h from a checkpoint file.
sam_checkpoint="sam_vit_h_4b8939.pth"
sam=sam_model_registry[model_type](checkpoint=sam_checkpoint)

from flask import Flask, request, Response
from PIL import Image
import requests, io

app = Flask(__name__)
def image_process(image_name,pattern):
  parts = image_name.split(' - ')
  ranges = []
  if len(parts) == 2:
    input_label=np.ones(int(parts[0]), dtype=int)
    range_str = parts[1]
    range_str = re.sub(r"\.[a-zA-Z0-9]+$", "", range_str)
    for range_part in range_str.split('&'):
      start, end = map(int, range_part.split('-'))
      ranges.append([start, end])
  input_points=np.array(ranges)
  try:
    image=cv2.imread(image_name)
    if image is None:
      print('Image not found')
  except Exception as e:
      print(e)
  predictor = SamPredictor(sam)
  predictor.set_image(image)
  masks, scores, logits = predictor.predict(
      point_coords=input_points,
      point_labels=input_label,
      multimask_output=True,
  )
  best_mask = masks[np.argmax(scores)]
  new_image=cv2.imread(pattern)
  new_image=cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)

  img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  x, y, w, h = cv2.boundingRect(best_mask.astype(np.uint8))
  resized_new_image=cv2.resize(new_image,(w,h))
  img[y:y+h,x:x+w][best_mask[y:y+h,x:x+w]]=resized_new_image[best_mask[y:y+h,x:x+w]]
  return img

@app.route('/process_images', methods=['POST'])
def process_images():
    product_image_url = request.json.get('product_image_url')
    room_image_url = request.json.get('room_image_url')

    if not product_image_url or not room_image_url:
        return 'Please provide product image URL and room image URL.', 400

    try:
        product_response = requests.get(product_image_url, stream=True)
        room_response = requests.get(room_image_url, stream=True)

        product_response.raise_for_status()
        room_response.raise_for_status()

        product_image = Image.open(io.BytesIO(product_response.content))
        room_image = Image.open(io.BytesIO(room_response.content))

  
        processed_image = image_process(room_image, product_image)

        processed_image_buffer = io.BytesIO()
        processed_image.save(processed_image_buffer, format='JPEG')  
        processed_image_data = processed_image_buffer.getvalue()

        return Response(processed_image_data, mimetype='image/jpeg', direct_passthrough=True)
    except requests.exceptions.RequestException as e:
        print(f'Error fetching images: {e}')  
        return 'Error fetching images.', 500
    except Exception as e:
        print(f'Error processing images: {e}') 
        return 'Error processing images.', 500
        
if __name__ == '__main__':
  app.run(debug=True, port=5000)