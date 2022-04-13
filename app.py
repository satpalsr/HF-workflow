import gradio as gr
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import numpy as np

model = from_pretrained_keras("keras-io/mobile-vit-xxs")

classes=['dandelion','daisy','tulip','sunflower','rose']
image_size = 256

def classify_images(image):
  image = tf.convert_to_tensor(image)
  image = tf.image.resize(image, (image_size, image_size))
  image = tf.expand_dims(image,axis=0)
  prediction = model.predict(image)
  prediction = tf.squeeze(tf.round(prediction))
  text_output = str(f'{classes[(np.argmax(prediction))]}!')
  return text_output
  
i = gr.inputs.Image()
o = gr.outputs.Textbox()

examples = [["tulip.png"]]
title = "Flowers Classification MobileViT"
description = "Upload an image or select from examples to classify flowers"

article = "<div style='text-align: center;'><a href='https://twitter.com/SatpalPatawat' target='_blank'>Space by Satpal Singh Rathore</a><br><a href='https://keras.io/examples/vision/mobilevit/' target='_blank'>Keras example by Sayak Paul</a></div>"
gr.Interface(classify_images, i, o, allow_flagging=False, analytics_enabled=False,
  title=title, examples=examples, description=description, article=article).launch(enable_queue=True)
