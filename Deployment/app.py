from fastai.vision.all import load_learner
import gradio as gr

#--Below 3 line of code should be uncommented when deploying gradio app / run app.py in terminal
import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

insect_names=[
    'Africanized Bees',
    'Aphids',
    'Armyworms',
    'Cabbage Loopers',
    'Corn Borers',
    'Corn Earwormes',
    'Corn Rootworms',
    'Fruit Flies',
    'Potato Beetles',
    'Stink Bugs',
    'Thrips',
    'Tomato Hornworms']

model=load_learner("models/insect-classifier-v3.pkl")

def recognize_image(image):
  pred,idx,probs=model.predict(image)
  print(pred)
  return dict(zip(insect_names, map(float,probs)))

image = gr.Image()
label = gr.Label()
examples = [
   "Test Data/Test_Image_1.jpg",
   "Test Data/Test_Image_2.jpg",
   "Test Data/Test_Image_3.jpg",
   "Test Data/Test_Image_4.jpg",
   "Test Data/Test_Image_5.jpg",
   "Test Data/Test_Image_6.jpg"]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label,examples=examples)
iface.launch(inline=False,share=True)