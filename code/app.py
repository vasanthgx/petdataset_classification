
from fastai.vision.all import *
import gradio as gr
import timm

learn = load_learner('pet_fastai_model.pkl')
categories = learn.dls.vocab
def classify_image(img):
    if img is None:
        return None
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

    
image = gr.Image()
label = gr.Label(num_top_classes=5)
examples = ['cat_russian_blue.jpg','cat_bombay.jpg','cat_abyssinian.jpg','dog_saint_bernard.jpg','dog_shih_tzu.jpg','dog_alabai(central_asian_shepherd_dog).jpg','basset.jpg']
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label,live=False, examples=examples,theme='freddyaboulton/dracula_revamped')
intf.launch()