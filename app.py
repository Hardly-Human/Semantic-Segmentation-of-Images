import streamlit as st
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from PIL import Image
from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
import cv2
import numpy as np
ctx = mx.cpu(0)

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

@st.cache(allow_output_mutation=True)
def load_model(model_name):
	model = gluoncv.model_zoo.get_model(model_name, pretrained = True)
	return model

def plot_image(model,img):
	st.warning("Inferencing from Model..")
	output = model.predict(img)
	predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
	mask = get_color_pallete(predict, 'mhpv1')
	mask.save('output.png')
	mmask = mpimg.imread('output.png')
	image = mpimg.imread('saved_image.jpg')

	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.success("Pose Estimation Successful!! Plotting Image..")
    
	st.image(image,width = 500, height = 500)
	st.image(mmask,width = 500, height = 500)

def footer():
	st.markdown("""
	* * *
	Built with ❤️ by [Rehan uddin](https://hardly-human.github.io/)
	""")
	st.success("Rehan uddin (Hardly-Human)👋😉")
	st.markdown("### [Give Feedback](https://www.iamrehan.me/forms/feedback_form/feedback_form.html)\
	 `            `[Report an Issue](https://www.iamrehan.me/forms/report_issue/report_issue.html)")


def main():
	st.title("Semantic Segmentation App for Images")
	st.text("Built with gluoncv and Streamlit")
	st.markdown("""### [Semantic Segmentation](https://towardsdatascience.com/semantic-segmentation-with-deep-learning-a-guide-and-code-e52fc8958823)\
     `            `[PSPNet](https://towardsdatascience.com/review-pspnet-winner-in-ilsvrc-2016-semantic-segmentation-scene-parsing-e089e5df177d) \
	 `			  `[[Paper]](https://arxiv.org/abs/1612.01105)\
	 `			  `[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/Hardly-Human/Semantic-Segmentation-of-Images)\
	 `            `[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://lbesson.mit-license.org/)""")
	
	

	image_file = st.file_uploader("Upload Image", type = ['jpg','png','jpeg'])

	if image_file is None:
		st.warning("Upload Image and Run Model  (Use Image size <300 KB for faster inference)")

	if image_file is not None:
		image1 = Image.open(image_file)
		rgb_im = image1.convert('RGB') 
		image2 = rgb_im.save("saved_image.jpg")
		image_path = "saved_image.jpg"
		st.image(image1,width = 500, height = 500)

	if st.button("Run Model"):
		st.warning("Loading Model..🤞")
		model = load_model('psp_resnet101_ade')
		img = image.imread(image_path)
		img = test_transform(img, ctx)
		st.success("Loaded Model Succesfully!!🤩👍")

		plot_image(model,img)


if __name__ == "__main__":
	main()
	footer()
