# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import keras
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from ann_visualizer.visualize import ann_viz;


#loading the model
json_file = open('models/keras/fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/keras/fer.h5")
# loaded_model.save("services/emotion/fullmodel.h5")
# SVG(model_to_dot(loaded_model).create(prog='dot', format='svg'))
# plot_model(loaded_model, to_file='model.png', show_shapes=True)
ann_viz(loaded_model, title="Emotion Recognition", filename="demo/graph")