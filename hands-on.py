print('''HEYA!!!!🖖🏾


    Please give me a moment to sort things out. 🕰️
    
    
    
    
    
    
    ''')

from keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def predict_image(img_path):
    img = cv2.imread(img_path) ## read the image
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ## convert the image color to gray
    r_img = cv2.resize(g,(28,28),interpolation = cv2.INTER_AREA) ## resize the image
    n_img = tf.keras.utils.normalize(r_img,axis =1 ) ## normalize the new resized image matrix
    n_img = np.array(n_img).reshape(-1,28,28,1) ##reshape the array dimensions
    pr = model.predict(n_img) ## predict the image
    return np.argmax(pr) ## return the prediction


image = input('\n \n Enter the path of your Image👉🏾:')

print('''Loading model...


''')
model = load_model('./digit_predictor.h5')
print('''   


Model Loaded :-)


''')
print('Trying to predict your Image..')
p = predict_image(image)
print(f'It\'s most likely {p} in my opinion')
