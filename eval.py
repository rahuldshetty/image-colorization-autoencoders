import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import numpy as np

input_path = "samples/"
output_path = "outputs/"
res_path = "predicted/"

model = load_model('model.h5')
print("Model loaded...")

def display(img):
    plt.figure()
    plt.set_cmap('gray')
    plt.imshow(img)
    plt.show()

def normalize(image):
    # convert image from range 0-256 to 
    image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = image/255
    return image

def unnormalize(image):
    image = (image*255)
    return image.astype('uint8')
    
def rgb_image(l, ab):
    shape = (l.shape[0],l.shape[1],3)
    img = np.zeros(shape)
    img[:,:,0] = l[:,:,0]
    img[:,:,1:]= ab
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

def get_image(path):
    return cv2.imread(path)

def get_gray_image(path):
    return cv2.imread(path,0)

def predict(image_path):

    expected_img = get_image( output_path + image_path )
    expected_img = cv2.resize(expected_img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    expected_img = cv2.cvtColor(expected_img,cv2.COLOR_BGR2RGB)

    gray_img = get_gray_image( input_path + image_path)
    gray_img = cv2.resize(gray_img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

    norm_img = np.array([ normalize(gray_img).reshape((64,64,1)) ]) 
    res = model.predict(norm_img)[0]
  
    actual_ab = unnormalize(res)       
    actual_l = unnormalize(norm_img[0])
    actual_img = rgb_image(actual_l,actual_ab)

    outputs = [gray_img , expected_img , actual_img]

    fig=plt.figure(figsize=(64, 64))
    columns = 3
    rows = 1
    plt.set_cmap('gray')
    for i in range(1, columns*rows + 1):
        img = outputs[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    


if __name__ == "__main__":    
    while True:
        file = input("Enter file path:")
        if file == "":
            break
        else:predict(file)
