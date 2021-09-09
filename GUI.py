# importing required packages ..

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
from keras.models import load_model
import os

# Loading the Model ..

os.chdir('PATH\\Traffic_Signs_Recognition')  # you must change your current work directory to the one which has the
# Traffic_Signs_Classifier.h5 inside of it
model = load_model('Traffic_Signs_Classifier.h5')

classes = {1: 'Speed limit (20Km/h)',
           2: 'Speed limit (30Km/h)',
           3: 'Speed limit (50Km/h)',
           4: 'Speed limit (60Km/h)',
           5: 'Speed limit (70Km/h)',
           6: 'Speed limit (80Km/h)',
           7: 'End of speed limit (80Km/h)',
           8: 'Speed limit (100Km/h)',
           9: 'Speed limit (120Km/h)',
           10: 'No passing',
           11: 'No passing for vehicles over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'vehicles over 3.5 tons are prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road Work',
           27: 'Traffic signals',
           28: 'Pedestrian',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'keep right',
           40: 'keep left',
           41: 'Roundabout mandatory',
           42: 'End of no Passing',
           43: 'End no passing vehicle over 3.5 tons'}

# Building GUI which receives a photo from the user and uses the model to classify it ..

gui = tk.Tk()
gui.geometry('1000x600+50+50')
gui.title('Traffic Sign Classification')
gui.configure(background='white')
gui.resizable(0, 0)

# the next label used to show the output of classification ..

label = Label(gui, background='white',
              font=('arial', 24, 'bold'))  # #00B294 in RGB is composed of 0% red , 69.8% green , 58% red

# the next label used to show the loaded image ..

sign_image = Label(gui)


def classify(file_path):
    image = cv2.imread(file_path, 1)
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image,
                           axis=0)  # insert a new axis that will appear at the 'axis' position in the expanded array
    # shape.
    # model.predict(list_of_images) gives a list contains predicted classes for each image in the passed images list
    pred = model.predict_classes([image])[0]
    sign = classes[pred + 1]  # classes indexing in the model starts from 0 , whereas in the dict 'classes' from 1
    label.configure(foreground='white', bg='#00B294', text=sign)


def show_classify_btn(file_path):
    classify_btn = Button(gui, text='classify Image', command=lambda: classify(file_path), padx=10, pady=5)
    classify_btn.configure(background='#000000', foreground='white', font=('arial', 10))
    classify_btn.place(relx=0.442, rely=0.78)


def upload_img():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path).resize((250, 250))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_btn(file_path)

    except:
        print("error in  upload")
        pass


upload = Button(gui, text="Upload image", command=upload_img, padx=10, pady=5)
upload.configure(background='#000000', foreground='white', font=('arial', 10))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True, pady=10)
label.pack(side=BOTTOM, expand=True)

heading = Label(gui, text="Traffic Signs Classification using CNN-based deep learning Model", pady=20, width=100,
                font=('arial', 20, 'bold'))
heading.configure(background='#000000', foreground='white')
heading.pack()

footer = Label(gui)
footer.configure(background="#000000", foreground='white', pady=10, width=111, font=('arial', 12))
footer.place(relx=0.0, rely=.94)

gui.mainloop()
