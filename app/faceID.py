#import kivy dependencies
from json.tool import main
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger


#import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

#build app and layout

class CamApp(App):

    def build(self):
        #main layout components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text = "Verify", on_press = self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        #add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #load keras model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        #setup video capture
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)


        return layout

    def update(self, *args):
        
        #read frame from open cv
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250,:]

        #convert image to texture
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture=img_texture

    def preprocess(self,file_path):
    
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0

        # Return image
        return img
    
    #used to verify our image
    def verify(self, *args):
        #specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.8

        #capture input image
        SAVE_PATH = os.path.join('application_data','input_image','input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250,:]
        cv2.imwrite(SAVE_PATH,frame)
    

        #detection threshold: Metric above which a prediction is considered positive
        #verification threshold: number of positive matches with validation images

        #build results array
        results = []
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_image = self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_image = self.preprocess(os.path.join('application_data','verification_images',image))

            result = self.model.predict(list(np.expand_dims([input_image,validation_image],axis=1)))
            results.append(result)


        #how many above detection threshold
        detection = np.sum(np.array(results) > detection_threshold)

        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        #is above verification threshold?
        verified = verification > verification_threshold

        #update verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)

        return results, verified


if __name__ == '__main__':
    CamApp().run()