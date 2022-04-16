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
        self.img1 = Image(size_hint=(1,.8))
        self.button = Button(text = "Verify", size_hint=(1,.1))
        self.verification = Label(text="Verification Uninitiated", size_hint=(1,.1))

        #add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        return layout


if __name__ == '__main__':
    CamApp().run()