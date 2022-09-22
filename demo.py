#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
from fltk import *
from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget
import dv4l

window        = Fl_Window(800, 600, "video4linux camera stream")
image_widget  = Fl_Gl_Image_Widget(0,0, 400,600)
image_widget2 = Fl_Gl_Image_Widget(400,0, 400,600)
window.resizable(window)
window.end()
window.show()

c = dv4l.camera("/dev/video0")


def update(fd):
    timestamp,image = c.get_frame()
    image_widget .update_image(image_data = image)
    image_widget2.update_image(image_data = 255-image)
Fl.add_fd(c.get_fd(),
          update)

Fl.run()
