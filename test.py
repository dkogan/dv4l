#!/usr/bin/python3

import sys
import numpy as np
from fltk import *
from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget
import dv4l

window = Fl_Window(800, 600, "video4linux camera stream")
image_widget  = Fl_Gl_Image_Widget(0,0, 800,600)
window.resizable(image_widget)
window.end()
window.show()

c = dv4l.camera("/dev/video0")


def update(*args):
    timestamp,image = c.get_frame()
    image_widget.update_image(image_data = image)


Fl.add_fd(c.get_fd(),
          update)

Fl.run()
