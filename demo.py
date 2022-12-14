#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
from fltk import *
from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget
import dv4l



class Fl_Gl_Image_Widget_Derived(Fl_Gl_Image_Widget):
    def handle(self, event):
        if event == FL_MOVE:
            try:
                q = self.map_pixel_image_from_viewport( (Fl.event_x(),Fl.event_y()), )
                status_widget.value(f"{q[0]:.1f},{q[1]:.1f}")
            except:
                status_widget.value("")
            # fall through to let parent handlers run

        return super().handle(event)


window        = Fl_Window(800, 600, "video4linux camera stream")
group         = Fl_Group(0,0,800,580)
image_widget  = Fl_Gl_Image_Widget_Derived(0,  0, 400,580)
image_widget2 = Fl_Gl_Image_Widget_Derived(400,0, 400,580)
group.end()
status_widget = Fl_Output(0, 580, 800, 20)
window.resizable(group)
window.end()
window.show()

c = dv4l.camera("/dev/video0")


def update(fd):
    timestamp,image = c.get_frame()
    image_widget .update_image(image_data = image, flip_x = True)
    image_widget2.update_image(image_data = 255-image, flip_x = True, flip_y = True)


Fl.add_fd(c.get_fd(),
          update)

Fl.run()
