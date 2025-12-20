import cv2
import numpy as np

from jetbot import Camera, bgr8_to_jpeg
import traitlets
from traitlets import HasTraits, Unicode, Float

class DataCollection(HasTraits):
    widget_image = traitlets.Any()
    xy_image = traitlets.Any()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = Camera()
        self.width_display = self.camera.width_display
        self.height_display = self.camera.height_display
        self.widget_image = np.empty((self.height_display, self.width_display, 3), dtype=np.uint8).tobytes()
        self.xy_image = np.empty((self.height_display, self.width_display, 3), dtype=np.uint8).tobytes()

    def get_xy_image(self, x_slider_value, y_slider_value):
        image = cv2.resize(self.camera.value, (self.width_display, self.height_display),
                           interpolation=cv2.INTER_LINEAR)
        x = x_slider_value
        y = y_slider_value
        x = int(x * self.width_display / 2 + self.width_display / 2)
        y = int(y * self.height_display / 2 + self.height_display / 2)
        image = cv2.circle(image, (x, y), 4, (0, 255, 0), 2)
        image = cv2.circle(image, (int(self.width_display / 2), int(self.height_display)), 4, (0, 0, 255), 2)
        image = cv2.line(image, (x, y), (int(self.width_display / 2), int(self.height_display)), (255, 0, 0), 2)
        image = cv2.line(image, (0, int(self.height_display / 2)), (int(self.width_display), int(self.height_display / 2)),
                         (0, 255, 255), 2)
        self.xy_image = bgr8_to_jpeg(image)

    def get_widget_image(self):
        self.widget_image = bgr8_to_jpeg(cv2.resize(self.camera.value, (self.width_display, self.height_display),
                                                    interpolation=cv2.INTER_LINEAR))

    def save_image(self, image_path):
        with open(image_path, 'wb') as f:
            f.write(bgr8_to_jpeg(self.camera.value))

