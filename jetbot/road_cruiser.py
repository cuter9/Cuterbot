import time

import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import traitlets

from jetbot import Camera
from jetbot import Robot

matplotlib.use('nbAgg')

class RoadCruiser(traitlets.HasTraits):
    speed_gain = traitlets.Float(default_value=0.15).tag(config=True)
    steering_gain = traitlets.Float(default_value=0.08).tag(config=True)
    steering_dgain = traitlets.Float(default_value=1.5).tag(config=True)
    steering_bias = traitlets.Float(default_value=0.0).tag(config=True)
    steering = traitlets.Float(default_value=0.0).tag(config=True)
    x_slider = traitlets.Float(default_value=0).tag(config=True)
    y_slider = traitlets.Float(default_value=0).tag(config=True)

    def __int__(self, cruiser_model, type_model):
        # self.cruiser_model = cruiser_model
        self.type_model = type_model
        if type_model == "mobilenet":
            self.cruiser_model = cruiser_model
            self.cruiser_model.classifier[3] = torch.nn.Linear(self.cruiser_model.classifier[3].in_features, 2)
            self.cruiser_model.load_state_dict(torch.load('best_steering_model_xy_mobilenet_v3_large.pth'))
            self.speed_gain = 0.2
            self.steering_gain = 0.08
            self.steering_dgain = 0.82
            self.steering_bias = -0.01

        elif type_model == "resnet":
            # model = torchvision.models.resnet18(pretrained=False)
            self.cruiser_model = cruiser_model
            # model = torchvision.models.resnet50(pretrained=False)
            self.cruiser_model.fc = torch.nn.Linear(self.cruiser_model.fc.in_features, 2)
            # model.load_state_dict(torch.load('best_steering_model_xy_resnet18.pth'))
            self.cruiser_model.load_state_dict(torch.load('best_steering_model_xy_resnet34.pth'))
            # model.load_state_dict(torch.load('best_steering_model_xy_resnet50.pth'))
            self.speed_gain = 0.2
            self.steering_gain = 0.08
            self.steering_dgain = 0.82
            self.steering_bias = -0.01

        self.camera = Camera()
        self.robot = Robot()
        self.angle = 0.0
        self.angle_last = 0.0
        self.execution_time = []
        self.x_slider = 0
        self.y_slider = 0

        # model = torchvision.models.mobilenet_v3_large(pretrained=False)
        # model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)

        # model = torchvision.models.resnet18(pretrained=False)
        # model = torchvision.models.resnet34(pretrained=False)
        # model = torchvision.models.resnet50(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, 2)
        # model.load_state_dict(torch.load('best_steering_model_xy_mobilenet_v3_large.pth'))
        # model.load_state_dict(torch.load('best_steering_model_xy_resnet18.pth'))
        # model.load_state_dict(torch.load('best_steering_model_xy_resnet34.pth'))
        # model.load_state_dict(torch.load('best_steering_model_xy_resnet50.pth'))

        self.device = torch.device('cuda')
        # model = model.to(device)
        # model = model.eval().half()
        self.cruiser_model = self.cruiser_model.float()
        self.cruiser_model = self.cruiser_model.to(self.device, dtype=torch.float)
        self.cruiser_model = self.cruiser_model.eval()

    # ---- Creating the Pre-Processing Function
    # 1. Convert from HWC layout to CHW layout
    # 2. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
    # 3. Transfer the data from CPU memory to GPU memory
    # 4. Add a batch dimension

    def preprocess(self, image):
        # mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        # std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        image = PIL.Image.fromarray(image)
        # image = transforms.functional.to_tensor(image).to(device).half()
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    # Now, let’s start and display our camera. You should be pretty familiar with this by now.

    # camera = Camera()

    # image_widget = ipywidgets.Image()
    # fps_widget = ipywidgets.FloatText(description='Capture rate')

    # traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
    # traitlets.dlink((camera, 'cap_time'), (fps_widget, 'value'))

    # display(image_widget)
    # display(fps_widget)

    # We’ll also create our robot instance which we’ll need to drive the motors.

    # Now, we will define sliders to control JetBot
    # > Note: We have initialized the slider values for best known configurations, however these might not work for your dataset, therefore please increase or decrease the sliders according to your setup and environment

    # 1. Speed Control (speed_gain_slider): To start your JetBot increase ``speed_gain_slider``
    # 2. Steering Gain Control (steering_gain_slider): If you see JetBot is wobbling, you need to reduce ``steering_gain_slider`` till it is smooth
    # 3. Steering Bias control (steering_bias_slider): If you see JetBot is biased towards extreme right or extreme left side of the track, you should control this slider till JetBot start following line or track in the center.  This accounts for motor biases as well as camera offsets

    # > Note: You should play around above-mentioned sliders with lower speed to get smooth JetBot road following behavior.

    # speed_gain_slider = ipywidgets.FloatSlider(min=0, max=1, step=0.001, value=0.20, description='speed gain',
    #                                           readout_format='.3f')
    # steering_gain_slider = ipywidgets.FloatSlider(min=0, max=0.5, step=0.001, value=0.08, description='steering gain',
    #                                              readout_format='.3f')
    # steering_dgain_slider = ipywidgets.FloatSlider(min=0, max=2.0, step=0.001, value=1.5, description='steering kd',
    #                                               readout_format='.3f')
    # steering_bias_slider = ipywidgets.FloatSlider(min=-0.1, max=0.1, step=0.001, value=-0.015, description='steering bias',
    #                                               readout_format='.3f')

    ### resnet18
    # speed_gain_slider = ipywidgets.FloatSlider(min=0, max=1, step=0.001, value=0.2, description='speed gain', readout_format='.3f')
    # steering_gain_slider = ipywidgets.FloatSlider(min=0, max=0.5, step=0.001, value=0.08, description='steering gain', readout_format='.3f')
    # steering_dgain_slider = ipywidgets.FloatSlider(min=0, max=1, step=0.001, value=0.82, description='steering kd', readout_format='.3f')
    # steering_bias_slider = ipywidgets.FloatSlider(min=-0.1, max=0.1, step=0.001, value=-0.01, description='steering bias', readout_format='.3f')

    # VBox_image = ipywidgets.VBox([image_widget, fps_widget], layout=ipywidgets.Layout(align_self='center'))
    # VBox_control = ipywidgets.VBox([speed_gain_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider],
    #                               layout=ipywidgets.Layout(align_self='center'))
    # display(speed_gain_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider)
    # display(ipywidgets.HBox([VBox_image, VBox_control]))

    # Next, let’s display some sliders that will let us see what JetBot is thinking.
    # The x and y sliders will display the predicted x, y values.
    # The steering slider will display our estimated steering value. Please remember,
    # this value isn’t the actual angle of the target, but simply a value that is nearly proportional.
    # When the actual angle is 0, this will be zero, and it will increase / decrease with the actual angle.

    # x_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='x')
    # y_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='y')
    # steering_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='steering')
    # speed_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='speed')

    # display(ipywidgets.HBox([y_slider, speed_slider]))
    # display(x_slider, steering_slider)

    # Next, we'll create a function that will get called whenever the camera's value changes. This function will do the following steps
    #
    # 1. Pre-process the camera image
    # 2. Execute the neural network
    # 3. Compute the approximate steering value
    # 4. Control the motors using proportional / derivative control (PD)

    # angle = 0.0
    # angle_last = 0.0
    # et = []

    def execute(self, change):
        start_time = time.process_time()
        # global angle, angle_last
        image = change['new']
        xy = self.cruiser_model(self.preprocess(image)).detach().float().cpu().numpy().flatten()
        x = xy[0]
        # y = (0.5 - xy[1]) / 2.0
        y = (1 + xy[1])

        self.x_slider = x
        self.y_slider = y

        # self.speed_slider.value = speed_gain_slider.value

        # angle = np.sqrt(xy)*np.arctan2(x, y)
        angle_1 = np.arctan2(x, y)
        self.angle = 0.5 * np.pi * np.tanh(0.5 * angle_1)
        pid = self.angle * self.steering_gain + (self.angle - self.angle_last) * self.steering_dgain
        self.angle_last = self.angle

        self.steering = pid + self.steering_bias

        self.robot.left_motor.value = max(min(self.speed_gain + self.steering, 1.0), 0.0)
        self.robot.right_motor.value = max(min(self.speed_gain - self.steering, 1.0), 0.0)
        end_time = time.process_time()
        self.execution_time.append(end_time - start_time + self.camera.cap_time)

    def __ceil__(self):
        # self.execute({'new': self.camera.value})
        self.camera.observe(self.execute, names='value')

    # We accomplish that with the observe function.

    # import time

    # from IPython.display import clear_output

    # out = ipywidgets.Output()
    # button_stop = ipywidgets.Button(description='Stop', tooltip='Click to stop running', icon='fa-circle-stop')
    # display(button_stop, out)

    def stop_cruising(self, b):
        self.camera.unobserve(self.execute, names='value')
        execute_time = np.array(self.execution_time[1:])
        mean_execute_time = np.mean(execute_time)
        max_execute_time = np.amax(execute_time)
        min_execute_time = np.amin(execute_time)

        # with out:
        print(
            "Mean execution time of model : %f \nMax execution time of model : %f \nMin execution time of model : %f " \
            % (mean_execute_time, max_execute_time, min_execute_time))
        plt.hist(execute_time, bins=(0.005 * np.array(list(range(101)))).tolist())
        plt.show()

        time.sleep(1.0)
        self.robot.stop()
        self.camera.stop()
        # clear_output(wait=True)
        # %reset -f

    # button_stop.on_click(stop_cruising)
