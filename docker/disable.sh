
source configure.sh

JUPYTER_WORKSPACE=${1:-$HOME}  # default to $HOME
JETBOT_CAMERA=${2:-opencv_gst_camera}  # default to opencv

if [ "$JETBOT_CAMERA" = "zmq_camera" ]; then
  sudo docker stop jetbot_camera
  sudo docker rm jetbot_camera
fi

sudo docker stop jetbot_jupyter 
sudo docker rm jetbot_jupyter

sudo docker stop jetbot_display 
sudo docker rm jetbot_display
