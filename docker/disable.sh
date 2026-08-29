#!/bin/bash

# shellcheck disable=SC2164
cd "$HOME"/Cuterbot/docker
source configure.sh

JETBOT_CAMERA=${1:-opencv_gst_camera}  # default to opencv

if [ "$JETBOT_CAMERA" = "zmq_camera" ]; then
  sudo docker stop jetbot_camera
  sudo docker rm jetbot_camera
fi

sudo docker stop jetbot_jupyter
sudo docker rm jetbot_jupyter

sudo docker stop jetbot_display
sudo docker rm jetbot_display
