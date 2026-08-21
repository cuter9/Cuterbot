#!/bin/bash

sudo chmod -R 777 ~/Cuterbot_Demo
sudo apt install dos2unix
find ~/Cuterbot_Demo -name *.sh | xargs dos2unix