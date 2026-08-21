#!/bin/bash

sudo apt install dos2unix -y
sudo chmod -R 777 ~/Cuterbot
sudo find $HOME/Cuterbot -type f -name "*.sh" | xargs dos2unix
sudo find $HOME/Cuterbot -type f -name "*.sh" -exec chmod +x {}