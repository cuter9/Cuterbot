#!/bin/bash

# Disable GUI to free up more RAM
sudo systemctl set-default multi-user

# Disable ZRAM
sudo systemctl disable nvzramconfig.service

# Default to Max-N power mode
sudo nvpmodel -m 0


# if you want to use another port other than the default port for ssh connection. you can un comment the nest scrips, and set the Port no, e.g. 2222
# sudo "$HOME"/Cuterbot.scripts/open_ssh.sh

