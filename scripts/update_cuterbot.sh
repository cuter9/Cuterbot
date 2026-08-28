#!/bin/bash

# shellcheck disable=SC2164
cd /home/cuterbot/Cuterbot

sudo git pull origin v2.01
sudo git reset --hard origin/v2.01
# shellcheck disable=SC1065
sleep 5
# shellcheck disable=SC1072
sudo chmod +x "/home/cuterbot/Cuterbot/scripts/update_github.sh"
sleep 5
# shellcheck disable=SC2046
sudo chown $(whoami) "/home/cuterbot/Cuterbot/scripts/fix_files.sh"
sudo chmod +x  "$HOME"/Cuterbot/scripts/fix_files.sh
# shellcheck disable=SC1001
sudo "/home/cuterbot/Cuterbot/scripts/fix_files.sh"
# shellcheck disable=SC2164
cd "$HOME"

