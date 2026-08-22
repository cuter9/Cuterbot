#!/bin/bash

sudo git pull origin v2.01
sudo git reset --hard origin/v2.01

sudo chmod +x "/home/cuterbot/Cuterbot/shell_scripts/update_github.sh"
# shellcheck disable=SC2046
sudo chown $(whoami) "/home/cuterbot/Cuterbot/shell_scripts/fix_files.sh"
sudo chmod +x  "$HOME"/Cuterbot/shell_scripts/fix_files.sh
# shellcheck disable=SC1001
sudo "/home/cuterbot/Cuterbot/shell_scripts/fix_files.sh/fix_files.sh"

