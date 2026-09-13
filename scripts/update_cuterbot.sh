#!/bin/bash

# shellcheck disable=SC2164
git stash save
cd /home/cuterbot/Cuterbot

sudo git pull origin master
sudo git reset --hard origin/master
# shellcheck disable=SC1065
sleep 5
# shellcheck disable=SC1072
sudo chmod +x "/home/cuterbot/Cuterbot/scripts/update_cuterbot.sh"
sleep 5
# shellcheck disable=SC2046
sudo chown $(whoami) "/home/cuterbot/Cuterbot/scripts/fix_files.sh"
sudo chmod +x  "$HOME"/Cuterbot/scripts/fix_files.sh
# shellcheck disable=SC1001
sudo "/home/cuterbot/Cuterbot/scripts/fix_files.sh"
# shellcheck disable=SC2164
cd "$HOME"

if [ ! -d ${HOME}/model_repo ]; then
  echo "Downloading models ------- "
  ${HOME}/Cuterbot/scripts/download_model_repo.sh
else
  echo -e "\n\e[48;5;172m Skip downloading models! Check the models is already in /home/cuterbot/Cuterbot/model_repo !\e[0m"
fi




