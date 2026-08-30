#!/bin/bash

LOG_FILE="$HOME"/Cuterbot/docker/docker_build_.log
# shellcheck disable=SC2188
> "$LOG_FILE"

# shellcheck disable=SC2164
cd "$HOME"/Cuterbot/docker
source ./configure.sh >> $LOG_FILE
# docker images --format "{{.Repository}}:{{.Tag}}" | grep -q ":base.*"
base_img=$(docker images --format "{{.Tag}}" | grep "^base.*") >> "$LOG_FILE"
if [[ -z "$base_img" || "$1" == "re_base" ]]; then
  echo -e "\e[48;5;172m Building bsae image! \e[0m"
  cd base && ./build.sh && cd .. >> "$LOG_FILE"
fi
echo -e "\e[48;5;172m bsae image will not be build! \e[0m"
# shellcheck disable=SC2129
cd jetbot && ./build.sh && cd .. >> "$LOG_FILE"
# cd models && ./build.sh && cd .. >> $LOG_FILE
cd display && ./build.sh && cd .. >> "$LOG_FILE"
cd jupyter && ./build.sh && cd .. >> "$LOG_FILE"
cd camera && ./build.sh &&  cd .. >> "$LOG_FILE"

./disable.sh
docker image prune -f

# shellcheck disable=SC2164
cd "$HOME"