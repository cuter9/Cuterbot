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
  echo -e "\n\e[48;5;172m Building bsae image! \e[0m"
  cd base && ./build.sh && cd .. >> "$LOG_FILE"
else
  echo -e "\n\e[48;5;172m The base image will not be built! \e[0m"
fi

# shellcheck disable=SC2129
echo -e "\n\e[48;5;172m Building jetbo image! \e[0m"
cd jetbot && ./build.sh && cd .. >> "$LOG_FILE"
# cd models && ./build.sh && cd .. >> $LOG_FILE

echo -e "\n\e[48;5;172m Building display image! \e[0m"
cd display && ./build.sh && cd .. >> "$LOG_FILE"

echo -e "\n\e[48;5;172m Building jupyter lab image! \e[0m"
cd jupyter && ./build.sh && cd .. >> "$LOG_FILE"

echo -e "\n\e[48;5;172m Building camera image! \e[0m"
cd camera && ./build.sh &&  cd .. >> "$LOG_FILE"

./disable.sh
docker image prune -f

# shellcheck disable=SC2164
cd "$HOME"