#!/bin/bash

LOG_FILE="$HOME"/Cuterbot/docker/docker_build_.log
# shellcheck disable=SC2188
> "$LOG_FILE"

# shellcheck disable=SC2164
cd "$HOME"/Cuterbot/docker
source ./configure.sh >> $LOG_FILE 2>&1
# docker images --format "{{.Repository}}:{{.Tag}}" | grep -q ":base.*"
base_img=$(docker images --format "{{.Tag}}" | grep "^base.*") >> $LOG_FILE 2>&1
if [[ -z "$base_img" ]]; then
  (cd base && ./build.sh) >> $LOG_FILE 2>&1
fi
# shellcheck disable=SC2129
(cd jetbot && ./build.sh) >> $LOG_FILE 2>&1
# cd models && ./build.sh && cd .. >> $LOG_FILE
(cd display && ./build.sh) >> $LOG_FILE 2>&1
(cd jupyter && ./build.sh)>> $LOG_FILE 2>&1
(cd camera && ./build.sh)>> $LOG_FILE 2>&1

./disable.sh
docker image prune -f

# shellcheck disable=SC2164
cd "$HOME"