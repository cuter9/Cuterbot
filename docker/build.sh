#!/bin/bash

LOG_FILE="$HOME"/Cuterbot/docker/docker_build_.log
# shellcheck disable=SC2188
> "$LOG_FILE"

# shellcheck disable=SC2164
cd "$HOME"/Cuterbot/docker
source ./configure.sh >> $LOG_FILE
cd base && ./build.sh && cd .. >> $LOG_FILE
# cd models && ./build.sh && cd .. >> $LOG_FILE
cd display && ./build.sh && cd .. >> $LOG_FILE
cd jupyter && ./build.sh && cd .. >> $LOG_FILE
cd camera && ./build.sh && cd .. >> $LOG_FILE

./disable.sh
docker image prune -f

# shellcheck disable=SC2164
cd "$HOME"