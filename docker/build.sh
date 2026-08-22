#!/bin/bash

LOG_FILE="$HOME"/Cuterbot/docker/docker_build_.log
> $LOG_FILE

# shellcheck disable=SC2164
cd "$HOME"/Cuterbot/docker
pre_docker_build.sh
source ./configure.sh >> $LOG_FILE
cd base && ./build.sh && cd .. >> $LOG_FILE
# cd models && ./build.sh && cd .. >> $LOG_FILE
cd display && ./build.sh && cd .. >> $LOG_FILE
cd jupyter && ./build.sh && cd .. >> $LOG_FILE
cd camera && ./build.sh && cd .. >> $LOG_FILE

cd "$HOME"