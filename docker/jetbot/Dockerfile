# ARG BASE_IMAGE=jetbot-models:jp44
ARG BASE_IMAGE=${BASE_IMAGE}
FROM ${BASE_IMAGE}

# ==============
# INSTALL JETBOT
# ==============
RUN python3 -m pip install pyserial pandas
RUN python3 -m pip install pillow --force-reinstall
ENV JETBOT_REPO_DIR=/opt/jetbot
COPY . ${JETBOT_REPO_DIR}
RUN cd ${JETBOT_REPO_DIR} && python3 setup.py bdist_wheel && cd dist && pip3 install *.whl

