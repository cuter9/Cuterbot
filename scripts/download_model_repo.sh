sudo apt install -y python3-pip
sudo pip3 install -U pip gdown
sudo chown -R $(whoami) $HOME/.cache/

gdown --no-cookies --folder https://drive.google.com/drive/folders/1l_PPNVJC2VVXZxv0N63uJhoBWndeIBaC -O ${HOME}/model_repo