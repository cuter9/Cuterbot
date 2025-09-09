sudo apt install -y python3-pip
sudo pip3 install -U pip gdown
sudo chown -R $(whoami) $HOME/.cache/
sudo cp ./gdrive_model_repo_cookies.json $HOME/.cache/gdown/cookies.json
# sudo cp ./gdrive_model_repo_cookies.txt $HOME/.cache/gdown/cookies.txt
gdown --no-cookies --folder https://drive.google.com/drive/folders/1-RIQ6lSdFI94IMlZbnRPMQHcG4L8t9Ss -O ${HOME}/model_repo