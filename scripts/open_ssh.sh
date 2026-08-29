#!/bin/bash

sudo apt install ufw -y

# 1. Update the SSH config file using sed
sudo sed -i -E 's/^#?Port.*$/Port 2222/' /etc/ssh/sshd_config

# 2. Tell the local firewall to allow your new port
sudo ufw allow ssh
sudo ufw allow 2222/tcp
# sudo ufw enable

# 3. Apply changes by restarting the network SSH service
sudo systemctl restart ssh

# 4. Verify it worked (should display port 2222)
sudo ss -tlpn | grep ssh | xargs -I {} echo "The port of ssh {}"