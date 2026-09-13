from jetbot.apps import JetbotStates
# jetbot_states need right to access /dev/ic2; thus run first sudo usermod -aG i2c $(whoami) and reboot

pwr_sensor = JetbotStates()

for i in range(5):
    pwr_states = pwr_sensor.pwr_states()
    print(pwr_states)
    i += 1

