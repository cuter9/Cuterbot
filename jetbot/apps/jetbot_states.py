#!/usr/bin/env python
#
# INA3221_jetbot
# Cuter Hsu
# 03/05/2024
#
#

# imports

# import sys
# import time
# import datetime
# from jetbot.apps.ina3221 import INA3221
from jetbot.apps import INA3221
from jtop import jtop

# Main Program

# filename = time.strftime("%Y-%m-%d%H:%M:%SRTCTest") + ".txt"
# starttime = datetime.datetime.utcnow()

# parameters for ina3221 power sensor on jetson nano
# the three channels of the INA3221 named for SunAirPlus Solar Power Controller channels (www.switchdoc.com)
BOARD = 1
GPU = 2
CPU = 3
I2C_BUS = 6  # jetbot i2c bus no and address of ina3221
I2C_ADDRESS = 0x40


class NanoStates:
    def __init__(self):
        self.jetson = jtop()
        self.jetson.start()
        while not self.jetson.ok():
            pass
        self.jetson.jetson_clocks = True
        self.jetson.jetson_clocks.boot = True
        self.jetson.fan.speed = 50

    @property
    def pwr_states(self):
        while self.jetson.ok():
            pwr_sensor = self.jetson.power
            in_current = pwr_sensor['tot']['curr'] / 1000
            in_voltage = pwr_sensor['tot']['volt'] / 1000
            in_power = pwr_sensor['tot']['power'] / 1000
            return {"in_volt": in_voltage,
                    "in_current": in_current,
                    "in_pwr": in_power}

class JetbotStates:
    def __init__(self):
        self.pwr_sensor = INA3221(twi=I2C_BUS, force=True, addr=I2C_ADDRESS)  # force : force to read ic2 data

    @property
    def pwr_states(self, channel=BOARD):
        bus_voltage = self.pwr_sensor.getBusVoltage_V(channel)  # in V
        shunt_voltage = self.pwr_sensor.getShuntVoltage_mV(channel)  # in mV
        # minus is to get the "sense" right.   - means the battery is charging, + that it is discharging
        in_current = self.pwr_sensor.getCurrent_mA(channel)  # in mA

        in_voltage = bus_voltage + (shunt_voltage / 1000)  # in Volt
        in_power = in_current * in_voltage / 1000  # in Watt
        return {"in_volt": in_voltage,
                "in_current": in_current,
                "shunt_volt": shunt_voltage,
                "bus_volt": bus_voltage,
                "in_pwr": in_power}
