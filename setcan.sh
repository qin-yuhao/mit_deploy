#!/bin/bash


# Configure CAN interfaces
echo "Configuring CAN interfaces..."

ip link set can0 down
ip link set can1 down

ip link set can1 type can bitrate 1000000 fd off
ip link set can0 type can bitrate 1000000 fd off

# Set queue length
ip link set can0 txqueuelen 100
ip link set can1 txqueuelen 100

# Enable CAN interfaces
echo "Enabling CAN interfaces..."
ip link set can1 up
ip link set can0 up

# Verify status
echo "CAN interface status:"
ip link show | grep can

echo "CAN reset completed successfully!"