#!/bin/bash

echo "Setting up RL Deploy service..."

# Copy service file to systemd directory
echo "Installing systemd service file..."
sudo cp rl_deploy.service /etc/systemd/system/

# Set proper permissions
sudo chmod 644 /etc/systemd/system/rl_deploy.service

# Reload systemd daemon
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the service (start on boot)
echo "Enabling service..."
sudo systemctl enable rl_deploy.service

echo ""
echo "RL Deploy service installation completed!"
echo ""
echo "Available commands:"
echo "  Start:      sudo systemctl start rl_deploy"
echo "  Stop:       sudo systemctl stop rl_deploy"
echo "  Restart:    sudo systemctl restart rl_deploy"
echo "  Status:     sudo systemctl status rl_deploy"
echo "  Check boot: sudo systemctl is-enabled rl_deploy"
echo "  Logs:       sudo journalctl -u rl_deploy -f"
echo ""
echo "The service will automatically start on next boot."
echo "To start the service now: sudo systemctl start rl_deploy"