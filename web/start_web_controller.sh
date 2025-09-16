#!/bin/bash

#echo "Installing Flask..."
#pip3 install -r requirements.txt

echo ""
echo "Starting Dog Web Controller..."
echo ""
echo "The web interface will be available at:"
echo "- Local: http://127.0.0.1:5000"
echo "- Network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 web_dog_controller.py
