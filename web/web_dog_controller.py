from flask import Flask, render_template, request, jsonify
import socket
import struct
import threading
import time

# 控制模式常量
PASSIVE = 0
LIE_DOWN = 1
STAND_UP = 2
RL_MODEL = 3
SOFT_STOP = 4


class WebDogController:
    def __init__(self):
        # Initialize UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 修改为本地地址，因为程序会在机器狗上运行
        self.target_addr = ('127.0.0.1', 5526)
        
        # Control parameters
        self.max_linear_speed = 1.0    # Maximum linear velocity (m/s)
        self.max_angular_speed = 1.0   # Maximum angular velocity (rad/s)
        self.min_height = 0.0          # Minimum target height (m)
        self.max_height = 0.4          # Maximum target height (m)
        
        # Current control state
        self.current_mode = PASSIVE
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.target_height = 0.2
        
        # Control thread
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def send_command(self, mode: int, vx: float, vy: float, wz: float,
                     target_height: float):
        """Send command via UDP with target height."""
        try:
            msg = struct.pack('=Bffff', mode, vx, vy, wz, target_height)
            self.sock.sendto(msg, self.target_addr)
        except Exception as e:
            print(f"[ERROR] Failed to send command: {e}")

    def update_control(self, mode: int = None, vx: float = None,
                       vy: float = None, wz: float = None,
                       target_height: float = None):
        """Update control parameters."""
        if mode is not None:
            self.current_mode = mode
        if vx is not None:
            self.vx = max(-self.max_linear_speed,
                          min(self.max_linear_speed, vx))
        if vy is not None:
            self.vy = max(-self.max_linear_speed,
                          min(self.max_linear_speed, vy))
        if wz is not None:
            self.wz = max(-self.max_angular_speed,
                          min(self.max_angular_speed, wz))
        if target_height is not None:
            self.target_height = max(self.min_height,
                                     min(self.max_height, target_height))

    def _control_loop(self):
        """Main control loop running in background."""
        while self.running:
            self.send_command(self.current_mode, self.vx, self.vy, self.wz,
                              self.target_height)
            time.sleep(0.05)  # 20Hz update rate

    def stop_all(self):
        """Emergency stop - set all velocities to zero and mode to PASSIVE."""
        self.current_mode = PASSIVE
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.stop_all()
        time.sleep(0.1)  # Wait for last command to be sent
        self.sock.close()


# Global controller instance
dog_controller = WebDogController()

# Flask app
app = Flask(__name__)


@app.route('/')
def index():
    """Main control interface."""
    return render_template('index.html')


@app.route('/api/control', methods=['POST'])
def control():
    """Handle control commands from web interface."""
    data = request.json
    
    # Update control parameters
    dog_controller.update_control(
        mode=data.get('mode'),
        vx=data.get('vx'),
        vy=data.get('vy'),
        wz=data.get('wz'),
        target_height=data.get('target_height')
    )
    
    return jsonify({
        'status': 'success',
        'mode': dog_controller.current_mode,
        'vx': dog_controller.vx,
        'vy': dog_controller.vy,
        'wz': dog_controller.wz,
        'target_height': dog_controller.target_height
    })


@app.route('/api/mode/<int:mode>', methods=['POST'])
def set_mode(mode):
    """Set control mode."""
    if mode in [PASSIVE, LIE_DOWN, STAND_UP, RL_MODEL, SOFT_STOP]:
        dog_controller.update_control(mode=mode)
        return jsonify({'status': 'success', 'mode': mode})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400


@app.route('/api/stop', methods=['POST'])
def emergency_stop():
    """Emergency stop."""
    dog_controller.stop_all()
    return jsonify({'status': 'success', 'message': 'Emergency stop activated'})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status."""
    return jsonify({
        'mode': dog_controller.current_mode,
        'vx': dog_controller.vx,
        'vy': dog_controller.vy,
        'wz': dog_controller.wz,
        'target_height': dog_controller.target_height
    })


if __name__ == '__main__':
    try:
        # Get local IP address
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print(f"Starting web controller...")
        print(f"Access from your phone: http://{local_ip}:5000")
        print(f"Local access: http://127.0.0.1:5000")
        
        # Run Flask app on all interfaces so it can be accessed from other devices
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        dog_controller.cleanup()
