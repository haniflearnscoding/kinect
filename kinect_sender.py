# kinect_sender.py
import freenect
import numpy as np
import cv2
from pythonosc.udp_client import SimpleUDPClient

# OSC client
ip = "127.0.0.1"
port = 8000  # Wekinator default input port
client = SimpleUDPClient(ip, port)

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth.astype(np.float32)

def get_video():
    rgb, _ = freenect.sync_get_video()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb

try:
    while True:
        depth = get_depth()
        rgb = get_video()
        
        # Send average depth to Max
        avg_depth = np.nanmean(depth)
        client.send_message("/kinect/avg_depth", float(avg_depth))
        print("Sending:", avg_depth)  # optional debugging
        
        # Optional local visualization
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth / 2048.0)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
    cv2.destroyAllWindows()
    freenect.sync_stop()
