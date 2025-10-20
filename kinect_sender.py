import freenect
import numpy as np
import cv2
from pythonosc.udp_client import SimpleUDPClient

# OSC client
ip = "127.0.0.1"
port = 8000
client = SimpleUDPClient(ip, port)

def get_depth():
    depth, _ = freenect.sync_get_depth()
    depth = depth.astype(np.float32)
    depth = np.clip(depth / 2048.0 * 255.0, 0, 255).astype(np.uint8)
    return depth

def get_video():
    rgb, _ = freenect.sync_get_video()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb.astype(np.uint8)

try:
    while True:
        # --- GET DATA ---
        depth = get_depth()
        rgb = get_video()
        
        # Downsample to 40x30 for RGB averaging
        rgb_small = cv2.resize(rgb, (40, 30))
        
        # --- SEND AVERAGE DEPTH ---
        avg_depth = float(np.nanmean(depth))
        client.send_message("/kinect/avg_depth", avg_depth)
        print("Sending avg_depth:", avg_depth)
        
        # --- SEND AVERAGE RGB ---
        avg_r = int(np.mean(rgb_small[:, :, 0]))
        avg_g = int(np.mean(rgb_small[:, :, 1]))
        avg_b = int(np.mean(rgb_small[:, :, 2]))
        client.send_message("/kinect/rgb", [avg_r, avg_g, avg_b])
        print("Sending avg_rgb:", [avg_r, avg_g, avg_b])
        
        # --- LOCAL VISUALIZATION ---
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    freenect.sync_stop()
