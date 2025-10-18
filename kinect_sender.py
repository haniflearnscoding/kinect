import freenect
import numpy as np
import cv2
from pythonosc.udp_client import SimpleUDPClient
import math
import sys
import time

# OSC client
ip = "127.0.0.1"
port = 8000
client = SimpleUDPClient(ip, port)

# Matrix dimensions (adjust for performance)
MATRIX_WIDTH = 40
MATRIX_HEIGHT = 30

def get_depth():
    depth, _ = freenect.sync_get_depth()
    depth = depth.astype(np.float32)
    # Normalize to 0-1 range for better processing
    depth = np.clip(depth / 2048.0, 0, 1)
    return depth

def get_video():
    rgb, _ = freenect.sync_get_video()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb

def send_matrix_data(address, data, width, height):
    """Send matrix data as a flat list with dimensions"""
    # Send dimensions first
    client.send_message(f"{address}/dim", [width, height])
    
    # Flatten and send data in chunks if needed
    flat_data = data.flatten().tolist()
    
    # For smaller matrices, send in one go
    if len(flat_data) <= 1000:
        client.send_message(f"{address}/data", flat_data)
    else:
        # Send in chunks
        chunk_size = 1000
        num_chunks = math.ceil(len(flat_data) / chunk_size)
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(flat_data))
            chunk = flat_data[start:end]
            client.send_message(f"{address}/chunk", [i, num_chunks] + chunk)

try:
    print("Starting Kinect data stream...")
    frame_count = 0
    
    while True:
        depth = get_depth()
        rgb = get_video()
        
        # Downsample for matrix processing
        depth_small = cv2.resize(depth, (MATRIX_WIDTH, MATRIX_HEIGHT))
        rgb_small = cv2.resize(rgb, (MATRIX_WIDTH, MATRIX_HEIGHT))
        
        # Convert RGB to grayscale for single channel matrix
        gray_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Send matrix dimensions and data
        send_matrix_data("/kinect/depth", depth_small, MATRIX_WIDTH, MATRIX_HEIGHT)
        send_matrix_data("/kinect/rgb", gray_small, MATRIX_WIDTH, MATRIX_HEIGHT)
        
        # Send average depth
        avg_depth = np.nanmean(depth)
        client.send_message("/kinect/avg_depth", float(avg_depth))
        
        # Send min/max for normalization info
        client.send_message("/kinect/depth/range", [float(depth.min()), float(depth.max())])
        
        # Local visualization
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", (depth * 255).astype(np.uint8))
        cv2.imshow("RGB Small", rgb_small)
        cv2.imshow("Depth Small", (depth_small * 255).astype(np.uint8))
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frames sent: {frame_count}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Small delay to prevent overwhelming the network
        time.sleep(0.033)  # ~30 fps

finally:
    print("Shutting down...")
    cv2.destroyAllWindows()
    freenect.sync_stop()
