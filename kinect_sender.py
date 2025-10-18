import freenect
import numpy as np
import cv2
from pythonosc.udp_client import SimpleUDPClient
import math
import sys

# OSC client
ip = "127.0.0.1"
port = 8000  # Wekinator default input port
client = SimpleUDPClient(ip, port)

# Chunk size (bytes) to keep UDP packets small (e.g., 1000 bytes)
CHUNK_SIZE = 1000  # Reduced to avoid "Message too long" error

def get_depth():
    depth, _ = freenect.sync_get_depth()
    # Normalize depth to 0-255 for visualization
    depth = depth.astype(np.float32)
    depth = np.clip(depth / 2048.0 * 255.0, 0, 255).astype(np.uint8)
    return depth

def get_video():
    rgb, _ = freenect.sync_get_video()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb

def send_chunked_data(address, data, chunk_size=CHUNK_SIZE):
    """Send a flattened array in chunks via OSC."""
    total_values = len(data)
    values_per_chunk = chunk_size // 4  # float32 = 4 bytes per value
    num_chunks = math.ceil(total_values / values_per_chunk)
    
    for i in range(num_chunks):
        start = i * values_per_chunk
        end = min(start + values_per_chunk, total_values)
        chunk = data[start:end].tolist()
        # Estimate OSC message size (approx. 4 bytes per float + overhead)
        approx_size = len(chunk) * 4 + 100  # 100 bytes for OSC overhead
        if approx_size > 1472:  # Typical max UDP payload without fragmentation
            print(f"Warning: Chunk {i+1}/{num_chunks} for {address} may be too large ({approx_size} bytes)", file=sys.stderr)
        client.send_message(f"{address}/chunk", [i, num_chunks] + chunk)
        print(f"Sent {address} chunk {i+1}/{num_chunks}, size: {len(chunk)} values (~{approx_size} bytes)")

try:
    while True:
        depth = get_depth()
        rgb = get_video()
        
        # Downsample images to 80x60 to reduce data
        rgb_small = cv2.resize(rgb, (80, 60))
        depth_small = cv2.resize(depth, (80, 60))
        
        # Flatten RGB (3 channels) and depth (1 channel) for OSC
        rgb_flat = rgb_small.flatten().astype(np.float32)  # float32 for OSC
        depth_flat = depth_small.flatten().astype(np.float32)
        
        # Send RGB and depth data in chunks
        send_chunked_data("/kinect/rgb", rgb_flat)
        send_chunked_data("/kinect/depth", depth_flat)
        
        # Optional: Send average depth
        avg_depth = np.nanmean(depth)
        client.send_message("/kinect/avg_depth", float(avg_depth))
        print("Sending avg_depth:", avg_depth)
        
        # Optional local visualization
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    freenect.sync_stop()
