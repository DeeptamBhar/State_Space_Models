import cv2
import torch
import time
import threading
import csv
import os
import numpy as np
from collections import deque
from torchvision import transforms

from models.hybrid_model import CNNMambaEdge

class ThreadedCamera:
    def __init__(self, src=0):
        # Using V4L2 to bypass the GStreamer crash
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        
        # Trying to force MJPG if the camera supports it
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.ret, self.frame = self.cap.read()
        self.running = True
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame

    def read(self):
        return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def run_live_thermal_test(handoff_idx, duration_minutes=30, log_interval=30):
    print(f"Live Camera Thermal Test: Handoff {handoff_idx}")
    print(f"Duration: {duration_minutes} mins | Logging Avg to CSV every {log_interval} sec")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup CSV
    os.makedirs("results", exist_ok=True)
    csv_filename = f"results/MAXN_live_thermal_handoff_{handoff_idx}.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Elapsed_Time_Sec', 'Avg_System_FPS', 'Avg_GPU_Latency_ms'])

    # Load Model
    print("Loading model architecture and weights")
    model = CNNMambaEdge(num_classes=101, handoff_index=handoff_idx, cnn_size='small').to(device)
    weights_path = f"checkpoints/best_model_handoff_{handoff_idx}.pth"
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError(f"No weights found at {weights_path}")
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Warming up camera thread")
    cam = ThreadedCamera(0)
    time.sleep(2)

    if not cam.ret:
        print("ERROR: Could not read from webcam.")
        cam.stop()
        return

    frame_buffer = deque(maxlen=16)
    duration_seconds = duration_minutes * 60
    
    # Tracking variables
    overall_start_time = time.time()
    interval_start_time = time.perf_counter()
    
    frames_processed = 0
    total_latency_in_interval = 0.0

    print(f"Test Started, Running for {duration_minutes} minutes. (Press Ctrl+C to stop)")
    
    try:
        while True:
            loop_start = time.perf_counter()
            ret, frame = cam.read()
            
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = transform(frame_rgb)
            frame_buffer.append(tensor_frame)
            
            if len(frame_buffer) == 16:
                video_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
                
                # Time the GPU
                torch.cuda.synchronize()
                gpu_start = time.perf_counter()
                
                with torch.no_grad():
                    outputs = model(video_tensor)
                    
                torch.cuda.synchronize()
                gpu_end = time.perf_counter()
                
                latency_ms = (gpu_end - gpu_start) * 1000
                total_latency_in_interval += latency_ms
                frames_processed += 1
                
                # Log to CSV
                if time.perf_counter() - interval_start_time >= log_interval:
                    elapsed_total = time.time() - overall_start_time
                    
                    # Calculate Averages
                    avg_system_fps = frames_processed / (time.perf_counter() - interval_start_time)
                    avg_latency = total_latency_in_interval / max(1, frames_processed)
                    
                    # Console Output
                    mins = int(elapsed_total // 60)
                    secs = int(elapsed_total % 60)
                    print(f"⏱️ {mins:02d}:{secs:02d} | ⚡ Sys FPS: {avg_system_fps:.1f} | 🧠 GPU Latency: {avg_latency:.1f} ms")
                    
                    # Write to CSV
                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([round(elapsed_total, 1), round(avg_system_fps, 2), round(avg_latency, 2)])

                    # Reset accumulators for next interval
                    frames_processed = 0
                    total_latency_in_interval = 0.0
                    interval_start_time = time.perf_counter()
            
            # Check exit condition
            if (time.time() - overall_start_time) >= duration_seconds:
                print(f"\n✅ Live Thermal Test Complete! Data saved to {csv_filename}")
                break
            
            # Prevent CPU overload and avoid processing the same frame multiple times
            time.sleep(0.005) 
            
    except KeyboardInterrupt:
        print(f"Test aborted early. Data up to this point is safe in {csv_filename}.")
    
    finally:
        cam.stop()

if __name__ == "__main__":
    handoffs = [4,6,8,10,12]
    
    for i, h in enumerate(handoffs):
        run_live_thermal_test(handoff_idx=h, duration_minutes=30, log_interval=30)
        
        # Cool down for 2 minutes between models, but skip the wait after the final model!
        if i < len(handoffs) - 1:
            next_h = handoffs[i+1]
            print(f"Cooling down Jetson for 2 minutes before starting Handoff {next_h}")
            time.sleep(120)