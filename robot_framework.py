import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from ultralytics import YOLO
import time
from collections import deque
import threading

LATENT_DIM = 512    # ???
ACTION_DIM = 2      # Modify this later
NUM_DIFFUSION_STEPS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
        
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
        
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (self.size, self.size))
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class UNetDiffusionModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=ACTION_DIM, time_dim=256, features_start=64, latent_condition_dim=LATENT_DIM):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_condition_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.latent_to_spatial = nn.Sequential(
            nn.Linear(latent_condition_dim, 64 * 8 * 8),
            nn.GELU()
        )
        self.inc = DoubleConv(in_channels, features_start)
        self.downsample = nn.MaxPool2d(kernel_size=4, stride=4)
        self.down1 = Down(features_start, features_start * 2, time_dim * 2)
        self.sa1 = SelfAttention(features_start * 2, 8)
        self.down2 = Down(features_start * 2, features_start * 4, time_dim * 2)
        self.sa2 = SelfAttention(features_start * 4, 4)
        self.down3 = Down(features_start * 4, features_start * 4, time_dim * 2)
        self.sa3 = SelfAttention(features_start * 4, 2)
        self.bot1 = DoubleConv(features_start * 4, features_start * 8)
        self.bot2 = DoubleConv(features_start * 8, features_start * 8)
        self.bot3 = DoubleConv(features_start * 8, features_start * 4)
        self.up1 = Up(features_start * 8, features_start * 2, time_dim * 2)
        self.sa4 = SelfAttention(features_start * 2, 4)
        self.up2 = Up(features_start * 4, features_start, time_dim * 2)
        self.sa5 = SelfAttention(features_start, 8)
        self.up3 = Up(features_start * 2, features_start, time_dim * 2)
        self.sa6 = SelfAttention(features_start, 16)
        self.outc = nn.Sequential(
            nn.Conv2d(features_start, out_channels, kernel_size=1)
        )
        self.final_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 128),
            nn.GELU(),
            nn.Linear(128, ACTION_DIM)
        )

    def forward(self, x, timestep, latent_condition):
        t = self.time_mlp(timestep)
        latent_emb = self.latent_proj(latent_condition)
        t_combined = torch.cat([t, latent_emb], dim=1)
        # For whatever reason this causes error so commenting it out
        # x = self.downsample(x)
        latent_spatial = self.latent_to_spatial(latent_condition)
        latent_spatial = latent_spatial.view(-1, 64, 8, 8)
        _, _, h, w = x.shape
        if h != 8 or w != 8:
            latent_spatial = F.adaptive_avg_pool2d(latent_spatial, (h, w))
        x1 = self.inc(x) + latent_spatial
        x2 = self.down1(x1, t_combined)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t_combined)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_combined)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        x = self.up1(x4, x3, t_combined)
        x = self.sa4(x)
        x = self.up2(x, x2, t_combined)
        x = self.sa5(x)
        x1_down = F.interpolate(x1, size=(x.shape[-2]*2, x.shape[-1]*2), mode='bilinear', align_corners=True)
        x = self.up3(x, x1_down, t_combined)
        x = self.sa6(x)
        x = self.outc(x)
        action = self.final_mlp(x)
        return action

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class YOLOv8LatentExtractor(nn.Module):
    def __init__(self, model_size='m', latent_dim=LATENT_DIM):
        super().__init__()
        self.yolo_model = YOLO(f"yolov8{model_size}.pt")
        self.latent_dim = latent_dim
        test_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            self.yolo_model.model.eval()
            test_output = self.yolo_model.model.model[0:9](test_input)
            
            if isinstance(test_output, list):
                pooled_features = [F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1) for feat in test_output]
                flattened_size = sum(feat.shape[1] for feat in pooled_features)
            else:
                pooled_features = F.adaptive_avg_pool2d(test_output, (1, 1))
                flattened_size = pooled_features.flatten(1).shape[1]
                
        self.projection = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
        for param in self.yolo_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            self.yolo_model.model.eval()
            features = self.yolo_model.model.model[0:9](x)
            detections = self.yolo_model(x, verbose=False)
            
        if isinstance(features, list):
            pooled_features = [F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1) for feat in features]
            flattened_features = torch.cat(pooled_features, dim=1)
        else:
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
            flattened_features = pooled_features.flatten(1)
            
        latent_vector = self.projection(flattened_features)
        return latent_vector, detections

class DiffusionModel:
    def __init__(self, model, action_dim=ACTION_DIM, device=DEVICE):
        self.model = model
        self.action_dim = action_dim
        self.device = device
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.num_diffusion_steps = NUM_DIFFUSION_STEPS
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_diffusion_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.num_diffusion_steps, size=(n,), device=self.device)
    
    def noise_action(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def sample(self, image, latent_condition, n=1):
        self.model.eval()
        x = torch.randn((n, self.action_dim)).to(self.device)
        
        if image is None:
            image = torch.zeros((n, 3, 160, 160)).to(self.device)
            
        steps_to_use = min(10, self.num_diffusion_steps)
        for i in reversed(range(0, steps_to_use)):
            timesteps = torch.full((n,), i, device=self.device, dtype=torch.long)
            try:
                predicted_noise = self.model(image, timesteps, latent_condition)
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                noise = torch.zeros_like(x) if i == 0 else torch.randn_like(x)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
            except Exception as e:
                print(f"Error in diffusion step {i}: {e}")
                break
            
        self.model.train()
        return x

class RobotActionGenerator:
    def __init__(self, yolo_model_size='m', latent_dim=LATENT_DIM, action_dim=ACTION_DIM, device=DEVICE):
        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.feature_extractor = YOLOv8LatentExtractor(model_size=yolo_model_size, latent_dim=latent_dim).to(device)
        self.unet = UNetDiffusionModel(in_channels=3, out_channels=action_dim, latent_condition_dim=latent_dim).to(device)
        self.diffusion = DiffusionModel(model=self.unet, action_dim=action_dim, device=device)
        self.action_buffer = deque(maxlen=5)
        self.lock = threading.Lock()
        self.current_action = torch.zeros(action_dim).to(device)
        self.current_latent = torch.zeros(latent_dim).to(device)
        
    def process_frame(self, frame):
        try:
            orig_frame = frame.copy() if isinstance(frame, np.ndarray) else None
            if isinstance(frame, np.ndarray):
                frame_tensor = torch.from_numpy(frame).float()
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
                orig_h, orig_w = frame.shape[:2]
                aspect = orig_w / orig_h
                
                if aspect > 1:
                    new_w = 640
                    new_h = int(640 / aspect)
                else:
                    new_h = 640
                    new_w = int(640 * aspect)
                    
                frame_tensor = F.interpolate(frame_tensor, size=(new_h, new_w))
                padded = torch.zeros(1, 3, 640, 640, device=frame_tensor.device)
                h_offset = (640 - new_h) // 2
                w_offset = (640 - new_w) // 2
                padded[:, :, h_offset:h_offset+new_h, w_offset:w_offset+new_w] = frame_tensor
                frame_tensor = padded
                frame_tensor = frame_tensor / 255.0
            else:
                frame_tensor = frame
                
            frame_tensor = frame_tensor.to(self.device)
            print(f"Input tensor shape: {frame_tensor.shape}")
            
            with self.lock:
                latent_vector, detections = self.feature_extractor(frame_tensor)
                self.current_latent = latent_vector
                print(f"Latent vector shape: {latent_vector.shape}")
                action = self.diffusion.sample(frame_tensor, latent_vector)[0]
                self.action_buffer.append(action)
                smoothed_action = torch.stack(list(self.action_buffer)).mean(dim=0)
                self.current_action = smoothed_action
                
            viz_frame = orig_frame.copy() if orig_frame is not None else frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            viz_frame = viz_frame.astype(np.uint8)
            
            for det in detections[0].boxes.data:
                x1, y1, x2, y2, conf, cls = det
                if conf > 0.5:
                    cv2.rectangle(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{detections[0].names[int(cls)]}: {conf:.2f}"
                    cv2.putText(viz_frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            action_np = self.current_action.cpu().numpy()
            h, w = viz_frame.shape[:2]
            
            for i, a in enumerate(action_np):
                cv2.putText(viz_frame, f"A{i}: {a:.2f}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            return self.current_action.cpu().numpy(), viz_frame
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            default_action = np.zeros(self.action_dim)
            if isinstance(frame, np.ndarray):
                return default_action, frame.copy()
            else:
                return default_action, np.zeros((640, 640, 3), dtype=np.uint8)
            
    def run_realtime(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                start_time = time.time()
                action, viz_frame = self.process_frame(frame)
                proc_time = time.time() - start_time
                fps = 1.0 / proc_time
                cv2.putText(viz_frame, f"FPS: {fps:.1f}", (viz_frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("YOLOv8 + Diffusion Robot Control", viz_frame)
                print(f"Action: {action}, Processing time: {proc_time:.3f}s")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    import math
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 + Diffusion for Robot Control")
    parser.add_argument("--image", type=str, help="Path to input image file (jpg, png)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID for real-time mode")
    parser.add_argument("--model", type=str, default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--output", type=str, default="output.jpg")
    args = parser.parse_args()
    robot_controller = RobotActionGenerator(yolo_model_size=args.model, latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
    
    if args.image:
        print(f"Processing single image: {args.image}")
        image = cv2.imread(args.image)
        
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        action, viz_frame = robot_controller.process_frame(image)
        proc_time = time.time() - start_time
        viz_frame = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(viz_frame, f"Proc time: {proc_time:.3f}s", (10, viz_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(args.output, viz_frame)
        print(f"Generated action: {action}")
        print(f"Processing time: {proc_time:.3f} seconds")
        print(f"Output saved to: {args.output}")
        cv2.imshow("Result", viz_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Starting real-time vision and action generation...")
        print("Press 'q' to quit")
        robot_controller.run_realtime(camera_id=args.camera)

if __name__ == "__main__":
    import math
    main()
