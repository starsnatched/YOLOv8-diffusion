import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from ultralytics import YOLO
import time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YOLOFeatureExtractor:
    def __init__(self, model_size='n', pretrained=True, feature_dim=512):
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.feature_dim = feature_dim
        backbone_dims = {'n': 256, 's': 512, 'm': 768, 'l': 1024, 'x': 1280}
        self.backbone_dim = backbone_dims.get(model_size, 256)
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim * 2),
            nn.LayerNorm(self.backbone_dim * 2),
            nn.GELU(),
            nn.Linear(self.backbone_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        ).to(device)
        self.hooks = []
        self._register_hooks()
        self.backbone_features = None
        self.detection_features = None

    def _register_hooks(self):
        def backbone_hook(module, input, output):
            self.backbone_features = output
        def detection_hook(module, input, output):
            self.detection_features = output
        for name, module in self.model.model.named_modules():
            if "backbone.stage4.2" in name:
                self.hooks.append(module.register_forward_hook(backbone_hook))
            elif "detect" in name:
                self.hooks.append(module.register_forward_hook(detection_hook))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_features(self, img):
        self.backbone_features = None
        self.detection_features = None
        results = self.model(img, verbose=False)
        detections = results[0]
        if self.backbone_features is not None:
            backbone_feat = F.adaptive_avg_pool2d(self.backbone_features, (1, 1))
            backbone_feat = backbone_feat.view(-1, self.backbone_dim)
        else:
            backbone_feat = torch.zeros(1, self.backbone_dim, device=device)
        det_features = []
        if len(detections.boxes) > 0:
            boxes = detections.boxes.xyxy.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()
            class_ids = detections.boxes.cls.cpu().numpy().astype(int)
            for i in range(min(10, len(boxes))):
                box = boxes[i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                area = width * height
                aspect_ratio = width / (height + 1e-5)
                det_feat = np.array([box[0], box[1], box[2], box[3],
                                     width, height, center_x, center_y,
                                     area, aspect_ratio, confs[i], class_ids[i]])
                det_features.append(det_feat)
        if det_features:
            det_tensor = torch.tensor(np.vstack(det_features), dtype=torch.float32).to(device)
            det_tensor = torch.mean(det_tensor, dim=0, keepdim=True)
        else:
            det_tensor = torch.zeros(1, 12, device=device)
        det_tensor = det_tensor.view(1, -1)
        projected_features = self.feature_projector(backbone_feat)
        if not hasattr(self, 'detection_projector'):
            self.detection_projector = nn.Linear(det_tensor.shape[1], 64).to(device)
        det_embedding = self.detection_projector(det_tensor)
        combined_features = torch.cat([projected_features, det_embedding], dim=1)
        if not hasattr(self, 'output_projector'):
            self.output_projector = nn.Linear(self.feature_dim + 64, self.feature_dim).to(device)
        final_features = self.output_projector(combined_features)
        return final_features

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        b, C, h, w = x.size()
        proj_query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key(x).view(b, -1, h * w)
        attention = self.softmax(torch.bmm(proj_query, proj_key))
        proj_value = self.value(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, C, h, w)
        return self.gamma * out + x

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        div_term = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, device=device) * -div_term)
        pos_enc = time[:, None] * inv_freq[None, :]
        return torch.cat((pos_enc.sin(), pos_enc.cos()), dim=-1)

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embedding_layer = nn.Linear(embedding_dim, num_features * 2)
        self.embedding_layer.weight.data[:, :num_features] = 1.0
        self.embedding_layer.weight.data[:, num_features:] = 0.0
        self.embedding_layer.bias.data[:num_features] = 0.0
        self.embedding_layer.bias.data[num_features:] = 0.0
    def forward(self, x, embedding):
        out = self.bn(x)
        embedding = self.embedding_layer(embedding)
        gamma, beta = embedding.chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * out + beta

class DoubleConvWithNorm(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = ConditionalBatchNorm2d(mid_channels, embedding_dim)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = ConditionalBatchNorm2d(out_channels, embedding_dim)
        self.relu2 = nn.SiLU(inplace=True)
    def forward(self, x, embedding):
        x = self.conv1(x)
        x = self.bn1(x, embedding)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, embedding)
        return self.relu2(x)

class DownWithNorm(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvWithNorm(in_channels, out_channels, embedding_dim)
    def forward(self, x, embedding):
        x = self.pool(x)
        return self.conv(x, embedding)

class UpWithNorm(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvWithNorm(in_channels, out_channels, embedding_dim, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvWithNorm(in_channels, out_channels, embedding_dim)
    def forward(self, x1, x2, embedding):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, embedding)

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, condition_dim):
        super().__init__()
        self.condition_proj = nn.Linear(condition_dim, in_channels * 2)
    def forward(self, x, condition):
        condition = self.condition_proj(condition)
        gamma, beta = condition.chunk(2, dim=1)
        gamma = gamma.view(-1, gamma.shape[1], 1, 1)
        beta = beta.view(-1, beta.shape[1], 1, 1)
        return gamma * x + beta

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = ConditionalBatchNorm2d(out_channels, embedding_dim)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(out_channels, embedding_dim)
        self.relu2 = nn.SiLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, embedding):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x, embedding)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, embedding)
        return self.relu2(x + shortcut)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, condition_dim=512, embedding_dim=256, bilinear=False):
        super().__init__()
        self.time_embed = nn.Sequential(
            PositionalEncoding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.combined_embed = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.inc = DoubleConvWithNorm(n_channels, 64, embedding_dim)
        self.down1 = DownWithNorm(64, 128, embedding_dim)
        self.down2 = DownWithNorm(128, 256, embedding_dim)
        self.sa1 = SelfAttention(256)
        self.down3 = DownWithNorm(256, 512, embedding_dim)
        self.sa2 = SelfAttention(512)
        factor = 2 if bilinear else 1
        self.down4 = DownWithNorm(512, 1024 // factor, embedding_dim)
        self.sa3 = SelfAttention(1024 // factor)
        self.up1 = UpWithNorm(1024, 512 // factor, embedding_dim, bilinear)
        self.sa4 = SelfAttention(512 // factor)
        self.res1 = ResidualBlock(64, 64, embedding_dim)
        self.up2 = UpWithNorm(512, 256 // factor, embedding_dim, bilinear)
        self.sa5 = SelfAttention(256 // factor)
        self.up3 = UpWithNorm(256, 128, embedding_dim, bilinear)
        self.up4 = UpWithNorm(128, 64, embedding_dim, bilinear)
        self.res2 = ResidualBlock(512 // factor, 512 // factor, embedding_dim)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.feature_affine = FeatureWiseAffine(64, condition_dim)
    def forward(self, x, timestep, condition):
        t_emb = self.time_embed(timestep)
        c_emb = self.cond_embed(condition)
        emb = self.combined_embed(torch.cat([t_emb, c_emb], dim=1))
        x1 = self.inc(x, emb)
        x1 = self.res1(x1, emb)
        x2 = self.down1(x1, emb)
        x3 = self.down2(x2, emb)
        x3 = self.sa1(x3)
        x4 = self.down3(x3, emb)
        x4 = self.sa2(x4)
        x5 = self.down4(x4, emb)
        x5 = self.sa3(x5)
        x = self.up1(x5, x4, emb)
        x = self.sa4(x)
        x = self.res2(x, emb)
        x = self.up2(x, x3, emb)
        x = self.sa5(x)
        x = self.up3(x, x2, emb)
        x = self.up4(x, x1, emb)
        x = self.feature_affine(x, condition)
        return self.outc(x)

class DiffusionModel:
    def __init__(self, input_dim=512, action_dim=6, timesteps=1000, embedding_dim=256):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.timesteps = timesteps
        self.embedding_dim = embedding_dim
        self.spatial_dim = 16
        self.unet = UNet(n_channels=1, n_classes=1, condition_dim=input_dim, embedding_dim=embedding_dim).to(device)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.spatial_dim * self.spatial_dim)
        ).to(device)
        self.action_decoder = nn.Sequential(
            nn.Linear(self.spatial_dim * self.spatial_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, action_dim)
        ).to(device)
        self._setup_noise_schedule()
        self.optimizer = torch.optim.AdamW(list(self.unet.parameters()) + list(self.action_encoder.parameters()) + list(self.action_decoder.parameters()), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-6)
        self.ema_rate = 0.9999
        self.ema_params = None

    def _setup_noise_schedule(self):
        def cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps) / timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clip(betas, 0.0001, 0.9999)
        self.beta = cosine_beta_schedule(self.timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        self.log_one_minus_alpha_cumprod = torch.log(1 - self.alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1 / self.alpha_cumprod)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1 / self.alpha_cumprod - 1)
        self.posterior_variance = self.beta[:-1] * (1 - self.alpha_cumprod[:-1]) / (1 - self.alpha_cumprod[1:])
        self.posterior_variance = torch.cat([self.posterior_variance, torch.tensor([0.0]).to(device)])
        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.beta[:-1] * torch.sqrt(self.alpha_cumprod[:-1]) / (1 - self.alpha_cumprod[:-1])
        self.posterior_mean_coef2 = (1 - self.alpha_cumprod[:-1]) * torch.sqrt(self.alpha[:-1]) / (1 - self.alpha_cumprod[:-1])

    def update_ema(self):
        if self.ema_params is None:
            self.ema_params = {
                'unet': {k: v.clone().detach() for k, v in self.unet.state_dict().items()},
                'encoder': {k: v.clone().detach() for k, v in self.action_encoder.state_dict().items()},
                'decoder': {k: v.clone().detach() for k, v in self.action_decoder.state_dict().items()},
            }
        else:
            with torch.no_grad():
                for model_name, model in [('unet', self.unet), ('encoder', self.action_encoder), ('decoder', self.action_decoder)]:
                    for k, v in model.state_dict().items():
                        self.ema_params[model_name][k] = self.ema_params[model_name][k] * self.ema_rate + v * (1 - self.ema_rate)

    def encode_action(self, action):
        batch_size = action.shape[0]
        x = self.action_encoder(action)
        return x.view(batch_size, 1, self.spatial_dim, self.spatial_dim)

    def decode_action(self, spatial_x):
        batch_size = spatial_x.shape[0]
        x = spatial_x.view(batch_size, -1)
        return self.action_decoder(x)

    def q_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    def p_mean_variance(self, x_t, t, condition):
        predicted_noise = self.unet(x_t, t, condition)
        x_0_predicted = self._predict_x0_from_xt(x_t, t, predicted_noise)
        x_0_predicted = torch.clamp(x_0_predicted, -1.0, 1.0)
        model_mean = self._q_posterior_mean(x_0_predicted, x_t, t)
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        posterior_log_variance = self.posterior_log_variance[t].view(-1, 1, 1, 1)
        return model_mean, posterior_variance, posterior_log_variance

    def _predict_x0_from_xt(self, x_t, t, noise):
        sqrt_recip_alpha_cumprod_t = self.sqrt_recip_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alpha_cumprod_t = self.sqrt_recipm1_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_recip_alpha_cumprod_t * x_t - sqrt_recipm1_alpha_cumprod_t * noise

    def _q_posterior_mean(self, x_0, x_t, t):
        t_safe = torch.clamp(t, 1) - 1
        mask = (t > 0).float().view(-1, 1, 1, 1)
        posterior_mean_coef1 = mask * self.posterior_mean_coef1[t_safe].view(-1, 1, 1, 1)
        posterior_mean_coef2 = mask * self.posterior_mean_coef2[t_safe].view(-1, 1, 1, 1)
        return posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t

    def p_sample(self, x_t, t, condition):
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, t, condition)
        noise = torch.randn_like(x_t) * (t > 0).float().view(-1, 1, 1, 1)
        return model_mean + torch.exp(0.5 * model_log_variance) * noise

    def sample(self, condition, timesteps=None, use_ema=True):
        if timesteps is None:
            timesteps = self.timesteps
        if timesteps < self.timesteps:
            skips = self.timesteps // timesteps
            timestep_seq = list(range(0, self.timesteps, skips))
            if timestep_seq[-1] != self.timesteps - 1:
                timestep_seq.append(self.timesteps - 1)
        else:
            timestep_seq = list(range(self.timesteps))
        timestep_seq = sorted(timestep_seq, reverse=True)
        batch_size = condition.shape[0]
        if use_ema and self.ema_params is not None:
            orig_unet = {k: v.clone() for k, v in self.unet.state_dict().items()}
            orig_encoder = {k: v.clone() for k, v in self.action_encoder.state_dict().items()}
            orig_decoder = {k: v.clone() for k, v in self.action_decoder.state_dict().items()}
            self.unet.load_state_dict(self.ema_params['unet'])
            self.action_encoder.load_state_dict(self.ema_params['encoder'])
            self.action_decoder.load_state_dict(self.ema_params['decoder'])
        try:
            x = torch.randn(batch_size, 1, self.spatial_dim, self.spatial_dim).to(device)
            for t in timestep_seq:
                t_batch = torch.ones(batch_size, device=device, dtype=torch.long) * t
                with torch.no_grad():
                    x = self.p_sample(x, t_batch, condition)
            actions = self.decode_action(x)
            return torch.tanh(actions)
        finally:
            if use_ema and self.ema_params is not None:
                self.unet.load_state_dict(orig_unet)
                self.action_encoder.load_state_dict(orig_encoder)
                self.action_decoder.load_state_dict(orig_decoder)

    def forward_diffusion(self, x_0, t):
        return self.q_sample(x_0, t)

    def reverse_diffusion(self, features, timesteps=None):
        return self.sample(features, timesteps)

    def train_step(self, x_0, features):
        batch_size = x_0.shape[0]
        x_0_spatial = torch.tanh(self.encode_action(x_0))
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        x_t, noise = self.q_sample(x_0_spatial, t)
        predicted_noise = self.unet(x_t, t, features)
        simple_loss = F.mse_loss(predicted_noise, noise)
        x_0_predicted = self._predict_x0_from_xt(x_t, t, predicted_noise)
        x0_loss = F.mse_loss(x_0_predicted, x_0_spatial)
        vlb_loss = self._compute_vlb_loss(x_t, x_0_spatial, t, predicted_noise)
        loss = simple_loss + 0.1 * x0_loss + 0.001 * vlb_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.unet.parameters()) + list(self.action_encoder.parameters()) + list(self.action_decoder.parameters()), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.update_ema()
        return loss.item()

    def _compute_vlb_loss(self, x_t, x_0, t, predicted_noise):
        true_mean = self._q_posterior_mean(x_0, x_t, t)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(
            x_t, t, condition=self.unet.cond_embed(torch.zeros(x_t.shape[0], self.input_dim, device=device))
        )
        kl = 0.5 * torch.mean((true_mean - model_mean) ** 2 / model_variance + model_log_variance - self.posterior_log_variance[t.clamp(min=1) - 1].view(-1, 1, 1, 1))
        decoder_nll = torch.mean(0.5 * torch.log(2 * torch.pi) + 0.5 * ((x_0 - model_mean) ** 2))
        vlb = torch.where(t == 0, decoder_nll, kl)
        return vlb.mean()

class RobotVisionActionSystem:
    def __init__(self, yolo_model_size='n', feature_dim=512, action_dim=6, buffer_size=10):
        print(f"Initializing YOLOv8 {yolo_model_size} model with feature extraction...")
        self.yolo = YOLOFeatureExtractor(model_size=yolo_model_size, feature_dim=feature_dim)
        print("Initializing diffusion model for action generation...")
        self.diffusion = DiffusionModel(input_dim=feature_dim, action_dim=action_dim)
        self.action_buffer = []
        self.buffer_size = buffer_size
        self.action_space = {
            0: ["move_forward", "translation", {"axis": "z", "max_velocity": 0.5}],
            1: ["move_backward", "translation", {"axis": "z", "max_velocity": -0.5}],
            2: ["turn_left", "rotation", {"axis": "y", "max_angular_velocity": 0.8}],
            3: ["turn_right", "rotation", {"axis": "y", "max_angular_velocity": -0.8}],
            4: ["grasp", "gripper", {"max_force": 10.0, "width": 0.0}],
            5: ["release", "gripper", {"max_force": 5.0, "width": 0.08}]
        }
        self.action_history = []
        self.feature_history = []
        self.timestamp_history = []
        self.safety_constraints = {
            "max_velocity": 0.5,
            "max_angular_velocity": 0.8,
            "acceleration_limit": 1.0,
            "jerk_limit": 2.0,
            "workspace_limits": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 0.8]}
        }
        self.adaptive_params = {"noise_reduction": 0.8, "confidence_threshold": 0.6, "sensitivity": 1.0}
        self.metrics = {"total_frames": 0, "processed_frames": 0, "skipped_frames": 0, "avg_processing_time": 0, "success_rate": 0, "cumulative_reward": 0}
        self._optimize_models()

    def _optimize_models(self):
        try:
            dummy_input = torch.zeros(1, self.diffusion.input_dim).to(device)
            self.diffusion_traced = torch.jit.trace(lambda x: self.diffusion.sample(x, timesteps=10), dummy_input)
            print("Models optimized with TorchScript for faster inference")
        except Exception as e:
            print(f"Model optimization failed: {e}. Using regular models.")
            self.diffusion_traced = None

    def process_frame(self, frame, timestamp=None):
        self.metrics["total_frames"] += 1
        if timestamp is None:
            timestamp = time.time()
        features = self.yolo.get_features(frame)
        self.feature_history.append(features.detach().cpu())
        if len(self.feature_history) > 30:
            self.feature_history.pop(0)
            self.timestamp_history.pop(0)
        self.timestamp_history.append(timestamp)
        if self.diffusion_traced is not None:
            actions = self.diffusion_traced(features)
        else:
            actions = self.diffusion.sample(features, timesteps=10)
        smoothed_actions = self._smooth_actions(actions)
        commands = self._actions_to_safe_commands(smoothed_actions)
        self.action_history.append(smoothed_actions.detach().cpu())
        if len(self.action_history) > 100:
            self.action_history.pop(0)
        self.metrics["processed_frames"] += 1
        metadata = {
            "timestamp": timestamp,
            "confidence": self._compute_confidence(features, actions),
            "frame_quality": self._assess_frame_quality(frame),
            "safety_score": self._compute_safety_score(smoothed_actions),
            "action_entropy": self._compute_action_entropy(actions)
        }
        return features, smoothed_actions, commands, metadata

    def _smooth_actions(self, actions):
        self.action_buffer.append(actions)
        if len(self.action_buffer) > self.buffer_size:
            self.action_buffer.pop(0)
        if len(self.action_buffer) < 2:
            return actions
        buffer_tensor = torch.stack(self.action_buffer)
        weights = torch.tensor([self.adaptive_params["noise_reduction"] ** (self.buffer_size - i - 1) for i in range(len(self.action_buffer))], device=device)
        weights = weights / weights.sum()
        return (weights.view(-1, 1, 1) * buffer_tensor).sum(dim=0)

    def _compute_confidence(self, features, actions):
        feature_norm = torch.norm(features).item()
        action_norm = torch.norm(actions).item()
        return torch.sigmoid(torch.tensor(feature_norm / (action_norm + 1e-5))).item()

    def _assess_frame_quality(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        norm_brightness = min(1.0, brightness / 128)
        norm_contrast = min(1.0, contrast / 64)
        norm_blur = min(1.0, blur / 500)
        return 0.2 * norm_brightness + 0.3 * norm_contrast + 0.5 * norm_blur

    def _compute_safety_score(self, actions):
        action_values = actions.cpu().numpy()[0]
        velocities = action_values[:3] * self.safety_constraints["max_velocity"]
        angular_velocities = action_values[3:6] * self.safety_constraints["max_angular_velocity"]
        vel_exceeded = np.any(np.abs(velocities) > self.safety_constraints["max_velocity"])
        ang_exceeded = np.any(np.abs(angular_velocities) > self.safety_constraints["max_angular_velocity"])
        acc_exceeded = False
        jerk_exceeded = False
        if len(self.action_history) >= 2:
            prev_actions = self.action_history[-1].numpy()
            prev_velocities = prev_actions[:3] * self.safety_constraints["max_velocity"]
            dt = self.timestamp_history[-1] - self.timestamp_history[-2] if len(self.timestamp_history) >= 2 else 1/30.0
            acceleration = np.abs((velocities - prev_velocities) / dt)
            acc_exceeded = np.any(acceleration > self.safety_constraints["acceleration_limit"])
            if len(self.action_history) >= 3:
                prev_prev_actions = self.action_history[-2].numpy()
                prev_prev_velocities = prev_prev_actions[:3] * self.safety_constraints["max_velocity"]
                prev_acceleration = (prev_velocities - prev_prev_velocities) / dt
                jerk = np.abs((acceleration - prev_acceleration) / dt)
                jerk_exceeded = np.any(jerk > self.safety_constraints["jerk_limit"])
        violations = sum([vel_exceeded, ang_exceeded, acc_exceeded, jerk_exceeded])
        return max(0, 1 - (violations * 0.25))

    def _compute_action_entropy(self, actions):
        probs = F.softmax(actions, dim=1)[0].cpu().numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = -np.log(1.0 / len(probs))
        return 1 - (entropy / max_entropy)

    def _actions_to_safe_commands(self, actions):
        batch_size = actions.shape[0]
        commands = []
        for b in range(batch_size):
            action_values = actions[b].cpu().numpy()
            safety_scale = min(1.0, self._compute_safety_score(actions))
            scaled_values = action_values * safety_scale
            continuous_commands = {}
            for i, val in enumerate(scaled_values):
                if i >= len(self.action_space):
                    continue
                action_name, action_type, params = self.action_space[i]
                if action_type == "translation":
                    speed = val * params["max_velocity"]
                    continuous_commands[f"{params['axis']}_velocity"] = float(speed)
                elif action_type == "rotation":
                    angular_speed = val * params["max_angular_velocity"]
                    continuous_commands[f"{params['axis']}_angular_velocity"] = float(angular_speed)
                elif action_type == "gripper":
                    if val > self.adaptive_params["confidence_threshold"]:
                        continuous_commands["gripper_position"] = params["width"]
                        continuous_commands["gripper_force"] = params["max_force"]
            max_action_idx = np.argmax(np.abs(action_values))
            max_action_val = action_values[max_action_idx]
            action_name, action_type, params = self.action_space[max_action_idx]
            command = {
                "action": action_name,
                "action_type": action_type,
                "value": float(max_action_val),
                "continuous_values": continuous_commands,
                "safety_scale": safety_scale,
                "full_actions": scaled_values.tolist()
            }
            commands.append(command)
        return commands

    def run_real_time(self, camera_id=0, display=True, fps_target=30):
        print(f"Starting real-time vision-action system from camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        frame_times = []
        processing_times = []
        target_frame_time = 1.0 / fps_target
        try:
            while True:
                cycle_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                start_time = time.time()
                features, actions, commands, metadata = self.process_frame(frame)
                process_time = time.time() - start_time
                processing_times.append(process_time)
                if len(processing_times) > 100:
                    processing_times.pop(0)
                self.metrics["avg_processing_time"] = np.mean(processing_times)
                if display:
                    viz_frame = self._create_visualization(frame, features, actions, commands, metadata)
                    cv2.imshow("Robot Vision Action System", viz_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                print(f"Commands: {commands}")
                print(f"Processing: {process_time:.4f}s, FPS: {1.0/process_time:.2f}, Confidence: {metadata['confidence']:.2f}, Safety: {metadata['safety_score']:.2f}")
                elapsed = time.time() - cycle_start
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                frame_times.append(time.time() - cycle_start)
                if len(frame_times) > 100:
                    frame_times.pop(0)
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            print("\nFinal System Metrics:")
            print(f"Total frames: {self.metrics['total_frames']}")
            print(f"Processed frames: {self.metrics['processed_frames']}")
            print(f"Average processing time: {self.metrics['avg_processing_time']:.4f}s")
            print(f"Average FPS: {1.0/np.mean(frame_times):.2f}")

    def _create_visualization(self, frame, features, actions, commands, metadata):
        viz_frame = frame.copy()
        overlay = viz_frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 240), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, viz_frame, 0.3, 0, viz_frame)
        cv2.putText(viz_frame, "Vision-Diffusion Control System", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2)
        fps = 1.0 / metadata["timestamp"] if "timestamp" in metadata else 0
        cv2.putText(viz_frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(viz_frame, f"Frame Quality: {metadata['frame_quality']:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(viz_frame, f"Confidence: {metadata['confidence']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(viz_frame, f"Safety Score: {metadata['safety_score']:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos = 140
        for cmd in commands:
            color = (0, 0, 255) if abs(cmd["value"]) > 0.3 else (180, 180, 180)
            text = f"{cmd['action']}: {cmd['value']:.2f}"
            cv2.putText(viz_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            bar_length = int(abs(cmd["value"]) * 100)
            bar_start = 200
            if cmd["value"] > 0:
                cv2.rectangle(viz_frame, (bar_start, y_pos-10), (bar_start+bar_length, y_pos-2), color, -1)
            else:
                cv2.rectangle(viz_frame, (bar_start-bar_length, y_pos-10), (bar_start, y_pos-2), color, -1)
            y_pos += 20
        return viz_frame

def train_diffusion_model(system, dataset_path=None, num_epochs=100, batch_size=16):
    print("Training diffusion model...")
    if dataset_path is None:
        print("Using synthetic data for training...")
        num_samples = 1000
        feature_dim = system.diffusion.input_dim
        action_dim = system.diffusion.action_dim
        features = torch.randn(num_samples, feature_dim).to(device)
        actions = torch.tanh(torch.randn(num_samples, action_dim).to(device))
    else:
        print(f"Loading dataset from {dataset_path}...")
        features = torch.load(f"{dataset_path}/features.pt").to(device)
        actions = torch.load(f"{dataset_path}/actions.pt").to(device)
        num_samples = features.shape[0]
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            batch_features = features[i*batch_size:(i+1)*batch_size]
            batch_actions = actions[i*batch_size:(i+1)*batch_size]
            loss = system.diffusion.train_step(batch_actions, batch_features)
            total_loss += loss
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    print("Training complete!")
    torch.save({
        'unet': system.diffusion.unet.state_dict(),
        'encoder': system.diffusion.action_encoder.state_dict(),
        'decoder': system.diffusion.action_decoder.state_dict()
    }, "diffusion_model.pth")
    print("Model weights saved!")

def main():
    print("Initializing Robot Vision Action System...")
    system = RobotVisionActionSystem(yolo_model_size='n', action_dim=6)
    print("Running real-time system...")
    system.run_real_time(camera_id=0, display=True)

if __name__ == "__main__":
    main()
