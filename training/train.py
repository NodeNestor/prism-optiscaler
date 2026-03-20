"""
Prism training loop — GAN + perceptual loss with temporal sequences.

Features:
  - FP8 training via torchao (RTX 4060/5060 Ti with compute 8.9+/12.0)
  - Mixed precision (AMP FP16/BF16) fallback
  - Apollo-Mini optimizer (SGD-level memory, AdamW-level quality)
  - Temporal training: sequences of N consecutive frames with hidden state
  - Multi-GPU: --device cuda:0 or cuda:1 (5060 Ti recommended)
  - Gradient accumulation for effective large batch sizes
  - wandb logging (optional)

Usage:
  # Fast on 5060 Ti with FP8 + Apollo-Mini:
  python train.py --data data/dataset --device cuda:1 --fp8 --optimizer apollo-mini --batch 8

  # Balanced on 4060:
  python train.py --data data/dataset --device cuda:0 --amp --batch 4

  # Quality with wandb logging:
  python train.py --data data/dataset --device cuda:1 --fp8 --model quality --wandb
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from model import (
    PrismDecoder, MultiScaleDiscriminator, PerceptualLoss, HingeLoss,
    prism_fast, prism_balanced, prism_quality,
)


# ============================================================================
# FP8 support via torchao
# ============================================================================

def enable_fp8_training(model: nn.Module) -> nn.Module:
    """Convert model's Linear layers to FP8 for faster training on Ada/Blackwell GPUs."""
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        config = Float8LinearConfig()
        convert_to_float8_training(model, config=config)
        print("  FP8 training enabled via torchao")
        return model
    except Exception as e:
        print(f"  FP8 not available ({e}), falling back to FP16/BF16")
        return model


# ============================================================================
# Optimizer factory
# ============================================================================

def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    """Create optimizer by name."""
    if name == "apollo-mini":
        from apollo_torch import APOLLOAdamW
        param_groups = [{
            "params": list(params),
            "rank": 1,
            "proj": "random",
            "scale_type": "tensor",
            "scale": 128,
            "update_proj_gap": 200,
            "proj_type": "std",
        }]
        return APOLLOAdamW(param_groups, lr=lr, betas=(0.9, 0.999))

    elif name == "apollo":
        from apollo_torch import APOLLOAdamW
        param_groups = [{
            "params": list(params),
            "rank": 256,
            "proj": "random",
            "scale_type": "channel",
            "scale": 1,
            "update_proj_gap": 200,
            "proj_type": "std",
        }]
        return APOLLOAdamW(param_groups, lr=lr, betas=(0.9, 0.999))

    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.0, 0.999))

    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ============================================================================
# Dataset — supports single frames and temporal sequences
# ============================================================================

class PrismDataset(Dataset):
    """
    Simple fast dataset. Loads pre-cropped .pt files directly.
    Splits samples into real/synthetic pools and serves balanced batches.
    """
    def __init__(self, data_dir: Path, crop_size: int = 256, seq_len: int = 1,
                 max_samples: int = 0, **kwargs):
        self.all_samples = sorted(data_dir.glob("sample_*.pt"))
        self.crop_size = crop_size
        self.seq_len = seq_len

        if not self.all_samples:
            raise RuntimeError(f"No samples in {data_dir}")
        if max_samples > 0:
            self.all_samples = self.all_samples[:max_samples]

        # Check if files are pre-cropped (small) or full-size (large)
        sample_size = self.all_samples[0].stat().st_size
        self.pre_cropped = sample_size < 2 * 1024 * 1024  # < 2MB = pre-cropped

        # Split into real and synthetic pools for balanced sampling
        self.real_indices = []
        self.synth_indices = []
        print(f"Scanning {len(self.all_samples)} samples for real/synthetic split...")
        for i, path in enumerate(self.all_samples):
            try:
                data = torch.load(path, weights_only=True)
                if data.get("is_real", torch.tensor(True)).item():
                    self.real_indices.append(i)
                else:
                    self.synth_indices.append(i)
            except:
                self.real_indices.append(i)  # assume real if can't load
            if (i + 1) % 20000 == 0:
                print(f"  Scanned {i+1}/{len(self.all_samples)}")

        print(f"Dataset: {len(self.all_samples)} samples ({len(self.real_indices)} real, "
              f"{len(self.synth_indices)} synthetic), seq_len={seq_len}, "
              f"crop={crop_size}, pre_cropped={self.pre_cropped}")

        # If we have both types, use balanced sampling
        self.balanced = len(self.real_indices) > 0 and len(self.synth_indices) > 0
        if self.balanced:
            import random
            self._rng = random.Random(42)
            print(f"  Balanced sampling: ~50/50 real/synthetic per batch")

    def __len__(self) -> int:
        return len(self.all_samples) - self.seq_len + 1

    def __getitem__(self, idx: int) -> list[dict]:
        seq = []

        # Pick which pool this sequence comes from (real or synthetic)
        # All frames in a sequence come from the same pool for consistency
        if self.balanced:
            use_synth = self._rng.random() < 0.5 and len(self.synth_indices) >= self.seq_len
            pool = self.synth_indices if use_synth else self.real_indices
            start = self._rng.randint(0, max(0, len(pool) - self.seq_len))
        else:
            pool = None
            start = idx

        for i in range(self.seq_len):
            if pool is not None:
                j = pool[min(start + i, len(pool) - 1)]
            else:
                j = (idx + i) % len(self.all_samples)
            try:
                data = torch.load(self.all_samples[j], weights_only=True)
            except Exception:
                j = torch.randint(0, len(self.all_samples), (1,)).item()
                data = torch.load(self.all_samples[j], weights_only=True)

            if not self.pre_cropped:
                data = self._crop(data)

            seq.append(data)
        return seq

    def _crop(self, data: dict) -> dict:
        _, rH, rW = data["color"].shape
        _, dH, dW = data["ground_truth"].shape
        cr = min(self.crop_size, rH, rW)
        cd = cr * 2

        if rH > cr and rW > cr:
            y = torch.randint(0, rH - cr, (1,)).item()
            x = torch.randint(0, rW - cr, (1,)).item()
            dy, dx = int(y * dH / rH), int(x * dW / rW)
            dh, dw = int(cr * dH / rH), int(cr * dW / rW)
        else:
            y, x, dy, dx, dh, dw = 0, 0, 0, 0, dH, dW

        return {
            "color": data["color"][:, y:y+cr, x:x+cr],
            "depth": data["depth"][:, y:y+cr, x:x+cr],
            "motion_vectors": data["motion_vectors"][:, y:y+cr, x:x+cr],
            "ground_truth": F.interpolate(
                data["ground_truth"][:, dy:dy+dh, dx:dx+dw].unsqueeze(0),
                size=(cd, cd), mode="bilinear", align_corners=False
            ).squeeze(0),
            **{k: data[k] for k in ["is_real", "jitter"] if k in data},
        }


def collate_sequences(batch: list[list[dict]]) -> list[dict]:
    seq_len = len(batch[0])
    result = []
    for t in range(seq_len):
        frame = {}
        for key in batch[0][0].keys():
            frame[key] = torch.stack([batch[b][t][key] for b in range(len(batch))])
        result.append(frame)
    return result


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    def __init__(
        self,
        model_name: str = "balanced",
        optimizer_name: str = "adamw",
        lr_g: float = 1e-4,
        lr_d: float = 4e-4,
        device: str = "cuda:1",
        use_amp: bool = True,
        use_fp8: bool = False,
        grad_accum: int = 1,
        use_wandb: bool = False,
    ):
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.use_fp8 = use_fp8
        self.grad_accum = grad_accum

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            gpu_cc = torch.cuda.get_device_capability(self.device)
            print(f"Device: {gpu_name} ({gpu_mem:.0f}GB, compute {gpu_cc[0]}.{gpu_cc[1]})")

        # Models
        factories = {"fast": prism_fast, "balanced": prism_balanced, "quality": prism_quality}
        self.G = factories[model_name]().to(self.device)
        self.D = MultiScaleDiscriminator().to(self.device)
        self.perceptual = PerceptualLoss().to(self.device)

        # FP8 conversion (before optimizer creation)
        if use_fp8:
            self.G = enable_fp8_training(self.G)
            self.D = enable_fp8_training(self.D)

        # torch.compile — disabled for now (Triton issues on compute 12.0)
        # if hasattr(torch, "compile"):
        #     try:
        #         self.G = torch.compile(self.G, mode="reduce-overhead")
        #         self.D = torch.compile(self.D, mode="reduce-overhead")
        #         print("  torch.compile: enabled")
        #     except Exception:
        #         print("  torch.compile: not available")

        g_params = sum(p.numel() for p in self.G.parameters())
        d_params = sum(p.numel() for p in self.D.parameters())
        print(f"Generator: {g_params/1e6:.2f}M params ({model_name})")
        print(f"Discriminator: {d_params/1e6:.2f}M params")

        # Optimizers
        self.opt_G = make_optimizer(optimizer_name, self.G.parameters(), lr=lr_g)
        # Discriminator always uses AdamW (Apollo not needed — D is discarded after training)
        self.opt_D = torch.optim.AdamW(self.D.parameters(), lr=lr_d, betas=(0.0, 0.999))
        print(f"Optimizer G: {optimizer_name} | D: AdamW")

        # Determine autocast dtype
        if use_fp8:
            self.amp_dtype = torch.bfloat16
        elif self.device.type == "cuda":
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.amp_dtype = torch.float32

        # GradScaler only needed for FP16, not BF16 or FP8
        use_scaler = self.use_amp and self.amp_dtype == torch.float16 and not use_fp8
        self.scaler_G = GradScaler("cuda", enabled=use_scaler)
        self.scaler_D = GradScaler("cuda", enabled=use_scaler)

        print(f"Precision: {'FP8+BF16' if use_fp8 else self.amp_dtype}")
        print(f"Grad accumulation: {grad_accum}")

        self.gan_loss = HingeLoss()
        self.epoch = 0

        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project="prism", config={
                "model": model_name, "optimizer": optimizer_name,
                "lr_g": lr_g, "lr_d": lr_d, "fp8": use_fp8, "amp": use_amp,
                "device": str(device),
            })

    def train_epoch(self, loader: DataLoader, adv_weight: float = 0.1,
                    temporal_weight: float = 0.3, d_every: int = 1) -> dict:
        """
        Streaming temporal training — each sample is one step.
        Hidden state flows forward between consecutive samples (detached).
        No backprop through time = same speed as non-temporal.
        Exactly matches inference behavior (one frame at a time).
        """
        self.G.train()
        self.D.train()

        totals = {"g": 0, "d": 0, "l1": 0, "perc": 0, "adv": 0, "temp": 0, "n": 0}

        # Persistent hidden state across steps (streaming)
        prev_output = None
        prev_hidden = None

        for step, sequence in enumerate(tqdm(loader, desc=f"Epoch {self.epoch}")):
            frame = sequence[0]  # seq_len=1 now, single frame per step
            color = frame["color"].to(self.device).float()
            depth = frame["depth"].to(self.device)
            mv = frame["motion_vectors"].to(self.device).float()
            gt = frame["ground_truth"].to(self.device).float()
            is_real = frame.get("is_real", torch.ones(color.shape[0], dtype=torch.bool))
            is_real = is_real.to(self.device)

            # G forward with streaming hidden state from previous step
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp or self.use_fp8):
                fake, hidden = self.G(color, depth, mv,
                                      prev_output=prev_output,
                                      prev_hidden=prev_hidden)

                if gt.shape != fake.shape:
                    gt = F.interpolate(gt, fake.shape[2:], mode="bilinear", align_corners=False)

                # D forward
                fake_preds = self.D(fake)

                if is_real.any():
                    real_gt = gt[is_real]
                    real_preds = self.D(real_gt.detach())
                else:
                    real_preds = None

                # D loss
                fake_preds_detached = [fp.detach() for fp in fake_preds]
                if real_preds is not None:
                    d_loss = self.gan_loss.d_loss(real_preds, fake_preds_detached)
                else:
                    d_loss = F.relu(1 + fake_preds_detached[0]).mean()

                # G loss
                l1 = F.l1_loss(fake, gt)
                adv = self.gan_loss.g_loss(fake_preds)

                # Temporal consistency loss
                if prev_output is not None:
                    from model import warp
                    warped_prev = warp(prev_output, F.interpolate(mv, prev_output.shape[2:],
                                       mode="bilinear", align_corners=False))
                    temp_loss = F.l1_loss(fake, warped_prev)
                else:
                    temp_loss = torch.tensor(0.0, device=self.device)

                g_loss = l1 + adv_weight * adv + temporal_weight * temp_loss

            # Update hidden state (detached — no BPTT, just streaming)
            prev_output = fake.detach()
            prev_hidden = hidden.detach() if hidden is not None else None

            # Reset hidden state occasionally to prevent staleness
            if step % 100 == 0:
                prev_output = None
                prev_hidden = None

            # Backward + step
            self.opt_D.zero_grad()
            d_loss.backward(retain_graph=True)
            self.opt_D.step()

            self.opt_G.zero_grad()
            g_loss.backward()
            self.opt_G.step()

            B = color.shape[0]
            totals["g"] += g_loss.item() * B
            totals["d"] += d_loss.item() * B
            totals["l1"] += l1.item() * B
            totals["perc"] += 0.0
            totals["adv"] += adv.item() * B
            totals["temp"] += temp_loss.item() * B
            totals["n"] += B

        n = max(totals["n"], 1)
        self.epoch += 1
        metrics = {k: totals[k] / n for k in ["g", "d", "l1", "perc", "adv", "temp"]}

        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=self.epoch)

            # Log sample images every 5 epochs
            if self.epoch % 5 == 0:
                self._log_sample_images()

        return metrics

    def set_fixed_sample(self, loader):
        """Grab a fixed sample from the dataset for consistent wandb image logging."""
        for sequence in loader:
            frame = sequence[0]
            self._fixed_color = frame["color"][:1].to(self.device).float()
            self._fixed_depth = frame["depth"][:1].to(self.device).float()
            self._fixed_mv = frame["motion_vectors"][:1].to(self.device).float()
            self._fixed_gt = frame["ground_truth"][:1].to(self.device).float()
            self._fixed_is_real = frame.get("is_real", torch.ones(1, dtype=torch.bool))[0].item()
            break

    @torch.no_grad()
    def _log_sample_images(self):
        """Generate and log sample images to wandb."""
        import wandb
        import numpy as np

        if not hasattr(self, "_fixed_color"):
            return

        self.G.eval()

        try:
            c = self._fixed_color
            d = self._fixed_depth
            mv = self._fixed_mv
            gt = self._fixed_gt
            _, _, rH, rW = c.shape
            _, _, dH, dW = gt.shape

            def to_np(t):
                return (t[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            images = {}

            # Input visualizations
            images["inputs/color"] = wandb.Image(to_np(c), caption=f"Color {rW}x{rH}")
            depth_np = d[0, 0].cpu().numpy()
            depth_np = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-7) * 255).astype(np.uint8)
            images["inputs/depth"] = wandb.Image(depth_np, caption="Depth")
            images["inputs/ground_truth"] = wandb.Image(to_np(gt), caption=f"Ground truth {dW}x{dH}")
            tag = "real" if self._fixed_is_real else "synthetic"
            images["inputs/type"] = wandb.Image(to_np(c), caption=f"Source: {tag}")

            # 2x upscale
            target_2x = (rH * 2, rW * 2)
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out_2x, _ = self.G(c, d, mv, target_h=target_2x[0], target_w=target_2x[1])
            nearest_2x = F.interpolate(c, target_2x, mode="nearest")
            bilinear_2x = F.interpolate(c, target_2x, mode="bilinear", align_corners=False)
            images["2x/nearest"] = wandb.Image(to_np(nearest_2x), caption="Nearest 2x")
            images["2x/bilinear"] = wandb.Image(to_np(bilinear_2x), caption="Bilinear 2x")
            images["2x/prism"] = wandb.Image(to_np(out_2x), caption=f"Prism 2x (ep{self.epoch})")

            # 3x upscale
            target_3x = (rH * 3, rW * 3)
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out_3x, _ = self.G(c, d, mv, target_h=target_3x[0], target_w=target_3x[1])
            nearest_3x = F.interpolate(c, target_3x, mode="nearest")
            images["3x/nearest"] = wandb.Image(to_np(nearest_3x), caption="Nearest 3x")
            images["3x/prism"] = wandb.Image(to_np(out_3x), caption=f"Prism 3x (ep{self.epoch})")

            # Side-by-side comparison at 2x
            comparison = torch.cat([nearest_2x, out_2x, F.interpolate(gt, target_2x, mode="bilinear", align_corners=False)], dim=3)
            images["compare/nearest_vs_prism_vs_gt"] = wandb.Image(
                to_np(comparison), caption="Nearest | Prism | Ground Truth")

            wandb.log(images, step=self.epoch)

        except Exception as e:
            print(f"  [wandb image log failed: {e}]")
        finally:
            self.G.train()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "epoch": self.epoch,
        }, path / f"checkpoint_ep{self.epoch}.pth")
        torch.save(self.G.state_dict(), path / "prism_generator_latest.pth")
        print(f"Saved epoch {self.epoch}")

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.G.load_state_dict(ckpt["generator"])
        self.D.load_state_dict(ckpt["discriminator"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.epoch = ckpt["epoch"]
        print(f"Resumed from epoch {self.epoch}")

    def export_onnx(self, path: Path, render_size: tuple[int, int] = (540, 960)):
        self.G.eval()
        rH, rW = render_size
        dummy = (
            torch.randn(1, 3, rH, rW, device=self.device),
            torch.randn(1, 1, rH, rW, device=self.device),
            torch.randn(1, 2, rH, rW, device=self.device),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self.G, dummy, str(path), opset_version=17,
            input_names=["color", "depth", "motion_vectors"],
            output_names=["output", "hidden"],
            dynamic_axes={
                "color": {0: "batch", 2: "height", 3: "width"},
                "depth": {0: "batch", 2: "height", 3: "width"},
                "motion_vectors": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
                "hidden": {0: "batch", 2: "height", 3: "width"},
            },
        )
        print(f"Exported ONNX: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Prism G-buffer decoder")
    parser.add_argument("--data", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--model", choices=["fast", "balanced", "quality"], default="balanced")
    parser.add_argument("--optimizer", choices=["adamw", "apollo-mini", "apollo"], default="apollo-mini")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--progressive", action="store_true",
                        help="Progressive training: start small crops, grow over epochs")
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=4e-4)
    parser.add_argument("--amp", action="store_true", help="FP16/BF16 mixed precision")
    parser.add_argument("--fp8", action="store_true", help="FP8 training (RTX 4060+/5060 Ti)")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--adv-warmup", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default="cuda:1", help="cuda:0=4060, cuda:1=5060Ti")
    parser.add_argument("--multi-gpu", action="store_true", help="Use DataParallel across all GPUs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--d-every", type=int, default=1, help="Train D every N steps (1=every step)")
    args = parser.parse_args()

    # Progressive training schedule: crop size grows over epochs
    if args.progressive:
        schedule = [
            (64, int(args.epochs * 0.4)),    # 40% of epochs at 64x64 (fast)
            (128, int(args.epochs * 0.35)),   # 35% at 128x128 (medium)
            (192, int(args.epochs * 0.15)),   # 15% at 192x192 (detailed)
            (256, int(args.epochs * 0.10)),   # 10% at 256x256 (full context)
        ]
        print(f"Progressive training schedule:")
        for crop, epochs in schedule:
            print(f"  {crop}x{crop} for {epochs} epochs")
    else:
        schedule = [(args.crop, args.epochs)]

    total_epochs_done = 0

    for phase, (crop_size, phase_epochs) in enumerate(schedule):
        print(f"\n{'='*60}")
        print(f"Phase {phase}: crop={crop_size}x{crop_size}, {phase_epochs} epochs")
        print(f"{'='*60}")

        dataset = PrismDataset(args.data, crop_size=crop_size, seq_len=args.seq_len)
        loader = DataLoader(
            dataset, batch_size=args.batch, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            drop_last=True, collate_fn=collate_sequences,
        )

        if phase == 0:
            trainer = Trainer(
                model_name=args.model, optimizer_name=args.optimizer,
                lr_g=args.lr_g, lr_d=args.lr_d,
                device=args.device, use_amp=args.amp, use_fp8=args.fp8,
                grad_accum=args.grad_accum, use_wandb=args.wandb,
            )

            # Multi-GPU with DataParallel
            if args.multi_gpu and torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
                trainer.G = torch.nn.DataParallel(trainer.G)
                trainer.D = torch.nn.DataParallel(trainer.D)

            if args.resume:
                trainer.load(args.resume)

        # Grab a fixed sample for wandb image logging
        if args.wandb and phase == 0:
            trainer.set_fixed_sample(loader)

        print(f"\nTraining: {phase_epochs} epochs | batch={args.batch} | seq={args.seq_len} | "
              f"crop={crop_size} | {'FP8' if args.fp8 else 'AMP' if args.amp else 'FP32'} | "
              f"{args.optimizer} | d_every={args.d_every}\n")

        try:
            for epoch in range(phase_epochs):
                total_epoch = total_epochs_done + epoch
                adv_w = 0.1 + 0.4 * min(1.0, total_epoch / max(args.adv_warmup, 1))

                t0 = time.time()
                m = trainer.train_epoch(loader, adv_weight=adv_w, d_every=args.d_every)
                dt = time.time() - t0

                print(f"  [{total_epoch+1}] L1={m['l1']:.4f} adv={m['adv']:.4f}(w={adv_w:.2f}) "
                      f"temp={m['temp']:.4f} D={m['d']:.4f} [{dt:.1f}s]")

                if (total_epoch + 1) % args.save_every == 0:
                    trainer.save(args.output)
        except KeyboardInterrupt:
            print("\nInterrupted")
            break

        total_epochs_done += phase_epochs

    trainer.save(args.output)
    if args.export_onnx:
        trainer.export_onnx(args.output / "prism_decoder.onnx")
    print("Done!")


if __name__ == "__main__":
    main()
