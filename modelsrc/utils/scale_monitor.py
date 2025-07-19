import os
import csv
import torch

class ScaleMonitor:
    def __init__(self, model, module_path='thermal_prior', log_dir='./', log_name='scale_log_2.csv'):
        self.model = model
        self.module_path = module_path
        self.log_path = os.path.join(log_dir, log_name)
        self.header_written = False

        os.makedirs(log_dir, exist_ok=True)

    def _get_scale_param(self):
        module = self.model
        for attr in self.module_path.split('.'):
            module = getattr(module, attr)
        return module.scale

    def record(self, epoch, step=None):
        try:
            scale = self._get_scale_param()
            scale_val = scale.item()
            scale_grad = scale.grad.item() if scale.grad is not None else 0.0
        except Exception as e:
            print(f"[ScaleMonitor] Failed to access scale: {e}")
            return

        if step is not None:
            print(f"[ScaleMonitor] Epoch {epoch}, Step {step} | scale = {scale_val:.4f}, grad = {scale_grad:.6f}")
        else:
            print(f"[ScaleMonitor] Epoch {epoch} | scale = {scale_val:.4f}, grad = {scale_grad:.6f}")

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not self.header_written:
                writer.writerow(['Epoch', 'Step', 'ScaleValue', 'ScaleGrad'])
                self.header_written = True
            writer.writerow([epoch, step if step is not None else '-', scale_val, scale_grad])
