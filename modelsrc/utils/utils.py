import torch

def clip_gradient(optimizer, grad_clip):
    """
    Trim the gradients for the TMSOD model to prevent gradient explosion.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30, min_lr=1e-6, verbose=False):
    """
    Dynamic adjustment of learning rate, suitable for your long-term training TMSOD.
    Support setting a minimum learning rate to avoid getting stuck in gradient stagnation too early.
    """
    decay = decay_rate ** (epoch // decay_epoch)
    new_lr = max(init_lr * decay, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    if verbose:
        print(f"Learning Rate adjusted to: {new_lr:.6f}")
    return new_lr


class AverageMeter:
    """
    The average value monitor is used to record the average changes of indicators such as MAE, F-measure, and Loss.
    Statistical analysis that can be used in the training/validation process.
    """
    def __init__(self, name='Metric'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (Current: {self.val:.4f})"

