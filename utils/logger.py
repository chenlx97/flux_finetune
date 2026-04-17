import logging
import os
from datetime import datetime
import yaml
import json
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, config,accelerator):
        self.accelerator = accelerator
        self.is_main = True if accelerator is None else accelerator.is_main_process
        if not self.is_main:
            return
        self.config = config
        # 创建实验目录
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(self.config.training.output_dir, f"log_{time_str}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # log 文件
        log_file = os.path.join(self.exp_dir, "train.log")
        self.metrics_file = os.path.join(self.exp_dir, "metrics.jsonl")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        # console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        # file
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        # 保存 config
        if config is not None:
            self.save_config(config)

    def log_metrics(self, metrics: dict, step: int):
        if not self.accelerator.is_local_main_process:
            return
        record = {
            "step": step,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **metrics
        }
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_config(self, config):
        config_path = os.path.join(self.exp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self._to_dict(config), f)

    def _to_dict(self, obj):
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        elif isinstance(obj, list):
            return [self._to_dict(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            return {k: self._to_dict(v) for k, v in vars(obj).items()}
        else:
            return str(obj)

    def info(self, msg):
        if self.is_main:
            self.logger.info(msg)

    def warning(self, msg):
        if self.is_main:
            self.logger.warning(msg)

    def error(self, msg):
        if self.is_main:
            self.logger.error(msg)
    
    def plot_curves(self, save_path=None, smooth=1):

        if not self.is_main:
            return
        if not os.path.exists(self.metrics_file):
            self.logger.warning("metrics.jsonl not found.")
            return
        steps, losses, lrs = [], [], []
        with open(self.metrics_file, "r") as f:
            for line in f:
                data = json.loads(line)
                steps.append(data.get("step"))
                losses.append(data.get("loss", None))
                lrs.append(data.get("lr", None))
        filtered = [(s, l, lr) for s, l, lr in zip(steps, losses, lrs) if l is not None]
        if len(filtered) == 0:
            self.logger.warning("No valid metrics to plot.")
            return
        steps, losses, lrs = zip(*filtered)

        if smooth > 1:
            import numpy as np
            def moving_avg(x, w):
                return np.convolve(x, np.ones(w) / w, mode="valid")
            losses_plot = moving_avg(losses, smooth)
            steps_plot = steps[smooth - 1:]
        else:
            losses_plot = losses
            steps_plot = steps

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.plot(steps_plot, losses_plot)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Learning Rate")
        ax2.plot(steps, lrs, alpha=0.5)
        plt.title("Training Curves")
        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.exp_dir, "training_curves.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        self.logger.info(f"Saved curves to {save_path}")