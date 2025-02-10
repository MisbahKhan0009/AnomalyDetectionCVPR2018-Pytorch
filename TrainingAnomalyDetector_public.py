""" Training procedure for video anomaly detection with multi-GPU optimization."""

import argparse
import os
from os import makedirs, path

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from features_loader import FeaturesLoader
from network.anomaly_detector_model import (
    AnomalyDetector,
    RegularizedLoss,
    custom_objective,
)
from network.TorchUtils import TorchModel
from utils.callbacks import DefaultModelCallback, TensorBoardCallback
from utils.utils import register_logger

# Explicitly set both GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video Anomaly Detection Training")
    parser.add_argument("--features_path", required=True,
                        help="path to features")
    parser.add_argument("--annotation_path", required=True,
                        help="path to train annotation")
    parser.add_argument("--log_file", type=str,
                        default="log.log", help="logging file")
    parser.add_argument("--exps_dir", type=str,
                        default="exps", help="experiments directory")
    parser.add_argument("--checkpoint", type=str,
                        help="load model for resume training")
    parser.add_argument("--save_every", type=int, default=1,
                        help="checkpoint save interval")
    parser.add_argument("--lr_base", type=float,
                        default=0.01, help="learning rate")
    parser.add_argument("--iterations_per_epoch", type=int,
                        default=20000, help="training iterations")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of training epochs")
    return parser.parse_args()


def main():
    args = get_args()

    # Setup directories
    register_logger(log_file=args.log_file)
    makedirs(args.exps_dir, exist_ok=True)
    models_dir = path.join(args.exps_dir, "models")
    tb_dir = path.join(args.exps_dir, "tensorboard")
    makedirs(models_dir, exist_ok=True)
    makedirs(tb_dir, exist_ok=True)

    # GPU Configuration
    device = torch.device("cuda")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    # Data loader
    train_loader = FeaturesLoader(
        features_path=args.features_path,
        annotation_path=args.annotation_path,
        iterations=args.iterations_per_epoch,
    )

    feature_dim = train_loader.get_feature_dim

    # Model setup
    if args.checkpoint and path.exists(args.checkpoint):
        model = TorchModel.load_model(args.checkpoint)
        assert feature_dim == model.model.input_dim, "Input dimension mismatch"
    else:
        network = AnomalyDetector(feature_dim)
        model = TorchModel(network)

    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # Training setup
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr_base, eps=1e-8)
    criterion = RegularizedLoss(model.get_model(), custom_objective).to(device)

    # Tensorboard
    tb_writer = SummaryWriter(log_dir=tb_dir)
    model.register_callback(DefaultModelCallback(
        visualization_dir=args.exps_dir))
    model.register_callback(TensorBoardCallback(tb_writer=tb_writer))

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = criterion(batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            model.save(path.join(models_dir, f"model_epoch_{epoch+1}.pth"))

    # Final model save
    model.save(path.join(models_dir, "final_model.pth"))
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
