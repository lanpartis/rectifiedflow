import os
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from rectifiedflow import RectifiedFlow
from dit import DiTModel
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR scheduler gamma')
    parser.add_argument('--lr_adjust_epoch', type=int, default=10, help='Epoch interval for LR adjustment')
    parser.add_argument('--print_interval', type=int, default=100, help='Print loss interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Model save interval')
    parser.add_argument('--base_channels', type=int, default=64, help='Base channels in model')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of model layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    args = parser.parse_args()

    experiment = f"gaussian2mnist_{args.num_heads}"
    save_path = f"./checkpoints/{experiment}"
    os.makedirs(save_path, exist_ok=True)
    tb_writer = SummaryWriter(f"./tblog/{experiment}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DiTModel(hidden_size=args.base_channels, num_heads=args.num_heads, num_layers=args.num_layers)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size=args.lr_adjust_epoch, gamma=args.gamma)
    rf = RectifiedFlow(model=model)

    with tqdm.trange(0, args.epochs) as t_e:
        for epoch in t_e:
            for batch, data in enumerate(dataloader):
                x_1, y = data
                x_1 = x_1.to(device)
                x_t, t, target = rf.get_train_tuple(z1=x_1, z0=None)
                optimizer.zero_grad()
                v_pred = rf.model(x_t, t)
                loss = torch.nn.functional.mse_loss(v_pred, target=target)
                loss.backward()
                optimizer.step()
                if batch % args.print_interval == 0:
                    tb_writer.add_scalar(
                        "loss", loss.item(), global_step=batch + len(dataloader) * epoch
                    )
                    t_e.set_postfix(batch=batch, loss=loss.item())
            if (1 + epoch) % args.save_interval == 0 or epoch == args.epochs - 1:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, f"{save_path}/{epoch}_{loss.item()}.ckpt")
            scheduler.step()

if __name__ == '__main__':
    main()
