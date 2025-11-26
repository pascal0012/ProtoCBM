from argparse import Namespace
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from analysis import Logger, AverageMeter, LossMeter
import os

from models.models import ModelXtoC, ModelXtoCtoY, ModelXtoY

def prepare_model(model: nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.compile()

    return model

def logger_and_summarywriter(args: Namespace):
    os.makedirs(args.log_dir, exist_ok=True)

    logger = Logger(os.path.join(args.log_dir, "log.txt"))
    logger.write(str(args) + "\n")
    logger.flush()

    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))

    return logger, tb_writer

def optimizer_and_scheduler_by_name(model: nn.Module, args: Namespace):
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.1
    )

    return optimizer, scheduler

def model_by_mode(args: Namespace) -> nn.Module:
    if args.mode == "XCY":
        model = ModelXtoCtoY(args)
    elif args.mode == "XY":
        model = ModelXtoY(args)
    elif args.mode == "XC":
        model = ModelXtoC(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}")
    
    return model