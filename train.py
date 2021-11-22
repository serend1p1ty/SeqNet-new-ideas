import logging
import os
from functools import partial

import torch
import torch.utils.data
from cpu import EvalHook, Trainer, save_args, set_random_seed, setup_logger

from config import get_argparser
from datasets import build_test_loader, build_train_loader
from eval_func import evaluate_performance
from models.seqnet import SeqNet

logger = logging.getLogger(__name__)


def main(args):
    if args.eval_only:
        # do not save log file when evaluating
        setup_logger()
    else:
        setup_logger(output_dir=args.output_dir)
        save_args(args, os.path.join(args.output_dir, "config.yaml"))

    logger.info("Command line arguments: " + str(args))

    device = torch.device(args.device)
    # If args.seed is negative or None, will use a randomly generated seed
    set_random_seed(args.seed)

    logger.info("Creating model")
    model = SeqNet(args)
    model.to(device)

    logger.info("Loading data")
    train_loader = build_train_loader(args)
    gallery_loader, query_loader = build_test_loader(args)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.base_lr, momentum=args.sgd_momentum, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_decay_milestones, gamma=0.1
    )

    trainer = Trainer(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        args.max_epochs,
        args.output_dir,
        warmup_method="linear",
        warmup_iters=len(train_loader) - 1,
        warmup_factor=0.001,
        clip_grad_norm=args.clip_grad,
        enable_amp=args.enable_amp,
    )

    eval_func = partial(
        evaluate_performance,
        model,
        gallery_loader,
        query_loader,
        device,
        use_gt=args.eval_with_gt,
        use_cache=args.eval_with_cache,
        use_cbgm=args.eval_with_cbgm,
    )
    eval_hook = EvalHook(args.eval_period, eval_func)
    trainer.register_hooks([eval_hook])

    if args.eval_only or args.resume:
        trainer.load_checkpoint(args.checkpoint)
        if args.eval_only:
            eval_func()
            exit(0)

    trainer.train()


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    if args.resume or args.eval_only:
        assert args.checkpoint or args.work_dir
    if args.checkpoint or args.work_dir:
        assert args.resume or args.eval_only
    if args.work_dir:
        args.config_file = os.path.join(args.work_dir, "config.yaml")
        args.checkpoint = os.path.join(args.work_dir, "checkpoints", "latest.pth")
    main(args)
