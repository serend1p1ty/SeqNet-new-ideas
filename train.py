import argparse
import logging
import os
from functools import partial

import torch
import torch.utils.data
from cpu import EvalHook, Trainer, collect_env, set_random_seed, setup_logger

from datasets import build_test_loader, build_train_loader
from defaults import get_default_cfg
from eval_func import evaluate_performance
from models.seqnet import SeqNet

logger = logging.getLogger(__name__)


def setup(args):
    cfg = get_default_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.eval_only:
        setup_logger()
    else:
        setup_logger(output=cfg.OUTPUT_DIR)

        logger.info("Environment info:\n" + collect_env())
        logger.info("Command line arguments: " + str(args))
        file_content = open(args.config_file, "r").read()
        logger.info(f"Contents of args.config_file={args.config_file}:\n{file_content}")
        logger.info(f"Running with full config:\n{cfg.dump()}")

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
        with open(filename, "w") as f:
            f.write(cfg.dump())
    return cfg


def main(args):
    cfg = setup(args)

    device = torch.device(cfg.DEVICE)
    set_random_seed(cfg.SEED)

    logger.info("Creating model")
    model = SeqNet(cfg, only_res5=args.only_res5)
    model.to(device)

    logger.info("Loading data")
    train_loader = build_train_loader(cfg)
    gallery_loader, query_loader = build_test_loader(cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    trainer = Trainer(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        cfg.SOLVER.MAX_EPOCHS,
        cfg.OUTPUT_DIR,
        warmup_method="linear",
        warmup_iters=len(train_loader) - 1,
        warmup_factor=0.001,
        clip_grad_norm=cfg.SOLVER.CLIP_GRADIENTS,
        enable_amp=cfg.ENABLE_AMP,
    )

    eval_func = partial(
        evaluate_performance,
        model,
        gallery_loader,
        query_loader,
        device,
        use_gt=cfg.EVAL_USE_GT,
        use_cache=cfg.EVAL_USE_CACHE,
        use_cbgm=cfg.EVAL_USE_CBGM,
    )
    eval_hook = EvalHook(cfg.EVAL_PERIOD, eval_func)
    trainer.register_hooks([eval_hook])

    if args.eval_only or args.resume:
        trainer.load_checkpoint(args.checkpoint)
        if args.eval_only:
            eval_func()
            exit(0)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", help="Path of the configuration file.")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from a checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only.")
    parser.add_argument(
        "--checkpoint", default="", help="Path of the checkpoint to resume or evaluate."
    )
    parser.add_argument("--work-dir", type=str, default="", help="Path of the working directory")
    parser.add_argument("--only_res5", action="store_true")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    if args.resume or args.eval_only:
        assert args.checkpoint or args.work_dir
    if args.checkpoint or args.work_dir:
        assert args.resume or args.eval_only
    if args.work_dir:
        args.config_file = os.path.join(args.work_dir, "config.yaml")
        args.checkpoint = os.path.join(args.work_dir, "checkpoints", "latest.pth")
    main(args)
