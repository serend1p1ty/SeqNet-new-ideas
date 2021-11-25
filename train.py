import logging
from functools import partial

import torch
import torch.utils.data
from cpu import (
    EvalHook,
    Trainer,
    collect_env,
    default_argparser,
    highlight,
    save_config,
    set_random_seed,
    setup_logger,
)

from datasets import build_test_loader, build_train_loader
from defaults import get_default_cfg
from eval_func import evaluate_performance
from models.seqnet import SeqNet

logger = logging.getLogger(__name__)


def main(args):
    cfg = get_default_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    setup_logger(output=cfg.OUTPUT_DIR)
    logger.info(f"\n{collect_env()}")
    print(
        "Contents of args.config_file={}:\n{}".format(
            args.config_file, highlight(open(args.config_file, "r").read(), args.config_file)
        )
    )
    print(f"Running with full config:\n{highlight(cfg.dump(), '.yaml')}")

    device = torch.device(cfg.DEVICE)
    set_random_seed(cfg.SEED)

    logger.info("Creating model")
    model = SeqNet(cfg)
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

    logger.info("Start training")
    save_config(cfg, cfg.OUTPUT_DIR)
    trainer.train()


if __name__ == "__main__":
    parser = default_argparser()
    args = parser.parse_args()
    main(args)
