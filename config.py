from cpu import ConfigArgumentParser


def get_argparser():
    parser = ConfigArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Whether to resume from a checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only.")
    parser.add_argument(
        "--checkpoint", default="", help="Path of the checkpoint to resume or evaluate."
    )
    parser.add_argument("--work-dir", type=str, default="", help="Path of the working directory")
    parser.add_argument("--backbone", type=str, default="seresnet50")
    parser.add_argument("--dataset", default="CUHK-SYSU", choices=["CUHK-SYSU", "PRW"])
    parser.add_argument("--data-root", default="data/CUHK-SYSU")
    parser.add_argument(
        "--min-size", type=int, default=900, help="Size of the smallest side of the image."
    )
    parser.add_argument(
        "--max-size", type=int, default=1500, help="Maximum size of the side of the image."
    )
    parser.add_argument(
        "--batchsize-train", type=int, default=5, help="Number of images per batch in training."
    )
    parser.add_argument(
        "--batchsize-test", type=int, default=1, help="Number of images per batch in test."
    )
    parser.add_argument("--workers-train", type=int, default=5)
    parser.add_argument("--workers-test", type=int, default=1)
    parser.add_argument(
        "--ckpt-period",
        type=int,
        default=1,
        help="Save a checkpoint after every this number of epochs.",
    )
    parser.add_argument("--device", default="cuda", help="The device loading the model.")
    parser.add_argument(
        "--seed", type=int, default=1, help="Set seed to negative to fully randomize everything."
    )
    parser.add_argument(
        "--output-dir", default="output", help="Directory where output files are written."
    )
    parser.add_argument(
        "--enable-amp",
        action="store_true",
        help="Whether to enable Automatic Mixed Precision (AMP) training.",
    )

    # solver
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--base-lr", type=float, default=0.003)
    parser.add_argument(
        "--lr-decay-milestones",
        type=int,
        default=[16],
        nargs="+",
        help="The epoch milestones to decrease the learning rate by GAMMA.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument(
        "--lw-rpn-reg", type=float, default=1.0, help="Loss weight of RPN regression."
    )
    parser.add_argument(
        "--lw-rpn-cls", type=float, default=1.0, help="Loss weight of RPN classification."
    )
    parser.add_argument(
        "--lw-proposal-reg", type=float, default=10.0, help="Loss weight of proposal regression."
    )
    parser.add_argument(
        "--lw-proposal-cls", type=float, default=1.0, help="Loss weight of proposal classification."
    )
    parser.add_argument(
        "--lw-box-reg", type=float, default=1.0, help="Loss weight of box regression."
    )
    parser.add_argument(
        "--lw-box-cls", type=float, default=1.0, help="Loss weight of box classification."
    )
    parser.add_argument(
        "--lw-box-reid",
        type=float,
        default=1.0,
        help="Loss weight of box OIM (i.e. Online Instance Matching).",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=10.0,
        help="Set to negative value to disable gradient clipping.",
    )

    # RPN
    parser.add_argument(
        "--rpn-nms-thresh", type=float, default=0.7, help="NMS threshold used on RoIs."
    )
    parser.add_argument(
        "--rpn-batchsize-train",
        type=int,
        default=256,
        help="Number of anchors per image used to train RPN.",
    )
    parser.add_argument(
        "--rpn-pos-frac-train",
        type=float,
        default=0.5,
        help="Target fraction of foreground examples per RPN minibatch.",
    )
    parser.add_argument(
        "--rpn-pos-thresh-train",
        type=float,
        default=0.7,
        help=(
            "Overlap threshold for an anchor to be considered foreground "
            "(if >= POS_THRESH_TRAIN)."
        ),
    )
    parser.add_argument(
        "--rpn-neg-thresh-train",
        type=float,
        default=0.3,
        help="Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN).",
    )
    parser.add_argument(
        "--rpn-pre-nms-topn-train",
        type=int,
        default=12000,
        help="Number of top scoring RPN RoIs to keep before applying NMS.",
    )
    parser.add_argument("--rpn-pre-nms-topn-test", type=int, default=6000)
    parser.add_argument(
        "--rpn-post-nms-topn-train",
        type=int,
        default=2000,
        help="Number of top scoring RPN RoIs to keep after applying NMS.",
    )
    parser.add_argument("--rpn-post-nms-topn-test", type=int, default=300)

    # RoI head
    parser.add_argument(
        "--roihead-no-bnneck",
        action="store_true",
        help="Whether to not use bn neck (i.e. batch normalization after linear).",
    )
    parser.add_argument(
        "--roihead-batchsize-train",
        type=int,
        default=128,
        help="Number of RoIs per image used to train RoI head.",
    )
    parser.add_argument(
        "--roihead-pos-frac-train",
        type=float,
        default=0.5,
        help="Target fraction of foreground examples per RoI minibatch.",
    )
    parser.add_argument(
        "--roihead-pos-thresh-train",
        type=float,
        default=0.5,
        help="Overlap threshold for an RoI to be considered foreground (if >= POS_THRESH_TRAIN).",
    )
    parser.add_argument(
        "--roihead-neg-thresh-train",
        type=float,
        default=0.5,
        help="Overlap threshold for an RoI to be considered background (if < NEG_THRESH_TRAIN).",
    )
    parser.add_argument(
        "--roihead-score-thresh-test", type=float, default=0.5, help="Minimum score threshold."
    )
    parser.add_argument(
        "--roihead-nms-thresh-test", type=float, default=0.4, help="NMS threshold used on boxes."
    )
    parser.add_argument(
        "--roihead-detections-per-img-test",
        type=int,
        default=300,
        help="Maximum number of detected objects.",
    )

    # OIM
    parser.add_argument(
        "--oim-lut-size", type=int, default=5532, help="Size of the lookup table in OIM."
    )
    parser.add_argument(
        "--oim-cq-size", type=int, default=5000, help="Size of the circular queue in OIM."
    )
    parser.add_argument("--oim-momentum", type=float, default=0.5)
    parser.add_argument("--oim-scalar", type=float, default=30.0)

    # evaluation
    parser.add_argument(
        "--eval-period",
        type=int,
        default=1,
        help="The period to evaluate the model during training.",
    )
    parser.add_argument(
        "--eval-with-gt",
        action="store_true",
        help="Evaluation with GT boxes to verify the upper bound of person search performance.",
    )
    parser.add_argument(
        "--eval-with-cache", action="store_true", help="Fast evaluation with cached features."
    )
    parser.add_argument(
        "--eval-with-cbgm",
        action="store_true",
        help="Evaluation with Context Bipartite Graph Matching (CBGM) algorithm.",
    )

    return parser
