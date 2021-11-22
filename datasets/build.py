import logging

import torch

from utils.transforms import build_transforms

from .cuhk_sysu import CUHKSYSU
from .prw import PRW

logger = logging.getLogger(__name__)


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def print_statistics(dataset):
    """
    Print dataset statistics.
    """
    num_imgs = len(dataset.annotations)
    num_boxes = 0
    pid_set = set()
    for anno in dataset.annotations:
        num_boxes += anno["boxes"].shape[0]
        for pid in anno["pids"]:
            pid_set.add(pid)
    statistics = {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_images": num_imgs,
        "num_boxes": num_boxes,
    }
    if dataset.name != "CUHK-SYSU" or dataset.split != "query":
        pid_list = sorted(list(pid_set))
        if dataset.split == "query":
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                }
            )
        else:
            unlabeled_pid = pid_list[-1]
            pid_list = pid_list[:-1]  # remove unlabeled pid
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                    "unlabeled_pid": int(unlabeled_pid),
                }
            )
    logger.info(f"=> {dataset.name}-{dataset.split} loaded:\n" + create_small_table(statistics))


def build_dataset(dataset_name, root, transforms, split, verbose=True):
    if dataset_name == "CUHK-SYSU":
        dataset = CUHKSYSU(root, transforms, split)
    elif dataset_name == "PRW":
        dataset = PRW(root, transforms, split)
    else:
        raise NotImplementedError(f"Unknow dataset: {dataset_name}")
    if verbose:
        print_statistics(dataset)
    return dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_train_loader(args):
    transforms = build_transforms(is_train=True)
    dataset = build_dataset(args.dataset, args.data_root, transforms, "train")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize_train,
        shuffle=True,
        num_workers=args.workers_train,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


def build_test_loader(args):
    transforms = build_transforms(is_train=False)
    gallery_set = build_dataset(args.dataset, args.data_root, transforms, "gallery")
    query_set = build_dataset(args.dataset, args.data_root, transforms, "query")
    gallery_loader = torch.utils.data.DataLoader(
        gallery_set,
        batch_size=args.batchsize_test,
        shuffle=False,
        num_workers=args.workers_test,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    query_loader = torch.utils.data.DataLoader(
        query_set,
        batch_size=args.batchsize_test,
        shuffle=False,
        num_workers=args.workers_test,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return gallery_loader, query_loader
