#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import sys
from loguru import logger

# activate rknn hack
if len(sys.argv)>=3 and '--rknpu' in sys.argv:
    _index = sys.argv.index('--rknpu')
    if sys.argv[_index+1].upper() in ['RK1808', 'RV1109', 'RV1126','RK3399PRO']:
        os.environ['RKNN_model_hack'] = 'npu_1'
    elif sys.argv[_index+1].upper() in ['RK3566', 'RK3568', 'RK3588','RK3588S','RV1106','RV1103']:
        os.environ['RKNN_model_hack'] = 'npu_2'
    else:
        assert False,"{} not recognized".format(sys.argv[_index+1])

import torch

from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX torchscript deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.torchscript.pt", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument('--rknpu', default=None, help='RKNN npu platform')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.head.decode_in_inference = False

    if os.getenv('RKNN_model_hack', '0') in ['npu_1', 'npu_2']:
        from yolox.models.network_blocks import Focus, Focus_conv
        for k,m in model.named_modules():
            if isinstance(m, Focus) and hasattr(m, 'sf'):
                m.sf = Focus_conv()

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    mod = torch.jit.trace(model, dummy_input)
    mod.save(args.output_name)
    logger.info("generated torchscript model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
