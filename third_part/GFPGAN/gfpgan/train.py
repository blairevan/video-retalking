# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import third_part.GFPGAN.gfpgan.archs
import third_part.GFPGAN.gfpgan.data
import third_part.GFPGAN.gfpgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
