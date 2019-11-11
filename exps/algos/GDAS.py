import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn

from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from procedures   import prepare_seed, prepare_logger
from log_utils    import AverageMeter, time_string, convert_secs2time
from datasets     import get_datasets, SearchDataset
from config_utils import load_config, dict2config, configure2str
from models       import  get_search_spaces, get_cell_based_tiny_net

# from utils        import get_model_infos, obtain_accuracy
# from models       import get_cell_based_tiny_net, get_search_spaces


def main(xargs):

  # Set CUDA attributes
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  # NOTE: cudnn.deterministic may affect run-time
  torch.backends.cudnn.deterministic = True

  # Basic Startup stuff
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  # Obtain the train_data (train + val data combined)
  # CIFAR Train Transform: [RandomHorizontalFlip(), RandomCrop(32, padding=4), ToTensor(), Normalize(mean, std)]
  # CIFAR Test Transform: [ToTensor(), Normalize(mean, std)]
  train_data, _, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1) # ignore the test data

  # Loading the train and validation splits
  if xargs.dataset == 'cifar10' or xargs.dataset == 'cifar100':
    # File path for Train-Validation Split
    split_Fpath = 'configs/cifar-split.txt'
    cifar_split = load_config(split_Fpath, None, None)
    # train: 25000, valid: 25000
    train_split, valid_split = cifar_split.train, cifar_split.valid
    logger.log('Load split file from {:}'.format(split_Fpath))
  elif xargs.dataset.startswith('ImageNet16'):
    split_Fpath = 'configs/nas-benchmark/{:}-split.txt'.format(xargs.dataset)
    imagenet16_split = load_config(split_Fpath, None, None)
    train_split, valid_split = imagenet16_split.train, imagenet16_split.valid
    logger.log('Load split file from {:}'.format(split_Fpath))
  else:
    raise ValueError('invalid dataset : {:}'.format(xargs.dataset))

  # Load GDAS specific scheduler and optimizer configs
  config_path = 'configs/algos/GDAS.config'
  config = load_config(config_path, {'class_num': class_num, 'xshape': xshape}, None)

  # Search Data Loader
  search_data = SearchDataset(xargs.dataset, train_data, train_split, valid_split)
  search_loader = torch.utils.data.DataLoader(search_data, batch_size=config.batch_size, shuffle=True,
                                              num_workers=xargs.workers, pin_memory=True)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  # Candidate operations in search space
  search_space = get_search_spaces('cell', xargs.search_space_name)
  logger.log('Candidate operations: {:}'.format(search_space))

  # GDAS model configuration
  model_config = dict2config({'name': 'GDAS', 'C': xargs.channel, 'N': xargs.num_cells,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space': search_space}, None)
  logger.log('Model Configuration: {:}'.format(model_config))

  # Get the GDAS search model
  search_model = get_cell_based_tiny_net(model_config)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("GDAS")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                      help='Choose between Cifar10/100 and ImageNet-16.')

  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')

  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--tau_min',            type=float,               help='The minimum tau for Gumbel')
  parser.add_argument('--tau_max',            type=float,               help='The maximum tau for Gumbel')

  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers '
                                                                             '(default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset '
                                                               '(tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')

  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)

  main(args)
