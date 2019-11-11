import torch
from os import path as osp

__all__ = ['change_key', 'get_cell_based_tiny_net', 'get_search_spaces', 'get_cifar_models', 'get_imagenet_models', \
           'obtain_model', 'obtain_search_model', 'load_net_from_checkpoint', \
           'CellStructure', 'CellArchitectures'
           ]

# useful modules
from config_utils import dict2config


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
def get_search_spaces(xtype, name):
  if xtype == 'cell':
    from .cell_operations import SearchSpaceNames
    return SearchSpaceNames[name]
  else:
    raise ValueError('invalid search-space type is {:}'.format(xtype))


# Cell-based NAS Models
def get_cell_based_tiny_net(config):
  group_names = ['DARTS-V1', 'DARTS-V2', 'GDAS']
  from .cell_searchs import nas_super_nets
  if config.name in group_names:
    return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space)
  elif config.name == 'infer.tiny':
    from .cell_infers import TinyNetwork
    return TinyNetwork(config.C, config.N, config.genotype, config.num_classes)
  else:
    raise ValueError('invalid network name : {:}'.format(config.name))
