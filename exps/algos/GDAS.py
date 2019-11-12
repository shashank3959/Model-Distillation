import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn

from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from procedures   import prepare_seed, prepare_logger, get_optim_scheduler, save_checkpoint, copy_checkpoint
from log_utils    import AverageMeter, time_string, convert_secs2time
from datasets     import get_datasets, SearchDataset
from config_utils import load_config, dict2config, configure2str
from models       import  get_search_spaces, get_cell_based_tiny_net
from utils        import get_model_infos, obtain_accuracy


def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()

  network.train()
  end = time.time()

  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)

    # Alternative Optimization: 1 iteration of weight update, 1 iteration of architecture weight update
    # Important Design Choice: Add Distillation loss to Weight Update OR Architecture Update OR both?
    # update the weights
    w_optimizer.zero_grad()
    _, logits = network(base_inputs)
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(), base_inputs.size(0))
    base_top1.update(base_prec1.item(), base_inputs.size(0))
    base_top5.update(base_prec5.item(), base_inputs.size(0))

    # update the architecture-weight
    a_optimizer.zero_grad()
    _, logits = network(arch_inputs)
    arch_loss = criterion(logits, arch_targets)
    arch_loss.backward()
    a_optimizer.step()
    # record
    arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
    arch_losses.update(arch_loss.item(), arch_inputs.size(0))
    arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
    arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
        batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
        loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
        loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
    print('At step:', step)
    break

  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


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

  # Get the GDAS search model. This setups the general architecture with all candidate operations
  search_model = get_cell_based_tiny_net(model_config)

  # Get optimizer, scheduler and loss criterion from config
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)

  # Manually define optimizer for architecture updates. alphas are init. to (1e-3)*randn each
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999),
                                 weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))

  # Print Model Information
  flop, param = get_model_infos(search_model, xshape)
  # logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search_space : {:}'.format(search_space))

  # Enable multi-GPU
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  # Automatically resume from previous checkpoint
  # You must use the same seed value as the last checkpoint to continue
  if last_info.exists():
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint = torch.load(last_info['last_checkpoint'])
    genotypes = checkpoint['genotypes']
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict(checkpoint['search_model'])
    w_scheduler.load_state_dict(checkpoint['w_scheduler'])
    w_optimizer.load_state_dict(checkpoint['w_optimizer'])
    a_optimizer.load_state_dict(checkpoint['a_optimizer'])
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> did not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {}

  # Start Training
  start_time, epoch_time, total_epoch = time.time(), AverageMeter(), config.epochs + config.warmup
  print(start_time, epoch_time, total_epoch)
  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)

    # Time to completion
    need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)

    # Update temperature of logits
    search_model.set_tau(xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1))
    logger.log('\n[Search the {:}-th epoch] {:}, tau={:}, LR={:}'.format(epoch_str, need_time, search_model.get_tau(),
                                                                         min(w_scheduler.get_lr())))

    search_w_loss, search_w_top1, search_w_top5, valid_a_loss, valid_a_top1, valid_a_top5 \
      = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str,
                    xargs.print_freq, logger)

    logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, search_w_loss,
                                                                                              search_w_top1,
                                                                                              search_w_top5))
    logger.log('[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss,
                                                                                              valid_a_top1,
                                                                                              valid_a_top5 ))
    # check the best accuracy at end of epoch
    valid_accuracies[epoch] = valid_a_top1
    if valid_a_top1 > valid_accuracies['best']:
      # Save the best genotype so far (The one that gives best validation accuracy so far)
      valid_accuracies['best'] = valid_a_top1
      genotypes['best'] = search_model.genotype()
      find_best = True
    else:
      find_best = False

    # Save the genotype at end of this epoch
    genotypes[epoch] = search_model.genotype()
    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))

    # save checkpoint
    save_path = save_checkpoint({'epoch': epoch + 1,
                                 'args': deepcopy(xargs),
                                 'search_model': search_model.state_dict(),
                                 'w_optimizer': w_optimizer.state_dict(),
                                 'a_optimizer': a_optimizer.state_dict(),
                                 'w_scheduler': w_scheduler.state_dict(),
                                 'genotypes': genotypes,
                                 'valid_accuracies': valid_accuracies},
                                model_base_path, logger)

    last_info = save_checkpoint({ 'epoch': epoch + 1,
                                  'args': deepcopy(args),
                                  'last_checkpoint': save_path,
                                }, logger.path('info'), logger)



    break


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
