
import torch
import os
import models.WideResNet as WRN


def load_settings(args):

    # Path to the teacher model
    WRN_path = os.path.join(args.teacher_path, 'WRN28-4_21.09.pt')
    assert os.path.exists(WRN_path), 'Can not find teacher model at {:}'.format(WRN_path)

    teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
    state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
    teacher.load_state_dict(state)

    return teacher