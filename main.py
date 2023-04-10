from backbone import get_network
from easydict import EasyDict

if __name__ == '__main__':
    cfg = dict(
        use_onenetwork=True,
        width_mult=1.0,
        num_verts=1000,
        input_size=256,
        task=1,
        network='resnet_jmlr',
        no_gap=False,
        use_arcface=False,
    )
    config = EasyDict(cfg)
    net = get_network(config)

    print(net)