from torch.utils.data import DataLoader
import pickle
import torchvision.models as models
import torch

import config
from datasets.cifar10 import Cifar10


def get_resnet50():
    resnet50 = models.resnet50(pretrained=True)
    resnet50.features = torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool,
                                            resnet50.layer1,
                                            resnet50.layer2, resnet50.layer3, resnet50.layer4)
    for module in filter(lambda m: type(m) == torch.nn.BatchNorm2d, resnet50.modules()):
        module.eval()
        module.train = lambda _: None
    return resnet50


def get_embedding(model):
    model.features_pooling = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.features_dropout = torch.nn.Dropout(0.01)

    def forward(x):
        x = model.features(x)
        x = model.features_pooling(x)
        x = model.features_dropout(x)
        bs = x.size(0)
        features = x.view(bs, -1)
        return features
    model.forward = forward


def get_features_with_resnet(save_features=True, load_features=True):
    if load_features:
        with open(config.RESNET_FEATURES_PICKLE_PATH, 'rb') as f:
            features = pickle.load(f)
    else:
        ds = Cifar10(anomaly_class='airplane')
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        resnet50 = get_resnet50().cuda()
        get_embedding(resnet50)

        features = []
        nb_batches = len(dl)
        with torch.no_grad():
            for i, (im, _) in enumerate(dl):
                print(f'{i}/{nb_batches}')
                out = resnet50(im.cuda()).cpu().numpy()
                features.extend(out)

        if save_features:
            with open(config.RESNET_FEATURES_PICKLE_PATH, 'wb') as f:
                pickle.dump(features, f)
    return features
