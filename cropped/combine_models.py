import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import model
import time
from util.preprocess import mean, std
import settings_CUB_CAR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PPNet_ensemble(nn.Module):

    def __init__(self, ppnets):
        super(PPNet_ensemble, self).__init__()
        self.ppnets = ppnets  # a list of ppnets

    def forward(self, x):
        logits, max_similarities, _ = self.ppnets[0](x)
        logits = logits[0] + logits[1]
        max_similarities_trivial = [max_similarities[0]]
        max_similarities_support = [max_similarities[1]]

        for i in range(1, len(self.ppnets)):
            logits_i, max_similarities_trivial_i, max_similarities_support_i = self.ppnets[i](x)
            logits_i = logits_i[0] + logits_i[1]
            logits.add_(logits_i)
            max_similarities_trivial.append(max_similarities_trivial_i)
            max_similarities_support.append(max_similarities_support_i)

        return logits, max_similarities_trivial, max_similarities_support


ppnets = []
###################################################################################################
# setting parameter and construct the model
base_architecture = 'vgg19'
img_size = settings_CUB_CAR.img_size
prototype_shape = settings_CUB_CAR.prototype_shape
num_classes = settings_CUB_CAR.num_classes
prototype_activation_function = settings_CUB_CAR.prototype_activation_function
add_on_layers_type = settings_CUB_CAR.add_on_layers_type
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
ppnet.load_state_dict(torch.load('./ensemble-models/vgg19-18nopush0.8350.pth'))
ppnet = ppnet.cuda()
ppnet.eval()
ppnets.append(ppnet)
# del ppnet
###################################################################################################
# setting parameter and construct the model
base_architecture = 'densenet121'
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
ppnet.load_state_dict(torch.load('./ensemble-models/dense121-15nopush0.8559.pth'))
ppnet = ppnet.cuda()
ppnet.eval()
ppnets.append(ppnet)
# del ppnet
###################################################################################################
# setting parameter and construct the model
base_architecture = 'densenet161'
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
ppnet.load_state_dict(torch.load('./ensemble-models/dense161-9nopush0.8571.pth')) 
ppnet = ppnet.cuda()
ppnet.eval()
ppnets.append(ppnet)
# del ppnet
###################################################################################################
# setting parameter and construct the model
base_architecture = 'resnet34'
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
ppnet.load_state_dict(torch.load('./ensemble-models/res34-16nopush0.8357.pth'))
ppnet = ppnet.cuda()
ppnet.eval()
ppnets.append(ppnet)
# del ppnet
###################################################################################################
# setting parameter and construct the model
base_architecture = 'resnet152'
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
ppnet.load_state_dict(torch.load('./ensemble-models/res152-11nopush0.8440.pth'))
ppnet = ppnet.cuda()
ppnet.eval()
ppnets.append(ppnet)
# del ppnet
###################################################################################################


ppnet_ensemble = PPNet_ensemble(ppnets)
ppnet_ensemble = ppnet_ensemble.cuda()
ppnet_ensemble_multi = torch.nn.DataParallel(ppnet_ensemble)

img_size = ppnets[0].img_size

# ppnet_multi = torch.nn.DataParallel(ppnet)
# img_size = ppnet_multi.module.img_size
# prototype_shape = ppnet.prototype_shape
# max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

# load test data
from settings_CUB_CAR import test_dir

test_batch_size = 100

normalize = transforms.Normalize(mean=mean, std=std)

test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=20, pin_memory=False)
print('test set size: {0}'.format(len(test_loader.dataset)))


for ppnet in ppnet_ensemble_multi.module.ppnets:
    print(ppnet)

class_specific = True


# only supports last layer adjustment
def _train_or_test_ppnet_ensemble(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                                  coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, _, _ = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            l1 = torch.tensor(0.0).cuda()

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['i_crs_ent'] * cross_entropy + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['i_crs_ent'] * cross_entropy + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted

    end = time.time()

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    return n_correct / n_examples


def train_ensemble(model, dataloader, optimizer, class_specific=True, coefs=None, log=print):
    assert (optimizer is not None)

    log('\ttrain')
    model.train()
    return _train_or_test_ppnet_ensemble(model=model, dataloader=dataloader, optimizer=optimizer,
                                         class_specific=class_specific, coefs=coefs, log=log)


def test_ensemble(model, dataloader, class_specific=True, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test_ppnet_ensemble(model=model, dataloader=dataloader, optimizer=None,
                                         class_specific=class_specific, log=log)


#check test accuracy
accu = test_ensemble(model=ppnet_ensemble_multi, dataloader=test_loader, class_specific=class_specific, log=print)


