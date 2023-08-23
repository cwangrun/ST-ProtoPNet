import os
import shutil
import random
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse
import re
import time
from util.helpers import makedir
import push_trivial, push_support, model, train_and_test as tnt
from util import save
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
import settings_CUB_CAR


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid',type=str, default='0') 
# parser.add_argument('-arch',type=str, default='vgg16')
parser.add_argument('-arch',type=str, default='resnet34')

parser.add_argument('-dataset',type=str,default="CUB")
parser.add_argument('-rand_seed', type=int, default=0)
args = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])



#setting parameter
experiment_run = settings_CUB_CAR.experiment_run
base_architecture = args.arch
dataset_name = args.dataset

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

model_dir ='saved_models/{}/'.format(datestr()) + base_architecture + '/' + experiment_run + '/'

if os.path.exists(model_dir) is True:
    shutil.rmtree(model_dir)
makedir(model_dir)


shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB_CAR.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models', base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'push_trivial.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'push_support.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), './util/helpers.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


# random.seed(args.rand_seed)  # python random seed
# np.random.seed(args.rand_seed)  # numpy random seed
# torch.manual_seed(args.rand_seed)  # torch random seed
# torch.cuda.manual_seed(args.rand_seed)
# torch.backends.cudnn.deterministic = True


#model param
num_classes = settings_CUB_CAR.num_classes
img_size = settings_CUB_CAR.img_size
add_on_layers_type = settings_CUB_CAR.add_on_layers_type
prototype_shape = settings_CUB_CAR.prototype_shape
prototype_activation_function = settings_CUB_CAR.prototype_activation_function
#datasets
train_dir = settings_CUB_CAR.train_dir
test_dir = settings_CUB_CAR.test_dir
train_push_dir = settings_CUB_CAR.train_push_dir
train_batch_size = settings_CUB_CAR.train_batch_size
test_batch_size = settings_CUB_CAR.test_batch_size
train_push_batch_size = settings_CUB_CAR.train_push_batch_size
# weighting of different training losses
coefs = settings_CUB_CAR.coefs
# number of training epochs, number of warm epochs, push start epoch, push epochs
num_train_epochs = settings_CUB_CAR.num_train_epochs
num_warm_epochs = settings_CUB_CAR.num_warm_epochs
push_start = settings_CUB_CAR.push_start
push_epochs = settings_CUB_CAR.push_epochs


log(train_dir)

normalize = transforms.Normalize(mean=mean, std=std)

# all datasets
# train set
num_workers = 8  # 16
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

log("backbone architecture:{}".format(base_architecture))
log("prototype shape:{}".format(prototype_shape))
# construct the model
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)



weight_decay_factor = 1.0
if ('resnet' in base_architecture) or ('vgg' in base_architecture):   # 0.5 (VGG, ResNet), 1.0 (DenseNet)
    weight_decay_factor = 0.5


class_specific = True

# define optimizer
from settings_CUB_CAR import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[
 {'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': weight_decay_factor*1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers_trivial.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay_factor*1e-3},
 {'params': ppnet.add_on_layers_support.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay_factor*1e-3},
 {'params': ppnet.prototype_vectors_trivial, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.prototype_vectors_support, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=1, gamma=0.2)


from settings_CUB_CAR import warm_optimizer_lrs
warm_optimizer_specs = \
[
 {'params': ppnet.add_on_layers_trivial.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay_factor*1e-3},
 {'params': ppnet.add_on_layers_support.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay_factor*1e-3},
 {'params': ppnet.prototype_vectors_trivial, 'lr': warm_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.prototype_vectors_support, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings_CUB_CAR import last_layer_optimizer_lr
last_layer_optimizer_specs = \
[
 {'params': ppnet.last_layer_trivial.parameters(), 'lr': last_layer_optimizer_lr},
 {'params': ppnet.last_layer_support.parameters(), 'lr': last_layer_optimizer_lr},
]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

#best acc
best_acc = 0
best_epoch = 0
best_time = 0

# train the model
log('start training')

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    #stage 1: Training of CNN backbone and prototypes
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                                     class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        if epoch in [7, 9, 11, 13]:
            joint_lr_scheduler.step()
        _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                                     class_specific=class_specific, coefs=coefs, log=log)

    #test
    accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.60, log=log)

    #stage2: prototype projection
    if epoch >= push_start and epoch in push_epochs:
        push_trivial.push_prototypes(
            train_push_loader,
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir + 'trivial',  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        push_support.push_prototypes(
            train_push_loader,
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir + 'support',  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.60, log=log)
    #stage3:  Training of FC layers
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(10):
                log('iteration: \t{0}'.format(i))
                _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                             class_specific=class_specific, coefs=coefs, log=log)

                accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.60, log=log)
   
logclose()

