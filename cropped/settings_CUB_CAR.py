img_size = 224
prototype_shape = (1000, 64, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

data_path = "/mnt/c/chong/data/CUB_200_2011_cropped/"
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'

train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 100

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3} 

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst_trv': -0.8,
    'sep_trv': 0.08,
    'clst_spt': -0.8,
    'sep_spt': 0.48,
    'orth': 1e-3,
    'l1': 1e-4,
    'discr': 1.0,
    'close': 1.0,
}


num_train_epochs = 30
num_warm_epochs = 5

push_start = 20
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]     # 10


