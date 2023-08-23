import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, \
    resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, \
    densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, \
    vgg16_bn_features, \
    vgg19_features, vgg19_bn_features

from util.receptive_field import compute_proto_layer_rf_info_v2

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}


class STProtoPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True, 
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(STProtoPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function  # log

        assert (self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features  #

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers_trivial = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Tanh(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Tanh()
            )
            self.add_on_layers_support = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Tanh(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Tanh()
            )

        self.prototype_vectors_trivial = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.prototype_vectors_support = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.last_layer_trivial = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        self.last_layer_support = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):

        x = self.features(x)
        x_trivial = self.add_on_layers_trivial(x)
        x_support = self.add_on_layers_support(x)

        return x_trivial, x_support

    def _cosine_convolution(self, prototypes, x):

        x = F.normalize(x, p=2, dim=1)
        prototype_vectors = F.normalize(prototypes, p=2, dim=1)
        similarity = F.conv2d(input=x, weight=prototype_vectors)

        return similarity

    def _project2prototype(self, prototypes, x):

        prototype_vectors = F.normalize(prototypes, p=2, dim=1)
        similarity = F.conv2d(input=x, weight=prototype_vectors)

        return similarity

    def prototype_similarities(self, x):

        conv_features_trivial, conv_features_support = self.conv_features(x)
        cosine_similarities_trivial = self._cosine_convolution(self.prototype_vectors_trivial, conv_features_trivial)
        cosine_similarities_support = self._cosine_convolution(self.prototype_vectors_support, conv_features_support)
        project_similarities_trivial = self._project2prototype(self.prototype_vectors_trivial, conv_features_trivial)
        project_similarities_support = self._project2prototype(self.prototype_vectors_support, conv_features_support)

        return project_similarities_trivial, project_similarities_support, cosine_similarities_trivial, cosine_similarities_support

    def distance_2_similarity(self, distances):

        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise Exception('other activation function NOT implemented')

    def distance_2_similarity_linear(self, distances):
        return (self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]) ** 2 - distances

    def global_min_pooling(self, input):

        min_output = -F.max_pool2d(-input, kernel_size=(input.size()[2], input.size()[3]))
        min_output = min_output.view(-1, self.num_prototypes)

        return min_output

    def global_max_pooling(self, input):

        max_output = F.max_pool2d(input, kernel_size=(input.size()[2], input.size()[3]))
        max_output = max_output.view(-1, self.num_prototypes)

        return max_output

    def forward(self, x):

        project_similarities_trivial, project_similarities_support, cosine_similarities_trivial, cosine_similarities_support = self.prototype_similarities(x)

        prototype_activations_trivial = self.global_max_pooling(project_similarities_trivial)
        prototype_activations_support = self.global_max_pooling(project_similarities_support)

        logits_trivial = self.last_layer_trivial(prototype_activations_trivial)
        logits_support = self.last_layer_support(prototype_activations_support)

        cosine_similarities_trivial_max = self.global_max_pooling(cosine_similarities_trivial)
        cosine_similarities_support_max = self.global_max_pooling(cosine_similarities_support)

        return (logits_trivial, logits_support), (cosine_similarities_trivial_max, cosine_similarities_support_max), \
               (project_similarities_trivial, project_similarities_support)

    def push_forward_trivial(self, x):

        conv_features_trivial, _ = self.conv_features(x)  # [batchsize, 64, 7, 7]

        similarities = self._project2prototype(self.prototype_vectors_trivial, conv_features_trivial)
        distances = - similarities

        conv_output = F.normalize(conv_features_trivial, p=2, dim=1)

        return conv_output, distances

    def push_forward_support(self, x):

        _, conv_features_support = self.conv_features(x)  # [batchsize, 64, 7, 7]

        similarities = self._project2prototype(self.prototype_vectors_support, conv_features_support)
        distances = - similarities

        conv_output = F.normalize(conv_features_support, p=2, dim=1)

        return conv_output, distances

    def set_last_layer_incorrect_connection(self, incorrect_strength):

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer_trivial.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
        self.last_layer_support.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers_trivial.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.add_on_layers_support.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)   # 0.0


def construct_STProtoPNet(base_architecture, pretrained=True, img_size=224,
                          prototype_shape=(2000, 128, 1, 1), num_classes=200,
                          prototype_activation_function='log',
                          add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return STProtoPNet(features=features,
                       img_size=img_size,
                       prototype_shape=prototype_shape,
                       proto_layer_rf_info=proto_layer_rf_info,
                       num_classes=num_classes,
                       init_weights=True,
                       prototype_activation_function=prototype_activation_function,
                       add_on_layers_type=add_on_layers_type)

