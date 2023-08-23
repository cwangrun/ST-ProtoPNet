import time
import torch
import torch.nn.functional as F
from util.helpers import list_of_distances
from util.helpers import list_of_distances_3d
from util.helpers import list_of_distances_3d_dot
from util.helpers import list_of_similarities_3d_dot
import numpy as np


def _training(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True, coefs=None, log=print):
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
    total_cluster_cost = 0
    total_separation_cost = 0
    total_orth_cost = 0

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:

            output, max_similarities, similarity_maps = model(input)

            # compute loss
            cross_entropy_trivial = F.cross_entropy(output[0], target)
            cross_entropy_support = F.cross_entropy(output[1], target)
            cross_entropy = 0.5 * (cross_entropy_trivial + cross_entropy_support)

            if class_specific:

                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()

                cluster_real_trivial, _ = torch.max(max_similarities[0] * prototypes_of_correct_class, dim=1)
                cluster_cost_trivial = torch.mean(cluster_real_trivial)
                cluster_real_support, _ = torch.max(max_similarities[1] * prototypes_of_correct_class, dim=1)
                cluster_cost_support = torch.mean(cluster_real_support)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                separation_real_trivial, _ = torch.max(max_similarities[0] * prototypes_of_wrong_class, dim=1)
                separation_cost_trivial = torch.mean(separation_real_trivial)
                separation_real_support, _ = torch.max(max_similarities[1] * prototypes_of_wrong_class, dim=1)
                separation_cost_support = torch.mean(separation_real_support)

                # calculate closeness and discrimination losses
                #######################################################################
                num_proto_per_class = model.module.num_prototypes_per_class
                I_operator_c = 1 - torch.eye(model.module.num_classes, model.module.num_classes).cuda()  # [200, 200]

                prototypes_support = model.module.prototype_vectors_support.squeeze()
                prototypes_matrix_support = prototypes_support.reshape(model.module.num_classes, num_proto_per_class, model.module.prototype_shape[1])  # [200, 5, 64]
                simi_dot_support = list_of_similarities_3d_dot(prototypes_matrix_support, prototypes_matrix_support)  # [200, 5, 200, 5]
                simi_dot_support_min = torch.min(simi_dot_support.permute(0, 2, 1, 3).reshape(model.module.num_classes, model.module.num_classes, -1), dim=-1)[0]  # [200, 200]
                closeness_cost = (- simi_dot_support_min * I_operator_c).sum() / I_operator_c.sum()

                prototypes_trivial = model.module.prototype_vectors_trivial.squeeze()
                prototypes_matrix_trivial = prototypes_trivial.reshape(model.module.num_classes, num_proto_per_class, model.module.prototype_shape[1])  # [200, 5, 64]
                simi_dot_trivial = list_of_similarities_3d_dot(prototypes_matrix_trivial, prototypes_matrix_trivial)  # [200, 5, 200, 5]
                simi_dot_trivial_max = torch.max(simi_dot_trivial.permute(0, 2, 1, 3).reshape(model.module.num_classes, model.module.num_classes, -1), dim=-1)[0]  # [200, 200]
                discrimination_cost = (simi_dot_trivial_max * I_operator_c).sum() / I_operator_c.sum()
                #######################################################################

                # calculate orthogonality loss
                #######################################################################
                I_operator_p = torch.eye(num_proto_per_class, num_proto_per_class).cuda()  # [5, 5]

                prototypes_matrix_support_T = torch.transpose(prototypes_matrix_support, 1, 2)   # [200, 64, 5]
                orth_dot_support = torch.matmul(prototypes_matrix_support, prototypes_matrix_support_T)   # [200, 5, 64] * [200, 64, 5] -> [200, 5, 5]
                difference_support = orth_dot_support - I_operator_p   # [200, 5, 5] - [5, 5]-> [200, 5, 5]
                orth_cost_support = torch.sum(torch.norm(difference_support, p=1, dim=[1, 2]))

                prototypes_matrix_trivial_T = torch.transpose(prototypes_matrix_trivial, 1, 2)   # [200, 64, 5]
                orth_dot_trivial = torch.matmul(prototypes_matrix_trivial, prototypes_matrix_trivial_T)   # [200, 5, 64] * [200, 64, 5] -> [200, 5, 5]
                difference_trivial = orth_dot_trivial - I_operator_p   # [200, 5, 5] - [5, 5]-> [200, 5, 5]
                orth_cost_trivial = torch.sum(torch.norm(difference_trivial, p=1, dim=[1, 2]))
                #######################################################################

                del prototypes_matrix_support
                del prototypes_matrix_support_T
                del orth_dot_support
                del difference_support
                del prototypes_matrix_trivial
                del prototypes_matrix_trivial_T
                del orth_dot_trivial
                del difference_trivial


                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1_trivial = (model.module.last_layer_trivial.weight * l1_mask).norm(p=1)
                    l1_support = (model.module.last_layer_support.weight * l1_mask).norm(p=1)
                else:
                    l1_trivial = model.module.last_layer_trivial.weight.norm(p=1)
                    l1_support = model.module.last_layer_support.weight.norm(p=1)
            else:
                cluster_cost_trivial = torch.Tensor([0]).cuda()
                separation_cost_trivial = torch.Tensor([0]).cuda()
                orth_cost_trivial = torch.Tensor([0]).cuda()
                cluster_cost_support = torch.Tensor([0]).cuda()
                separation_cost_support = torch.Tensor([0]).cuda()
                orth_cost_support = torch.Tensor([0]).cuda()
                l1_trivial = torch.Tensor([0]).cuda()
                l1_support = torch.Tensor([0]).cuda()

            # summed logits
            _, predicted = torch.max(output[0].data + output[1].data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += 0.5 * (cluster_cost_trivial + cluster_cost_support).item()
            total_separation_cost += 0.5 * (separation_cost_trivial + separation_cost_support).item()
            total_orth_cost += 0.5 * (orth_cost_trivial + orth_cost_support).item()


        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    if i % 2 == 0:
                        loss = (
                              coefs['crs_ent'] * cross_entropy_trivial
                              + coefs['clst_trv'] * cluster_cost_trivial
                              + coefs['sep_trv'] * separation_cost_trivial
                              + coefs['orth'] * orth_cost_trivial
                              + coefs['discr'] * discrimination_cost
                              + coefs['l1'] * l1_trivial
                            )
                    else:
                        loss = (
                              coefs['crs_ent'] * cross_entropy_support
                              + coefs['clst_spt'] * cluster_cost_support
                              + coefs['sep_spt'] * separation_cost_support
                              + coefs['orth'] * orth_cost_support
                              + coefs['close'] * closeness_cost
                              + coefs['l1'] * l1_support
                            )
                else:
                    loss = cross_entropy


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # normalize prototype vectors
            model.module.prototype_vectors_trivial.data = F.normalize(model.module.prototype_vectors_trivial, p=2, dim=1).data
            model.module.prototype_vectors_support.data = F.normalize(model.module.prototype_vectors_support, p=2, dim=1).data


            if i % 20 == 0:
                print(
                    '{} {} \tLoss: {:.4f} \tCE: {:.4f} '
                    '\tL_clust_trivial: {:.4f} \tL_clust_support: {:.4f} '
                    '\tL_sep_trivial: {:.4f} \tL_sep_support: {:.4f} '
                    '\tL_orth_trivial: {:.4f} \tL_orth_support: {:.4f} '
                    '\tL_discr: {:.4f} \tL_close: {:.4f} \t '
                    '\tAcc: {:.4f}'.format(
                        i, len(dataloader), loss.item(), cross_entropy.item(),
                        cluster_cost_trivial.item(), cluster_cost_support.item(),
                        separation_cost_trivial.item(), separation_cost_support.item(),
                        orth_cost_trivial.item(), orth_cost_support.item(),
                        discrimination_cost.item(), closeness_cost.item(),
                        n_correct / (n_examples + 0.000001) * 100))

        del input
        del target
        del output
        del predicted
        del max_similarities
        del similarity_maps


    end = time.time()

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\torth: \t{0}'.format(total_orth_cost / n_batches))

    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1_trivial: \t\t{0}'.format(model.module.last_layer_trivial.weight.norm(p=1).item()))
    log('\tl1_support: \t\t{0}'.format(model.module.last_layer_support.weight.norm(p=1).item()))

    results_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'cluster_loss': total_cluster_cost / n_batches,
                    'separation_loss': total_separation_cost / n_batches,
                    'orth_loss': total_orth_cost / n_batches,
                    'l1_trivial': model.module.last_layer_trivial.weight.norm(p=1).item(),
                    'l1_support': model.module.last_layer_support.weight.norm(p=1).item(),
                    'accu': n_correct / n_examples
                    }
    return n_correct / n_examples, results_loss


def _testing(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True, coefs=None, log=print):
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

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:

            output, max_similarities, _ = model(input)
            # compute loss
            cross_entropy_trivial = F.cross_entropy(output[0], target)
            cross_entropy_support = F.cross_entropy(output[1], target)
            cross_entropy = 0.5 * (cross_entropy_trivial + cross_entropy_support)

            # summed logits
            _, predicted = torch.max(output[0].data + output[1].data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()

        del input
        del target
        del output
        del predicted
        del max_similarities


    end = time.time()

    log('\ttime: \t{0}'.format(end - start))
    log('\ttest cross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\ttest accu: \t\t{0}%'.format(n_correct / n_examples * 100))

    results_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'l1 trivial': model.module.last_layer_trivial.weight.norm(p=1).item(),
                    'l1 support': model.module.last_layer_support.weight.norm(p=1).item(),
                    'accu': n_correct / n_examples
                    }
    return n_correct / n_examples, results_loss


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert (optimizer is not None)
    log('\ttrain')
    model.train()
    return _training(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _testing(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_trivial.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_support.parameters():
        p.requires_grad = False
    model.module.prototype_vectors_trivial.requires_grad = False
    model.module.prototype_vectors_support.requires_grad = False
    for p in model.module.last_layer_trivial.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_support.parameters():
        p.requires_grad = True
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_trivial.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers_support.parameters():
        p.requires_grad = True
    model.module.prototype_vectors_trivial.requires_grad = True
    model.module.prototype_vectors_support.requires_grad = True
    for p in model.module.last_layer_trivial.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_support.parameters():
        p.requires_grad = False
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers_trivial.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers_support.parameters():
        p.requires_grad = True
    model.module.prototype_vectors_trivial.requires_grad = True
    model.module.prototype_vectors_support.requires_grad = True
    for p in model.module.last_layer_trivial.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_support.parameters():
        p.requires_grad = True
    
    log('\tjoint')
