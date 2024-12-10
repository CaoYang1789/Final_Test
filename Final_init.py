#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


import torch
import torch.nn as nn
import numpy as np
import random
from .utils import * 
from importlib import import_module  
from nas_201_api import NASBench201API as API201
available_measures = []  
_measure_impls = {}  


def lazy_import(module_name):
    try:
        print(f"Attempting to import: {module_name}")  # ???????
        return import_module(module_name, package='measures')  # ???????
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        raise



def calc_measure(name, net, device, *args, **kwargs):


    module = lazy_import(f'.{name}')
    

    func_name_per_weight = f'compute_{name}_per_weight'
    func_name_get_arr = f'get_{name}_arr'
    func_name_special = f'compute_{name}' 
    

    measure_func = getattr(module, func_name_per_weight, None)
    if measure_func is None:
        measure_func = getattr(module, func_name_get_arr, None)
    if measure_func is None:
        measure_func = getattr(module, func_name_special, None)  
    

    if measure_func is None:
        raise ValueError(
            f"Function {func_name_per_weight}, {func_name_get_arr}, or {func_name_special} not found in module {name}"
        )
    

    return measure_func(net, *args, **kwargs)

def measure(name, bn=True, copy_net=False, force_clean=True, **impl_args):
    def make_impl(func):
        def measure_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net_orig
            ret = func(net, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc
                import torch
                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _measure_impls
        if name in _measure_impls:
            raise KeyError(f'Duplicated measure! {name}')
        available_measures.append(name)
        _measure_impls[name] = measure_impl
        return func
    return make_impl




def enum_gradient_measure(net, device, data, label, *args, **kwargs):

    def sum_arr(arr):

        return sum(torch.sum(item) for item in arr).item()

    score_list = []
    # ???? calc_measure ?????? data ?? label
    score_list.append(sum_arr(calc_measure('grad_norm', net, device, data, label, *args, **kwargs)))
    score_list.append(sum_arr(calc_measure('snip', net, device, data, label, *args, mode='param', **kwargs)))
    if kwargs.get('space', 'cv') == 'cv':
        score_list.append(sum_arr(calc_measure('grasp', net, device, data, label, *args, mode='param',**kwargs)))
    score_list.append(sum_arr(calc_measure('fisher', net, device, data, label, mode='channel',**kwargs)))
    score_list.append(calc_measure('jacob_cov', net, device, data, label, *args,**kwargs))
    score_list.append(sum_arr(calc_measure('plain', net, device, data, label, *args,mode='param', **kwargs)))
    score_list.append(sum_arr(calc_measure('synflow', net, device, data, label, *args,mode='param', **kwargs)))
    return score_list

# def get_ntk_n(dataloader, network, train_mode=False, num_batch=-1):
#     device = torch.cuda.current_device()
#     ntks = []
#     network.to(device)
#     networks=[network]
#     for network in networks:
#         if train_mode:
#             network.train()
#         else:
#             network.eval()
#     ######
#     grads = [[] for _ in range(len(networks))]
#     for i, (inputs, targets) in enumerate(dataloader):
#         if num_batch > 0 and i >= num_batch: break
#         inputs = inputs.cuda(device=device, non_blocking=True)
#         for net_idx, network in enumerate(networks):
#             network.zero_grad()
#             inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
#             logit = network(inputs_)[1]
#             if isinstance(logit, tuple):
#                 logit = logit[1]  # 201 networks: return features and logits
#             for _idx in range(len(inputs_)):
#                 logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
#                 grad = []
#                 for name, W in network.named_parameters():
#                     if 'weight' in name and W.grad is not None:
#                         grad.append(W.grad.view(-1).detach())
#                 grads[net_idx].append(torch.cat(grad, -1))
#                 network.zero_grad()
#                 torch.cuda.empty_cache()
#     ######
#     grads = [torch.stack(_grads, 0) for _grads in grads]
#     ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
#     conds = []
#     for ntk in ntks:
#         eigenvalues, _ = torch.symeig(ntk)  # ascending
#         conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
#     return conds[0]
def get_ntk_n(dataloader, network, train_mode=False, num_batch=-1):
    device = torch.cuda.current_device()
    ntks = []
    network.to(device)
    networks = [network]
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(dataloader):
        if num_batch > 0 and i >= num_batch:
            break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)[1]
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx + 1].backward(torch.ones_like(logit[_idx:_idx + 1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues = torch.linalg.eigvalsh(ntk, UPLO='L')  # ?滻? eigvalsh
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds[0]

def get_batch_jacobian( net, x, target, ):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()


def get_logdet(network, args, data, label):
    network.cuda()

    network.K = np.zeros((args.batchsize, args.batchsize))
    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
        except:
            pass
    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True
    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            #hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)
    
    jacobs, labels, y, out = get_batch_jacobian(network, data, label)
    s, logdet_val = np.linalg.slogdet(network.K)
    return logdet_val


def network_weight_gaussian_init(net):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def get_zenscore(model, resolution, batch_size, repeat=1, mixup_gamma=1e-2, fp16=False, space = 'cv'):
    info = {}
    nas_score_list = []

    device = torch.device('cuda:0')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32
    if space == 'asr':
        inputsize = resolution
    else:
        inputsize = [batch_size, 3, resolution, resolution]
    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            
            input = torch.randn(size=inputsize, device=device, dtype=dtype)
            input2 = torch.randn(size=inputsize, device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            output, logits = model(input, outpreap=True)
            mixup_output, logits = model(mixup_input, outpreap=True)
            print(output.size())
            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))


    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    scorelist=[info[keyname] for keyname in info.keys() ]
    return scorelist[0],scorelist[1],scorelist[2]


def get_grad_score(network,  data, label, loss_fn, split_data=1, device='cuda', space='cv'):
    score_list = enum_gradient_measure(network, device, data, label, loss_fn=loss_fn, split_data=split_data, space=space)
    return score_list


#def calculate_nrs(model, dataloader, data, label, resolution, batch_size, loss_fn, grad_weights,
 #                 alpha=1.0, beta=0.5, gamma=0.8, delta=0.3, repeat=3, zen_metric="avg"):
def calculate_nrs(model, data, label, resolution, batch_size, loss_fn, grad_weights,
                alpha=1.0, gamma=0.8, delta=0.3, repeat=3, zen_metric="avg"):

    # ??? Zen Score
    result = get_zenscore(model, resolution, batch_size, repeat=repeat)
    if len(result) != 3:
        raise ValueError("Unexpected return format from get_zenscore. Expected 3 values.")
    avg_nas_score, std_nas_score, avg_precision = result
    if zen_metric == "avg":
        zen_score = avg_nas_score
    elif zen_metric == "std":
        zen_score = std_nas_score
    elif zen_metric == "precision":
        zen_score = avg_precision
    else:
        raise ValueError(f"Invalid zen_metric: {zen_metric}. Choose from 'avg', 'std', or 'precision'.")
    if zen_score <= 0:
        zen_score = 1e-6

    # ??? NTK ??????
    #ntk_values = [get_ntk_n(dataloader, model) for _ in range(repeat)]
    #ntk_cond = np.mean(ntk_values)
    #if ntk_cond == 0:
    #    ntk_cond = 1e-6
    #ntk_stability = 1.0 / ntk_cond

    # ??? Log Determinant
    # 在 calculate_nrs() 函数中，创建一个新的变量来传递给 get_logdet()
    #args_dict = {'batchsize': batch_size, 'resolution': resolution}
    #logdet_k = get_logdet(model, args_dict, data, label)

    #args = {'batchsize': batch_size, 'resolution': resolution}
    #logdet_k = get_logdet(model, args, data, label)
    
    # ??? Gradient Sensitivity
    grad_scores = enum_gradient_measure(model, device='cuda', data=data, label=label, loss_fn=loss_fn, space='cv')
    # ???????????δ????????????
    if grad_weights is None:
        grad_weights = np.ones(len(grad_scores)) / len(grad_scores)  # ????????
    assert len(grad_weights) == len(grad_scores), f"Length mismatch: grad_weights({len(grad_weights)}) and grad_scores({len(grad_scores)})"
    
     # ??????????????
    std_grad_scores = np.std(grad_scores)
    if std_grad_scores == 0:
        std_grad_scores = 1e-6
    normalized_scores = (grad_scores - np.mean(grad_scores)) / std_grad_scores

    gradient_sensitivity = np.sum(grad_weights * normalized_scores)

    # ?????? NRS
    #log_zen = torch.log(torch.tensor(zen_score))
    #nrs = (
    #    alpha * float(log_zen) +
    #    beta * float(ntk_stability) +
    #    gamma * float(logdet_k) -
    #    delta * float(gradient_sensitivity)
    #)

      # ?????? NRS
    #log_zen = torch.log(torch.tensor(zen_score))
    #nrs = (
    #    alpha * float(log_zen) +
    #    gamma * float(logdet_k) -
    #    delta * float(gradient_sensitivity)
    #)

    log_zen = torch.log(torch.tensor(zen_score))
    nrs = (
        alpha * float(log_zen) +
        delta * float(gradient_sensitivity)
    )


    
    return nrs

def get_new_score(latency_matrics,params, macs, test_acc, args,netid,api):
    from main import getmisc
    from main import search201

    imgsize, ce_loss, trainloader, testloader = getmisc(args)
    top_k_list = []  # 存储符合条件的网络

    
    for i, batch in enumerate(trainloader):
        data,label = batch[0],batch[1]
        data,label=data.cuda(),label.cuda()
        break            

    netid=netid
    network, metrics, adjacency, operations, latency_matric= search201(api, netid, args.dataset)
    network.cuda()
    # zen_score=get_zenscore(network, imgsize, args.batchsize)
    # print(f"Log-determinant (logdet_k): {zen_score}")
    #得到参数
    zen_score = get_zenscore(network, imgsize, args.batchsize)
    grad_sensitivity=get_grad_score(network,  data, label, ce_loss, split_data=1, device='cuda')
    # print(f"zen_score : {zen_score }, type: {type(zen_score )}")
    # (143.10256958007812, 0.0, 0.0)
    # print(f"grad_sensitivity: {grad_sensitivity}, type: {type(grad_sensitivity)}")
    # [5.596492290496826, 140.700927734375, -0.724810004234314, 0.11414037644863129, -301.7180719350535, 5.285810470581055, 254.1832782289741]
#############################以上为参数部分#################################################################

    
    # 设置权重
    zen_value = zen_score[0]  # 使用 zen_score 的第一个值
    grad_mean = np.mean(grad_sensitivity)  # 计算梯度的均值
    grad_std = np.std(grad_sensitivity)  # 计算梯度的标准差
    grad_abs_mean = np.abs(grad_mean)  # 均值的绝对值

    # 设置权重
    w1, w2, w3, w4, w5 = 1, -0.5, -0.4,  -2, 1
                             #zen  grd    grd       energy acc           

    # 评分公式
    score = (
        w1 * zen_value +              # zen_score 的权重
        w2 * grad_std +                # 梯度标准差的权重（越小越好）
        w3 * grad_abs_mean +     # 梯度均值绝对值的权重（越接近 0 越好）
        w4 *latency_matrics['edgegpu']['energy'] + #能耗越低越好
        w5* test_acc
    )

    #score = zen_score[0] + grad_sensitivity[0]+ test_acc + np.log10(params) + np.log10(macs)
    return score

