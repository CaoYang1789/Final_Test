from typing import Dict
import os
import sys

# 添加 HW-NAS-Bench 的路径
hw_nas_bench_path = "/home/test0/HW-NAS-Bench"  # 请根据实际路径修改
if hw_nas_bench_path not in sys.path:
    sys.path.append(hw_nas_bench_path)

# 确保其他模块正常导入
from nas_201_api import NASBench201API as API201
import models
from models import NB101Network
import datasets
from nats_bench import create as create_nats
import argparse
import torch
import torch.nn as nn
import random
import numpy as np
from nasbench import api as api101
from ptflops import get_model_complexity_info
from measures import get_grad_score, get_ntk_n, get_batch_jacobian, get_logdet, get_zenscore, calculate_nrs, get_new_score
import json

# 导入 HWNASBenchAPI
from hw_nas_bench_api import HWNASBenchAPI as HWAPI

parser = argparse.ArgumentParser(description='ZS-NAS')
parser.add_argument('--searchspace', metavar='ss', type=str, choices=['201'],
                    help='define the target search space of benchmark')
parser.add_argument('--dataset', metavar='ds', type=str, choices=['cifar10','cifar100','ImageNet16-120','imagenet-1k', 'cifar10-valid'],
                    help='select the dataset')
parser.add_argument('--data_path', type=str, default='/home/test0/dataset/',
                    help='the path where you store the dataset')
parser.add_argument('--cutout', type=int, default=0,
                    help='use cutout or not on input data')
parser.add_argument('--batchsize', type=int, default=256,
                    help='batch size for each input batch (default: 256)')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of threads for data pipelining (default: 8)')
parser.add_argument('--metric', type=str, choices=['basic', 'lp','nrs'],
                    help='define the zero-shot proxy for evaluation')
parser.add_argument('--startnetid', type=int, default=0,
                    help='the index of the first network to be evaluated in the search space. currently only works for nb101')
parser.add_argument('--manualSeed', type=int, default=0,
                    help='random seed')
parser.add_argument('--testName', type=str, default=None, choices=["basicGroup.txt","lpGroup.txt"],
                    help='The group used to test')
args = parser.parse_args()

def get_hardware_metrics(api, netid, dataset):
    """
    从 HW-NAS-Bench 获取指定网络架构在目标数据集上的所有硬件性能指标。

    参数:
        api: HW-NAS-Bench 的 API 实例。
        netid: 网络的索引。
        dataset: 数据集名称（如 'cifar10'）。

    返回:
        一个嵌套字典，格式为:
        {
            'edgegpu': {'latency': 5.8074, 'energy': 24.2266},
            'raspi4': {'latency': 10.482, 'energy': None},
            ...
        }
    """
    # 查询原始数据
    metrics = api.query_by_index(netid, dataset)

    # 初始化结果字典
    result = {}

    # 遍历原始数据
    for k, v in metrics.items():
        if "_latency" in k or "_energy" in k:
            metric_type = "latency" if "_latency" in k else "energy"
            hardware = k.replace(f"_{metric_type}", "")  # 提取硬件名称
            if hardware not in result:
                result[hardware] = {"latency": None, "energy": None}  # 初始化默认值为 None
            result[hardware][metric_type] = round(v, 4)  # 四舍五入到 4 位小数
    # print("EdgeGPU Latency:", formatted_metrics['edgegpu']['latency'])  # 输出: 5.8074
    # print("EdgeGPU Energy:", formatted_metrics['edgegpu']['energy'])  # 输出: 24.2266

    # 访问 raspi4 的 energy（不存在）
    # print("Raspi4 Energy:", formatted_metrics['raspi4']['energy'])  # 输出: None
    return result



def getmisc(args):
    manualSeed=args.manualSeed
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.dataset == "cifar10":
        root = args.data_path
        imgsize=32
    elif args.dataset == "cifar100":
        root = args.data_path
        imgsize=32
    elif args.dataset.startswith("imagenet-1k"):
        root = args.data_path+'ILSVRC/Data/CLS-LOC'
        imgsize=224
    elif args.dataset.startswith("ImageNet16"):
        root = args.data_path+'img16/ImageNet16/'
        imgsize=16
    

    #根据用户指定的数据集（如 CIFAR-10），加载训练和测试数据。
    train_data, test_data, xshape, class_num = datasets.get_datasets(args.dataset, root, args.cutout)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsize, shuffle=True, num_workers=args.num_worker)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batchsize, shuffle=False, num_workers=args.num_worker)

    ce_loss = nn.CrossEntropyLoss().cuda()
    # filename = 'misc/'+'{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(args.metric, args.searchspace, args.dataset,args.batchsize, \
    #                                 args.cutout, args.gamma1, args.gamma2, args.maxbatch)
    return imgsize, ce_loss, trainloader, testloader


def search201(api, netid, dataset):
    """
    查询网络性能，并遍历输出所有支持硬件的延迟值（包含硬件名和延迟）。
    
    参数:
        api: NATS-Bench 的 API 实例
        netid: 网络索引
        dataset: 数据集名称（如 'cifar10'）
    
    返回:
        网络模型、性能指标（包括延迟信息列表）、邻接矩阵、操作列表
    """
    if dataset == 'cifar10':
        dsprestr = 'ori'
    else:
        dsprestr = 'x'

    # 查询网络的训练结果
    results = api.query_by_index(netid, dataset, hp='200')  # 获取网络的完整评估结果
    train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0

    for seed, result in results.items():
        train_loss += result.get_train()['loss']
        train_acc += result.get_train()['accuracy']
        test_loss += result.get_eval('ori-test')['loss']
        test_acc += result.get_eval('ori-test')['accuracy']

    # 计算平均值
    num_trials = len(results)
    train_loss /= num_trials
    train_acc /= num_trials
    test_loss /= num_trials
    test_acc /= num_trials

    # 获取网络配置信息
    config = api.get_net_config(netid, dataset)
    arch_str = config['arch_str']
    adjacency = api.str2matrix(arch_str)
    operations = api.str2lists(arch_str)

    # 实例化网络模型
    network = models.get_cell_based_tiny_net(config)

    # 查询支持的硬件设备
    hw_api = HWAPI("/home/test0/HW-NAS-Bench/HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
    latency_matric = get_hardware_metrics(hw_api, netid, dataset)

    # 返回结果
    return network, [train_acc, train_acc, test_loss, test_acc], adjacency, operations, latency_matric


# def search_nats(api, netid, dataset, hpval):
#     """
#     查询 NATS-size 搜索空间中的网络信息。
#     """
#     # 获取网络性能指标
#     info = api.get_more_info(netid, dataset, hp=hpval)
#     test_acc = info['test-accuracy']
#     test_loss = info['test-loss'] if 'test-loss' in info else None
#     train_acc = info['train-accuracy'] if 'train-accuracy' in info else None
#     train_loss = info['train-loss'] if 'train-loss' in info else None

#     # 获取网络配置
#     config = api.get_net_config(netid, dataset)

#     # 实例化网络
#     network = models.get_cell_based_tiny_net(config)

#     # 获取架构字符串
#     arch_str = api.meta_archs[netid]

#     # 自定义解析架构字符串为邻接矩阵和操作列表
#     adjacency, operations = parse_architecture(arch_str)

#     # 构造 metrics，符合 get_basic 的需求
#     metrics = [train_loss, train_acc, test_loss, test_acc]
    
#     # 返回结果(加了test_acc)
#     return network, test_acc, metrics, adjacency, operations



def parse_architecture(arch_str):
    """
    解析架构字符串为邻接矩阵和操作列表。
    假设 arch_str 是用 '|' 分隔的操作名称，例如 "nor_conv_3x3|skip_connect|avg_pool_3x3"。
    """
    # 分割操作名称
    operations = arch_str.split("|")
    num_nodes = len(operations)

    # 构建邻接矩阵，假设为完全有向图
    adjacency = np.array([[1 if i < j else 0 for j in range(num_nodes)] for i in range(num_nodes)])

    return adjacency, operations



def get101acc(data_dict:dict):
    acc4=(data_dict[4][0]['final_test_accuracy']+data_dict[4][1]['final_test_accuracy']+data_dict[4][2]['final_test_accuracy'])/3.0
    acc12=(data_dict[12][0]['final_test_accuracy']+data_dict[12][1]['final_test_accuracy']+data_dict[12][2]['final_test_accuracy'])/3.0
    acc36=(data_dict[36][0]['final_test_accuracy']+data_dict[36][1]['final_test_accuracy']+data_dict[36][2]['final_test_accuracy'])/3.0
    acc108=(data_dict[108][0]['final_test_accuracy']+data_dict[108][1]['final_test_accuracy']+data_dict[108][2]['final_test_accuracy'])/3.0
    return [acc4,acc12,acc36,acc108]
    

def save_network_structure(netid, adjacency, operations, save_path="saved_networks"):
    """
    保存网络的架构到JSON文件中。
    """
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"network_{netid}.json")

    network_config = {
        "netid": netid,
        "adjacency": adjacency.tolist(),
        "operations": operations
    }

    with open(file_path, "w") as f:
        json.dump(network_config, f, indent=4)
    print(f"Network structure saved to {file_path}")


def calculate_score(api,A, 
                    netid,
                    args, #nrs 要用
                    network, #nrs 要用
                    metrics, #basic 要用 准确度
                    macs, #basic 要用
                    params, #basic 要用
                    data, #nrs 要用
                    label, #nrs 要用
                    imgsize, #nrs 要用
                    ce_loss,#nrs 要用
                    latency_matrics,
                    grad_weights,#nrs 要用
                    alpha=1.0, beta=0.5, gamma=0.3, delta=0.05, repeat=3, zen_metric="avg"):
    """
    根据输入值 A 的类型计算得分，初步实现 `basic` 类型计算，其他情况留空。
    
    参数:
        A (str): 唯一输入值，决定具体的计算类型。
        network: 神经网络模型，需包含 `test_acc`、`macs` 和 `params` 属性。
        alpha, beta, gamma: 计算得分的权重参数。
        其他参数保留以支持未来扩展。
    
    返回:
        network: 更新了 `score` 属性的网络模型。
    """
    test_acc = metrics[3]  # 获取测试准确率
    if A == "basic":
        score= (alpha * test_acc - beta * np.log10(macs) - gamma * np.log10(params))
        return score
    elif A == "lp":
        score = (alpha * test_acc - beta * np.log10(macs) - gamma * np.log10(params) - delta * latency_matrics['edgegpu']['latency'])
        return score                                                                                     
    elif A == "nrs":

        score = get_new_score(latency_matrics,params, macs, test_acc, args,netid,api
                        
                    )
        return score
    else:
        raise ValueError(f"未知的输入类型 A: {A}")
    


    
def get_metric_feedback(api,args, netid, network, metrics, imgsize, adjacency, operations, latency_matrics, top_k_list, data, label, ce_loss, grad_weights,space='cv'):
    """
    修改后的get_basic函数，记录所有网络的信息，并更新top_k_list，包含完整的Latency信息。
    """
    if space == 'cv':
        # 计算模型的复杂性指标（MACs和参数数量）
        macs, params = get_model_complexity_info(network, (3, imgsize, imgsize), as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        
        # 提取 GPU 延迟，或计算平均延迟
        hardware = "GPU" #这个地方可以替换 换成想要的device
        score=calculate_score(api,args.metric,netid, args, #nrs 要用
                                network, #nrs 要用
                                metrics, #basic 要用
                                macs, #basic 要用
                                params, #basic 要用
                                data, #nrs 要用
                                label, #nrs 要用
                                imgsize, #nrs 要用
                                ce_loss,#nrs 要用
                                latency_matrics, #lp要用
                                grad_weights,#nrs 要用
                                alpha=1.0, beta=0.5, gamma=0.3, delta=0.3, repeat=3, zen_metric="avg")
        # 保存网络的所有信息，包括Latency
        if metrics[3] > 93.7:
            top_k_list.append({
                'score': score,
                'netid': netid,
                'macs': macs,
                'params': params,
                'metrics': metrics,  # 完整的性能指标，包括训练和测试信息
                'adjacency': adjacency,  # 保存邻接矩阵
                'operations': operations,  # 保存操作列表
                'latency': latency_matrics  # 保存完整的Latency信息
            })
        top_k_list.sort(key=lambda x: x['score'], reverse=True)  # 按score 降序排序
        print("test1 for output")
        top_k_list = top_k_list[:10]  # 截取前 10 个
    return top_k_list

    
def enumerate_networks(args):
    imgsize, ce_loss, trainloader, testloader = getmisc(args)
    top_k_list = []  # 存储符合条件的网络
    print("2 successfully.")
    
    
    if  args.metric=='nrs':
        print("4 successfully.")
        api = API201('/home/test0/dataset/nasbench/NAS-Bench-201-v1_1-096897.pth', verbose=False)
        groupIDs=load_group_ids(args.testName)
        
        for i, batch in enumerate(trainloader):
            data,label = batch[0],batch[1]
            data,label=data.cuda(),label.cuda()
            break
        
        # for netid in groupIDs:  # 遍历 groupIDs 数组
        for netid in groupIDs:
            print(f"Processing network {netid}")
            # 使用 search201 函数获取网络实例和相关信息
            network, metrics, adjacency, operations, latency_matric = search201(api, netid, args.dataset)
            train_acc, val_acc, test_loss, test_acc = metrics
            network.cuda()
            grad_weights = np.ones(7)
            top_k_list = get_metric_feedback(
                api,args, netid, network, metrics, imgsize, adjacency, operations, latency_matric, 
                top_k_list, data, label, ce_loss, grad_weights
            )
    
    # elif '101' in args.searchspace.lower():
    #     assert args.dataset == "cifar10"
    #     NASBENCH_TFRECORD = '/home/test0/dataset/nasbench/nasbench_full.tfrecord'
    #     nasbench = api101.NASBench(NASBENCH_TFRECORD)

    #     def getallacc(data_dict: dict):
    #         acc4 = sum(data_dict[4][i]['final_test_accuracy'] for i in range(3)) / 3.0
    #         acc12 = sum(data_dict[12][i]['final_test_accuracy'] for i in range(3)) / 3.0
    #         acc36 = sum(data_dict[36][i]['final_test_accuracy'] for i in range(3)) / 3.0
    #         acc108 = sum(data_dict[108][i]['final_test_accuracy'] for i in range(3)) / 3.0
    #         return [acc4, acc12, acc36, acc108]

    #     if args.metric in ['logdet', 'grad']:
    #         for i, batch in enumerate(trainloader):
    #             data, label = batch[0], batch[1]
    #             data, label = data.cuda(), label.cuda()
    #             break

    #     allnethash = list(nasbench.hash_iterator())
    #     #for netid in range(args.startnetid, len(allnethash)):
    #     for netid in range(args.startnetid, min(args.startnetid + 100, len(allnethash))):
    #         unique_hash = allnethash[netid]
    #         fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
    #         acc_metrics = getallacc(computed_metrics)

    #         ops = fixed_metrics['module_operations']
    #         adjacency = fixed_metrics['module_adjacency']

    #         network = NB101Network((adjacency, ops))
    #         network.cuda()
        
    #         if args.metric == 'basic':
    #             get_basic(netid, network, acc_metrics, imgsize, adjacency, ops, top_k_list=top_k_list)
    #         elif args.metric == 'Final':
    #             zen_score = get_zenscore(netid, network, imgsize, args.batchsize)
    #             ntk = get_ntk_n(trainloader, network, train_mode=True, num_batch=1)
    #             logdet_k = get_logdet(netid, network, args, data, label)
    #             grad_sensitivity = get_grad_score(netid, network, data, label, ce_loss, split_data=1, device='cuda')


        
                
    elif '201' in args.searchspace.lower():
        api = API201('/home/test0/dataset/nasbench/NAS-Bench-201-v1_1-096897.pth', verbose=False)
 
        for i, batch in enumerate(trainloader):
            data,label = batch[0],batch[1]
            data,label=data.cuda(),label.cuda()
            break

        for netid in range(1000):
            print(f"Processing network {netid}")
            # 使用 search201 函数获取网络实例和相关信息
            network, metrics, adjacency, operations, latency_matric= search201(api, netid, args.dataset)
            train_acc, val_acc, test_loss, test_acc= metrics
            network.cuda()
            data, label = next(iter(trainloader))
            data, label = data.cuda(), label.cuda()
            grad_weights = np.ones(7) 
            top_k_list=get_metric_feedback(api,args, netid, network, metrics, imgsize, adjacency, operations, latency_matric, top_k_list, data, label, ce_loss, grad_weights)
            
                
                
                # zen_score = get_zenscore(network, imgsize, args.batchsize)#未发现问题 201用时：几分钟
                # print(f"Zen Score: {zen_score}")
                # ntk = get_ntk_n(trainloader, network, train_mode=True, num_batch=1)#可能有问题  201用时：15分钟
                # print(f"NTK Condition Number: {ntk}")
                
                # logdet_k = get_logdet(network, args, data, label)#有问题 输出-inf，不知道是什么  201用时：几分钟
                # print(f"Log-determinant (logdet_k): {logdet_k}")
                
                # grad_sensitivity = get_grad_score( network, data, label, ce_loss, split_data=1, device='cuda')#未发现问题
                # print(f"Gradient Sensitivity: {grad_sensitivity}")
                    
                        
                        
  
                
    # elif 'nats' in args.searchspace.lower():
    #     if 'tss' in args.searchspace.lower():
    #         # Create the API instance for the topology search space in NATS
    #         api = create_nats('/home/test0/dataset/nasbench/NATS/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=True)
    #         hpval='200'
    #     else:
    #         # Create the API instance for the size search space in NATS
    #         api = create_nats('/home/test0/dataset/nasbench/NATS/NATS-sss-v1_0-50262-simple', 'sss', fast_mode=True, verbose=True)
    #         hpval='90'

    #     if args.metric in ['logdet', 'grad']:
    #         for i, batch in enumerate(trainloader):
    #             data,label = batch[0],batch[1]
    #             data,label=data.cuda(),label.cuda()
    #             break

    #     for netid in range(100):
    #         network, test_acc, adjacency, operations = search_nats(api, netid, args.dataset, hpval)
    #         network.cuda()
    #         if args.metric =='basic':
    #             #get_basic(netid, network, metric, imgsize)
    #             get_basic(netid, network, test_acc, imgsize, adjacency, operations, top_k_list=top_k_list)
    #         elif args.metric =='ntk':
    #             get_ntk_n(netid, trainloader, network, train_mode=True, num_batch=1)
    #         elif args.metric =='logdet':
    #             get_logdet(netid, network, args, data, label)
    #         elif args.metric =='zen':
    #             get_zenscore(netid, network, imgsize, args.batchsize)
    #         elif args.metric =='grad':
    #             get_grad_score(netid, network,  data, label, ce_loss, split_data=1, device='cuda')
    #         elif args.metric == 'Final':
    #             zen_score = get_zenscore(netid, network, imgsize, args.batchsize)
    #             ntk = get_ntk_n(netid, trainloader, network, train_mode=True, num_batch=1)
    #             logdet_k = get_logdet(netid, network, args, data, label)
    #             grad_sensitivity = get_grad_score(netid, network, data, label, ce_loss, split_data=1, device='cuda')
                
    if args.metric != "nrs":            
        # Collect all network IDs
        net_ids = [net['netid'] for net in top_k_list]

        # Save all network IDs to a grouped file
        group_file_name = f"{args.metric}Group.txt"
        with open(group_file_name, "w") as f:
            for net_id in net_ids:
                f.write(f"{net_id}\n")

        print(f"Network IDs saved to {group_file_name}")

        # Save individual network structures and print information
        for net in top_k_list:
            save_network_structure(net['netid'], net['adjacency'], net['operations'])
            print(f"Saved network {net['netid']} with score {net['score']:.4f}")
    else:
    # 计算平均分并打印
        if top_k_list:  # 确保列表非空
            average_score = sum(net['score'] for net in top_k_list) / len(top_k_list)
            print(f"{args.testName} NRS score is ：{average_score:.4f}")
        else:
            print(f"{args.testName} NRS score is：no data")
        

def load_group_ids(group_file_name=None):
    """
    Load up to 10 IDs from the specified group file.

    Parameters:
        group_file_name (str): The name of the file to read IDs from. 
                               Must be either 'basicGroup.txt' or 'lpGroup.txt'.
                               Default is None, which raises an exception.

    Returns:
        list: A list of up to 10 IDs (integers).
    """
    # Validate input
    if group_file_name is None:
        raise ValueError("You must specify a group file name ('basicGroup.txt' or 'lpGroup.txt').")
    if group_file_name not in ["basicGroup.txt", "lpGroup.txt"]:
        raise ValueError("Invalid file name! Must be 'basicGroup.txt' or 'lpGroup.txt'.")

    # Initialize an empty list to store IDs
    group_ids = []

    # Read up to 10 IDs from the specified file
    try:
        with open(group_file_name, "r") as f:
            for line in f:
                # Convert the line to an integer and add it to group_ids
                group_ids.append(int(line.strip()))
                if len(group_ids) >= 10:  # Stop after reading 10 IDs
                    break
    except FileNotFoundError:
        raise FileNotFoundError(f"File {group_file_name} not found. Please ensure it exists.")

    # Return the list of IDs
    return group_ids

if __name__ == '__main__':
    print("1 successfully.")
    enumerate_networks(args)
    
    # imgsize, ce_loss, trainloader, testloader = getmisc(args)
    # top_k_list = []  # 存储符合条件的网络
    # print("2 successfully.")
    
    
    # if '201' in args.searchspace.lower(): 
    #     api = API201('/home/test0/dataset/nasbench/NAS-Bench-201-v1_1-096897.pth', verbose=False)
        
    #     for i, batch in enumerate(trainloader):
    #         print("1")
    #         data,label = batch[0],batch[1]
    #         data,label=data.cuda(),label.cuda()
    #         break            

    #     for netid in 1,3,4,5:
    #         network, metrics, adjacency, operations, latency_matric= search201(api, netid, args.dataset)
    #         network.cuda()

            
            
            
            
    #         # zen_score=get_zenscore(network, imgsize, args.batchsize)
    #         # print(f"Log-determinant (logdet_k): {zen_score}")
    #         logdet_k=get_logdet(network, args, data, label)
    #         print(f"Zen Score: {logdet_k}")
    #         grad_sensitivity=get_grad_score(network,  data, label, ce_loss, split_data=1, device='cuda')
    #         print(f"Gradient Sensitivity: {grad_sensitivity}")
            
                
                # zen_score = get_zenscore(network, imgsize, args.batchsize)#未发现问题 201用时：几分钟
                # print(f"Zen Score: {zen_score}")
                
                
                # logdet_k = get_logdet(network, args, data, label)#有问题 输出-inf，不知道是什么  201用时：几分钟
                # print(f"Log-determinant (logdet_k): {logdet_k}")
                
                # grad_sensitivity = get_grad_score( network, data, label, ce_loss, split_data=1, device='cuda')#未发现问题
                # print(f"Gradient Sensitivity: {grad_sensitivity}")