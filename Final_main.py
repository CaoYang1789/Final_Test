from typing import Dict
import os
import sys

# Add HW-NAS-Bench path
hw_nas_bench_path = "/home/test0/HW-NAS-Bench"  # modify by yourself
if hw_nas_bench_path not in sys.path:
    sys.path.append(hw_nas_bench_path)

# Ensure that other modules are imported properly
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

# Import HWNASBenchAPI
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
    Gets all hardware performance metrics for the specified network architecture on the target dataset from HW-NAS-Bench.

    Parameters:
    api: An API instance of HW-NAS-Bench.
    netid: indicates the index of the network.
    dataset: The name of the dataset (such as 'cifar10').

    Back:
    A nested dictionary of the format:
    {
    'edgegpu: {' latency' : 5.8074, 'energy' : 24.2266},
    'raspi4': {'latency': 10.482, 'energy': None},
    ...
    }
    """

    metrics = api.query_by_index(netid, dataset)
    result = {}

    for k, v in metrics.items():
        if "_latency" in k or "_energy" in k:
            metric_type = "latency" if "_latency" in k else "energy"
            hardware = k.replace(f"_{metric_type}", "")  
            if hardware not in result:
                result[hardware] = {"latency": None, "energy": None}  
            result[hardware][metric_type] = round(v, 4)  
    # print("EdgeGPU Latency:", formatted_metrics['edgegpu']['latency'])  # print: 5.8074
    # print("EdgeGPU Energy:", formatted_metrics['edgegpu']['energy'])  # print: 24.2266
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
    Queries network performance and iterates output latency values (including hardware name and latency) for all supported hardware.

    Parameters:
    api: An API instance of NTS-Bench
    netid: indicates the network index
    dataset: Name of the dataset (e.g. 'cifar10')

    Back:
    Network model, performance metrics (including delay information list), adjacency matrix, operation list
    """
    if dataset == 'cifar10':
        dsprestr = 'ori'
    else:
        dsprestr = 'x'

    # Query the training result of the network
    results = api.query_by_index(netid, dataset, hp='200')  # Get a complete evaluation of your network
    train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0

    for seed, result in results.items():
        train_loss += result.get_train()['loss']
        train_acc += result.get_train()['accuracy']
        test_loss += result.get_eval('ori-test')['loss']
        test_acc += result.get_eval('ori-test')['accuracy']

    
    num_trials = len(results)
    train_loss /= num_trials
    train_acc /= num_trials
    test_loss /= num_trials
    test_acc /= num_trials

    
    config = api.get_net_config(netid, dataset)
    arch_str = config['arch_str']
    adjacency = api.str2matrix(arch_str)
    operations = api.str2lists(arch_str)

    # Instantiate the network model
    network = models.get_cell_based_tiny_net(config)

    # Obtain hardware device information
    hw_api = HWAPI("/home/test0/HW-NAS-Bench/HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
    latency_matric = get_hardware_metrics(hw_api, netid, dataset)

    return network, [train_acc, train_acc, test_loss, test_acc], adjacency, operations, latency_matric






def parse_architecture(arch_str):
    """
    Parse schema strings into adjacency matrices and action lists.
    Assume that arch_str is an operation name separated by '|', for example, "nor_conv_3x3|skip_connect|avg_pool_3x3".
    """

    operations = arch_str.split("|")
    num_nodes = len(operations)

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
    Save the network architecture to a JSON file.
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
                    args,
                    network, 
                    metrics, 
                    macs, 
                    params, 
                    data, 
                    label, 
                    imgsize, 
                    ce_loss,
                    latency_matrics,
                    grad_weights,
                    alpha=1.0, beta=0.5, gamma=0.3, delta=0.05, repeat=3, zen_metric="avg"):
    """
    The score is calculated according to the type of the input value A, and the 'basic' type is preliminarily realized, and other cases are left blank.

    Parameters:
    A (str): Unique input value that determines the specific type of calculation.
    network: The neural network model, which must contain the 'test_acc', 'macs' and' params' attributes.
    alpha, beta, gamma: Calculate the weight parameters of the score.
    Other parameters are retained to support future expansion.

    Back:
    network: Updated the network model with the 'score' attribute.
    """
    test_acc = metrics[3]  
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
    The modified get_basic function logs all network information and updates the top_k_list to include the full Latency information.
    """
    if space == 'cv':
        # Calculate the complexity indicators of the model (MACs and number of parameters)
        macs, params = get_model_complexity_info(network, (3, imgsize, imgsize), as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        
        
        hardware = "GPU" #No need actually
        score=calculate_score(api,args.metric,netid, args, 
                                network,
                                metrics, 
                                macs, 
                                params, 
                                data, 
                                label, 
                                imgsize,
                                ce_loss,
                                latency_matrics, 
                                grad_weights,
                                alpha=1.0, beta=0.5, gamma=0.3, delta=0.3, repeat=3, zen_metric="avg")
        # Save all information about the network
        if metrics[3] > 93.7:
            top_k_list.append({
                'score': score,
                'netid': netid,
                'macs': macs,
                'params': params,
                'metrics': metrics,  
                'adjacency': adjacency,  
                'operations': operations,  
                'latency': latency_matrics  
            })
        top_k_list.sort(key=lambda x: x['score'], reverse=True)  # Sort by score in descending order
        print("test1 for output")
        top_k_list = top_k_list[:10]  # Intercept the first 10 to avoid memory explosion
    return top_k_list

    
def enumerate_networks(args):
    imgsize, ce_loss, trainloader, testloader = getmisc(args)
    top_k_list = []  
    print("2 successfully.")
    
    
    if  args.metric=='nrs':
        print("4 successfully.")
        api = API201('/home/test0/dataset/nasbench/NAS-Bench-201-v1_1-096897.pth', verbose=False)
        groupIDs=load_group_ids(args.testName)
        
        for i, batch in enumerate(trainloader):
            data,label = batch[0],batch[1]
            data,label=data.cuda(),label.cuda()
            break
        
        
        for netid in groupIDs:
            print(f"Processing network {netid}")
            
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
            
            network, metrics, adjacency, operations, latency_matric= search201(api, netid, args.dataset)
            train_acc, val_acc, test_loss, test_acc= metrics
            network.cuda()
            data, label = next(iter(trainloader))
            data, label = data.cuda(), label.cuda()
            grad_weights = np.ones(7) 
            top_k_list=get_metric_feedback(api,args, netid, network, metrics, imgsize, adjacency, operations, latency_matric, top_k_list, data, label, ce_loss, grad_weights)
            
                

                        
                        
  
                
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
    
        if top_k_list: 
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
    # top_k_list = []  
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
