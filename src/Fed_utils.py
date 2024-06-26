import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def model_to_device(model, parallel, device):
    if parallel:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        card = torch.device("cuda:{}".format(device))
        model.to(card)
    return model

def participant_exemplar_storing(clients, num, model_g, old_client, task_id, clients_index):
    print(f"\n=========participant_exemplar_storing()====================\n")
    print(f"num: {num}, old_client: {old_client}, task_id: {task_id}, clients_index: {clients_index}")
    for index in range(num):
        clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)
            clients[index].update_new_set()

def participant_exemplar_storing_fcil(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        # clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)
            clients[index].update_new_set()        

def local_train(clients, index, model_g, task_id, model_old, ep_g, old_client):
    print(f"\n========local_train()====================")
    print(f"index: {index}, task_id: {task_id}, ep_g: {ep_g}, old_client: {old_client}")
    clients[index].model = copy.deepcopy(model_g)

    if index in old_client:
        clients[index].beforeTrain(task_id, 0)
    else:
        clients[index].beforeTrain(task_id, 1)

    clients[index].update_new_set()
    # print(clients[index].signal)
    print(f"signal: {clients[index].signal}")
    clients[index].train(ep_g, model_old)
    local_model = clients[index].model.state_dict()
    proto_grad = clients[index].proto_grad_sharing()

    print('*' * 60)

    return local_model, proto_grad

def local_train_fcil(clients_index_push, clients, index, model_g, task_id, model_old, ep_g, old_client, client_index, global_task_id_real, class_real=None):
    if index in clients_index_push:
        if "sharedcodap" in model_g.args.method and clients[index].model is not None:
            client_learned_global_task_id_saved = clients[index].model.prompt.client_learned_global_task_id
            clients[index].model = copy.deepcopy(model_g)
            clients[index].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved
        else:
            clients[index].model = copy.deepcopy(model_g)
    else:
        if "sharedcodap" in model_g.args.method and clients[index].model is not None:
            client_learned_global_task_id_saved = clients[index].model.prompt.client_learned_global_task_id
            temp_model = copy.deepcopy(model_g)
            temp_state_dict = copy.deepcopy(clients[index].model.state_dict())
            temp_model.load_state_dict(temp_state_dict)
            clients[index].model = temp_model
            clients[index].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved
        else:
            temp_model = copy.deepcopy(model_g)
            temp_state_dict = copy.deepcopy(clients[index].model.state_dict())
            temp_model.load_state_dict(temp_state_dict)
            clients[index].model = temp_model
            model_old = [clients[index].old_model, clients[index].old_model]
    
    if index in old_client:             
        clients[index].beforeTrain(task_id, 0, client_index, global_task_id_real, class_real)
    else:
        clients[index].beforeTrain(task_id, 1, client_index, global_task_id_real, class_real)
    clients[index].update_new_set(task_id, client_index)
    print(f"client[{index}].signal: {clients[index].signal}")
    num_samples = clients[index].train(ep_g, model_old)
    local_model = clients[index].model.state_dict()
    proto_grad =  None    
    print('*' * 60)

    return local_model, proto_grad, num_samples

def FedAvg_weit(models, model_last_round, num_samples_list, global_weight, client_index, class_distribution_client, taskid_local, models_model, task_id, clients_index_pull, num_clients, global_task_id_real):
    if task_id == 0:
        global_weight = 1
    
    summation = sum([num_samples_list[i] for i in range(len(num_samples_list)) if client_index[i] in clients_index_pull])
    if "extension" not in models_model[0].model.args.method:
        w_avg = copy.deepcopy(models[0])
    else:
        w_avg = copy.deepcopy(model_last_round)    
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            if client_index[i] in clients_index_pull:
                weight = num_samples_list[i] / summation
                print(f"key: {key}, client_index[{i}]: {client_index[i]}")
                if key == "fc.weight" or key == "fc.bias":    
                    if client_index[i] != 20:   # TODO: why 20?
                        if weighted_sum is None:
                            weighted_sum = copy.deepcopy(models[i][key])
                            #weighted_sum[list(class_frequency.keys())] = 0 * models[i][key][list(class_frequency.keys())]
                            #for c in models_model[client_index[i]].model.current_class:
                                #weighted_sum[c] = (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                            #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                            weighted_sum[models_model[client_index[i]].model.current_class] = models[i][key][models_model[client_index[i]].model.current_class]
                        else:
                            #for c in models_model[client_index[i]].model.current_class:
                                #weighted_sum[c] += (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                            #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                            weighted_sum[models_model[client_index[i]].model.current_class] = models[i][key][models_model[client_index[i]].model.current_class]
                            #weighted_sum[clients_learned_class[i]] = models[i][key][clients_learned_class[i]]
                    else:
                        pass
                elif key == "aggregate_weight" and "weit" in models_model[0].model.args.method:
                    if weighted_sum is None:
                        weighted_sum = copy.deepcopy(models[i][key])
                        weighted_sum[:, models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index] = models[i][key][:, models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index]
                    else:
                        weighted_sum[:, models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index] = models[i][key][:, models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index]
                else:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                    else:
                        weighted_sum += weight * models[i][key]
        w_avg[key] = weighted_sum
    
    return w_avg
                
def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg

def model_global_eval_hard(model_g, test_dataset, task_id, task_size, device,
                           method, task_num, global_class_output, global_class_output_real):
    model_to_device(model_g, False, device)
    model_g.eval()
    print(f"[Function: model_global_eval_hard]")
    print(f"model_g.global_class_min_output: {model_g.global_class_min_output}")
    print(f"model_g.client_class_min_output: {model_g.client_class_min_output}")
    print(f"model_g.client_index: {model_g.client_index}")
    print(f"global_class_output: {global_class_output}")
    test_dataset.getTestData_hard(global_class_output, global_class_output_real)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
    correct, total = 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(imgs)
        
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        if step == 1:
            predicts_1 = predicts.cpu()
            labels_1 = labels.cpu()
        total += len(labels)
    accuracy = 100 * correct / total
    
    accuracys = []
    
    for i in global_class_output_real:
        test_dataset.getTestData_hard([global_class_output[global_class_output_real.index(i)]], [i])
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128, num_workers=8, pin_memory=True)
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
            with torch.no_grad():
                outputs = model_g(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracys.append(100 * correct / total)
    
    model_g.train()
    print(f"accuracy: {accuracy}\naccuracys: {accuracys}")
    return accuracy, accuracys
            
def model_global_eval(model_g, test_dataset, task_id, task_size, device):
    model_to_device(model_g, False, device)
    model_g.eval()
    test_dataset.getTestData([0, task_size * (task_id + 1)])
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)
    print(f"len(test_loader): {len(test_loader)}")
    correct, total = 0, 0
    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    model_g.train()
    return accuracy

