from GLFC import GLFC_model
from GLFC_hard import GLFC_model_hard
from ResNet import resnet18_cbam
from models_Cprompt.vision_transformer import VisionTransformer
import torch
import copy
import random
import os.path as osp
import os
from myNetwork import network, LeNet
from Fed_utils import * 
from ProxyServer import * 
from ProxyServer_hard import *
from mini_imagenet import *
from tiny_imagenet import *
from option import args_parser
import dataloaders
from iDOMAINNET import *
from myNetwork_hard import network_hard, LeNet_hard
import nni

import os
current_directory = os.getcwd()
print("Current working directory:", current_directory)

params = {'task_index':30, 'topk_for_task':3, 'topk_for_task_selection':3,
          'class_index':30, 'topk_for_class':3,
          'learning_rate':0.001, 'epochs_local': 2, 'batch_size': 32, 'seed': 0}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(f"params: {params}")

args = args_parser()
print(f"vars(args): {vars(args)}")
## seed settings
setup_seed(args.seed)
if len(args.device) == 1:
    args.device = args.device[0]
else:
    torch.cuda.set_device(args.device[0])
    
class_distribution_client_di = None


if "centralized" in args.method or "one" in args.method or args.num_clients == 1:
    class_list = list(range(args.numclass))
    class_list_real = random.sample(list(range(args.numclass)), args.numclass)
    class_distribution_client = {0:[]}
    class_distribution_client_real = {0:[]}
    class_distribution_client_proportion = {0:[]}
    class_per_task = int(args.numclass / args.epochs_global * args.tasks_global)
    for i in range(int(args.epochs_global / args.tasks_global)):
        class_distribution_client[0].append(class_list[i*class_per_task:(i+1)*class_per_task])
        class_distribution_client_real[0].append(class_list_real[i*class_per_task:(i+1)*class_per_task])
        class_distribution_client_proportion[0].append([0,1])
elif "dil" in args.method:
    class_list = random.sample(list(range(200)), 200)[0:20]
    class_distribution_client = {}
    class_distribution_client_real = {}
    class_distribution_client_proportion = {}
    class_distribution_client_di = {}
    label_distribution = np.random.dirichlet([0.1]*25, 20)
    for i in range(args.num_clients):
        task_list = []
        for j in range(int(args.epochs_global / args.tasks_global)):
            task_list.append(list(range((i * int(args.epochs_global / args.tasks_global) + j) * 20, (i * int(args.epochs_global / args.tasks_global) + j + 1) * 20)))
        class_distribution_client[i] = task_list
    for i in range(args.num_clients):
        task_list_real = []
        for j in range(int(args.epochs_global / args.tasks_global)):
            task_list_real.append(class_list)
        class_distribution_client_real[i] = task_list_real
    for i in range(args.num_clients):
        task_list_pro = []
        for j in range(int(args.epochs_global / args.tasks_global)):
            task_list_pro.append([0,0.2])
        class_distribution_client_proportion[i] = task_list_pro
    for i in range(args.num_clients):
        task_list_di = []
        for j in range(int(args.epochs_global / args.tasks_global)):
            class_list_di = []
            for k in range(20):
                class_list_di.append(label_distribution[k][i * int(args.epochs_global / args.tasks_global) + j])
            task_list_di.append(class_list_di)
        class_distribution_client_di[i] = task_list_di
else:
    if args.dataset == 'ImageNet_R':
        class_distribution_client = {}
        class_distribution_client_real = {}
        class_distribution_client_proportion = {}
        if "full" not in args.method:
            for i in range(args.num_clients):
                task_list = []
                for j in range(int(args.epochs_global / args.tasks_global)):
                    task_list.append(list(range((i * int(args.epochs_global / args.tasks_global) + j) * 20, (i * int(args.epochs_global / args.tasks_global) + j + 1) * 20)))
                class_distribution_client[i] = task_list
        else:    
            task_list = []
            for i in range(20):
                task_list.append(list(range(i * 20, (i + 1) * 20)))
            class_distribution_client[0] = task_list
            for i in range(1,50):
                class_distribution_client[i] = [list(range(i * 20 + 400, (i + 1) * 20 + 400))]
                
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                if i * int(args.epochs_global / args.tasks_global) + j > 0:
                    similar_global_task_id = random.sample(list(range(i * int(args.epochs_global / args.tasks_global) + j)), 1)[0]
                    similar_level = random.sample(list(range(args.sim, 21)), 1)[0]
                    if similar_level != 0:
                        similar_client_id = int(similar_global_task_id // int(args.epochs_global / args.tasks_global))
                        similar_task_id = int(similar_global_task_id % int(args.epochs_global / args.tasks_global))
                        if similar_client_id == i:
                            task_list.append(list(random.sample(task_list[similar_task_id], similar_level)) + list(random.sample(list(set(list(range(200)))-set(task_list[similar_task_id])), 20-similar_level)))
                        else:
                            task_list.append(list(random.sample(class_distribution_client_real[similar_client_id][similar_task_id], similar_level)) + list(random.sample(list(set(list(range(200)))-set(class_distribution_client_real[similar_client_id][similar_task_id])), 20-similar_level)))                    
                    else:
                        task_list.append(list(random.sample(list(range(200)), 20))) 
                else:
                    task_list.append(list(random.sample(list(range(200)), 20))) 
            class_distribution_client_real[i] = task_list

        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                proportion_list = [0, 0.2, 0.4, 0.6, 0.8]
                start = proportion_list[int(i * int(args.epochs_global / args.tasks_global) + j) % 5]
                task_list.append([start, start + 0.2])
                
            class_distribution_client_proportion[i] = task_list

print("********* FULL TEST **********")
print(f"class_distribution_client: {class_distribution_client}")
print(f"class_distribution_client_real: {class_distribution_client_real}")
print(f"class_distribution_client_proportion: {class_distribution_client_proportion}")
print(f"class_distribution_client_di: {class_distribution_client_di}")

print(f"fcil in {args.method}: {'fcil' in args.method}")

## parameters for learning
global_class_output = []
global_trained_task_id = []
global_trained_task_id_nosame = []
global_not_trained_task_id = []
# feature_extractor = resnet18_cbam()
feature_extractor = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, 
                                      depth=12, num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                        drop_path_rate=0, args=args
                                        )
from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
load_dict = vit_base_patch16_224_in21k(pretrained=True).state_dict()
del load_dict['head.weight']; del load_dict['head.bias']
feature_extractor.load_state_dict(load_dict)
print(" freezing original model")
for n,p  in feature_extractor.named_parameters():
    if not "prompt" in n:
        print(f"freezing {n}")
        p.requires_grad = False
        
num_clients = args.num_clients
old_client_0 = []
old_client_1 = [i for i in range(args.num_clients)]
new_client = []
models = []


## model settings
# model_g = network(args.numclass, feature_extractor)
model_g = network_hard(args.numclass, feature_extractor, 
                       class_distribution_client,
                        class_distribution_real=class_distribution_client_real, class_distribution_proportion=class_distribution_client_proportion, args=args)
model_g = model_to_device(model_g, False, args.device)
model_old = None

# train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),
#                                     transforms.RandomHorizontalFlip(p=0.5),
#                                     transforms.ColorJitter(brightness=0.24705882352941178),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
# test_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), 
#                                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=True, resize_imnet=True)
test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=True, resize_imnet=True)

if args.dataset == 'CIFAR100':
    train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
    test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

elif args.dataset == 'TINY_IMAGENET':
    train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset
elif args.dataset == 'ImageNet_R':
    train_dataset = iIMAGENET_R(args.dataroot, train=True, transform=train_transform, download_flag=True, seed=args.seed, validation=args.validation)
    test_dataset = iIMAGENET_R(args.dataroot, train=False, transform=test_transform, download_flag=False, seed=args.seed, validation=args.validation)
else:
    train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset

if "fcil" in args.method:
    encode_model = LeNet_hard(num_classes=100)
    encode_model.apply(weights_init)
# encode_model = LeNet(num_classes=100)
# encode_model.apply(weights_init)

# for i in range(125):
for i in range(args.num_clients):
    model_temp = GLFC_model_hard(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                    args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model, args.dataset)
    # model_temp = GLFC_model(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                #  args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model)
    models.append(model_temp)
print(f"len(models): {len(models)}")

## the proxy server
if "fcil" in args.method:
    proxy_server = proxyServer_hard(args.device, args.learning_rate, args.numclass, feature_extractor, encode_model, train_transform)
# proxy_server = proxyServer(args.device, args.learning_rate, args.numclass, feature_extractor, encode_model, train_transform)

## training log
output_dir = osp.join('./training_log', args.method, 'seed' + str(args.seed))
if not osp.exists(output_dir):
    os.system('mkdir -p ' + output_dir)
if not osp.exists(output_dir):
    os.mkdir(output_dir)

out_file = open(osp.join(output_dir, 'log_tar_' + str(args.task_size) + '.txt'), 'w')
log_str = 'method_{}, task_size_{}, learning_rate_{}'.format(args.method, args.task_size, args.learning_rate)
out_file.write(log_str + '\n')
out_file.flush()

old_task_id = -1

choosing = {}
finished_task = {}
finished_task_forchoosing = {}
choosing_class = {}
finished_class = {}

client_finish_task_num = {}
for c in range(args.num_clients):
    client_finish_task_num[c] = 0

unpush_correlation = {}
for c in range(args.num_clients):
    unpush_correlation[c] = 0

unpull_correlation = {}
for c in range(args.num_clients):
    unpull_correlation[c] = 0

# global_task_id_real = {}  # for 'cprompt' prompt_flag
global_task_id_real = {}
for i in range(800):
    global_task_id_real[i] = i

class_real = {}
for i in range(args.numclass):
    class_real[i] = i

clients_index_pull = list(range(num_clients))
clients_index_push = list(range(num_clients))
acc_global_list = []
old_client_1_temp = []

for ep_g in range(args.epochs_global):  
    pool_grad = []
    num_samples_list = []
    model_old = proxy_server.model_back()
    task_id = ep_g // args.tasks_global 
    new_task = (task_id != old_task_id) # whether a new task is coming
    print(f"new_task is comming: {new_task}")

    if task_id != old_task_id and old_task_id != -1 and "extension" not in args.method and "extencl" not in args.method:
        overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
        new_client = []
        old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.5))
        old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
        num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
        print(f"old_client_0 :{old_client_0}\nold_client_1: {old_client_1}\nnew_client: {new_client}")
        for c in old_client_1:
            client_finish_task_num[c] = client_finish_task_num[c] + 1
        
    if task_id != old_task_id:
        model_g.Incremental_learning(task_id)
    if task_id != old_task_id and old_task_id != -1:   
        model_g = model_to_device(model_g, False, args.device)
    
    print('federated global round: {}, task_id: {}'.format(ep_g, task_id))
    w_local = []
    m_local = []
    taskid_local = []
    clients_learned_task_id = []
    clients_learned_class = []
    
    clients_index = random.sample(range(num_clients), args.local_clients)
    print('select part of clients to conduct local training')
    print(clients_index)
    
    w_g_last = copy.deepcopy(model_g.state_dict())
    for c in clients_index:
        if c in old_client_0:
            continue
        else:
            new_classes = []
            for i in class_distribution_client[c][task_id]:
                new_classes.append(class_real[i])
            global_class_output.extend(new_classes)
            if task_id == 0:
                global_trained_task_id.append(global_task_id_real[c])
                global_trained_task_id_nosame.append(c)
            else:
                global_trained_task_id.append(global_task_id_real[49 + task_id])
                global_trained_task_id_nosame.append(49 + task_id)
    global_class_output = sorted(list(set(global_class_output)))
    global_trained_task_id = sorted(list(set(global_trained_task_id)))
    global_trained_task_id_nosame = sorted(list(set(global_trained_task_id_nosame)))
    print(f"global_class_output: {global_class_output}")
    print(f"global_trained_task_id: {global_trained_task_id}")
    print(f"global_trained_task_id_nosame: {global_trained_task_id_nosame}")
    
    #TODO: some codes to be implemented
    w_g_not_trained = copy.deepcopy(model_g.state_dict())
    for c in clients_index:
        if "fcil" in args.method:
            local_model, proto_grad, num_samples = local_train_fcil(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, c, global_task_id_real=global_task_id_real, class_real=class_real)
            
        if ((ep_g + 1) % args.tasks_global == 0 and ("direct" in args.method or "FLorigin" in args.method or "notran" in args.method)):
            acc_global, accs_global = model_global_eval_hard(models[c].model, test_dataset, task_id, args.task_size, args.device, args.method, int(args.epochs_global / args.tasks_global), models[c].current_class, models[c].current_class_real)
            log_str = 'Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'\
                                        .format(c, models[c].model.task_id, ep_g, acc_global, accs_global)
            out_file.write(log_str + '\n')
            out_file.flush()
        
        w_local.append(local_model)
        if num_samples != None:
            num_samples_list.append(num_samples)
        # local_client_index += 1

    if "extension" not in args.method:
        clients_index_pull = clients_index
        clients_index_push = clients_index
    print("clients_index_pull")
    print(clients_index_pull)
    print("clients_index_push")
    print(clients_index_push)
    print('every participant start updating their exemplar set and old model...')
    if "fcil" in args.method:
        participant_exemplar_storing_fcil(models, num_clients, model_g, old_client_0, task_id, clients_index)
    
    print('updating finishes')
    print('federated aggregation...')
    
    if "fcil" in args.method:
        w_g_new = FedAvg_weit(w_local, w_g_last, num_samples_list, args.global_weight, clients_index, class_distribution_client, taskid_local, models, task_id, clients_index_pull, args.num_clients, global_task_id_real)
    
    model_g.load_state_dict(w_g_new)

    if "fcil" in args.method:
        proxy_server.model = copy.deepcopy(model_g)
        proxy_server.dataloader(pool_grad, new_task)
    
    # 存在task switch的情况
    if (ep_g + 1) % args.tasks_global == 0 and "direct" not in args.method and "notran" not in args.method:
        acc_global_list = []
        model_for_eval = None
        print(f"ep_g: {ep_g}")
        for i in global_trained_task_id_nosame:
            if "full" not in args.method:
                current_class = class_distribution_client[int(i % args.num_clients)][int(i // args.num_clients)]
            classes_list = []
            for j in current_class:
                classes_list.append(class_real[j])
            current_class = classes_list
            print(f"curren_class: {current_class}")
            if "full" not in args.method:
                current_class_real = class_distribution_client_real[int(i % args.num_clients)][int(i // args.num_clients)]
                client_index = int(i % args.num_clients)
                model_task_id = int(i // args.num_clients)

            client_class_min_output = sorted(list(set(list(range(args.numclass)))-set(current_class)))
            client_class_max_output = current_class

            if "extension" not in args.method and "extencl" not in args.method:
                if client_index in clients_index_push:
                    model_for_eval = copy.deepcopy(model_g)
                else:
                    model_for_eval = copy.deepcopy(models[client_index].model)
            model_for_eval.client_index = client_index
            
            model_for_eval.client_class_min_output = client_class_min_output
            model_for_eval.client_class_max_output = client_class_max_output

            print(f"Not privacy, centralized, but {args.method}")
            acc_global, accs_global = model_global_eval_hard(model_for_eval, test_dataset, model_task_id, args.task_size, args.device, args.method, int(args.epochs_global / args.tasks_global), current_class, current_class_real)
            acc_global_list.append(acc_global)
            log_str = 'Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'\
                .format(client_index, model_task_id, ep_g, acc_global, accs_global)
            out_file.write(log_str + '\n')
            out_file.flush()
            print('powder Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'.format(client_index, model_task_id, ep_g, acc_global, accs_global))
        del model_for_eval
        nni.report_intermediate_result({'default': sum(acc_global_list) / len(acc_global_list)})
    
    old_task_id = task_id
    if "notran" in args.method or "direct" in args.method:
        model_g.load_state_dict(w_g_not_trained)

print(sum(acc_global_list) / len(acc_global_list))
print(type(sum(acc_global_list) / len(acc_global_list)))
nni.report_final_result(float(sum(acc_global_list) / len(acc_global_list)))
            
# classes_learned = args.task_size
# print(f"Initial, classes_learned: {classes_learned}")
# old_task_id = -1
# for ep_g in range(args.epochs_global):  # 100
#     pool_grad = []
#     model_old = proxy_server.model_back()
#     task_id = ep_g // args.tasks_global # task_global: 10

#     print(f"ep_g: {ep_g}, task_id: {task_id}, old_task_id: {old_task_id}")
#     if task_id != old_task_id and old_task_id != -1:
#         overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
#         new_client = [i for i in range(overall_client, overall_client + args.task_size)]
#         old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9)) # 90%从旧任务到新任务的迁移
#         old_client_0 = [i for i in range(overall_client) if i not in old_client_1]  # 10%保持旧任务
#         num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
#         # print(old_client_0)
#         print(f"old_client_0 :{old_client_0}\nold_client_1: {old_client_1}\nnew_client: {new_client}")

#     if task_id != old_task_id and old_task_id != -1:
#         classes_learned += args.task_size
#         print(f"classes_learned: {classes_learned}")
#         model_g.Incremental_learning(classes_learned)
#         model_g = model_to_device(model_g, False, args.device)
    
#     print('federated global round: {}, task_id: {}'.format(ep_g, task_id))

#     w_local = []
#     clients_index = random.sample(range(num_clients), args.local_clients)   # 10
#     print('select part of clients to conduct local training')
#     # print(clients_index)
#     print('clients_index:', clients_index)

#     for c in clients_index:
#         local_model, proto_grad = local_train(models, c, model_g, task_id, model_old, ep_g, old_client_0)
#         w_local.append(local_model)
#         if proto_grad != None:
#             for grad_i in proto_grad:
#                 pool_grad.append(grad_i)

#     ## every participant save their current training data as exemplar set
#     print('every participant start updating their exemplar set and old model...')
#     participant_exemplar_storing(models, num_clients, model_g, old_client_0, task_id, clients_index)
#     print('updating finishes')

#     print('federated aggregation...')
#     w_g_new = FedAvg(w_local)
#     w_g_last = copy.deepcopy(model_g.state_dict())
    
#     model_g.load_state_dict(w_g_new)

#     proxy_server.model = copy.deepcopy(model_g)
#     proxy_server.dataloader(pool_grad)

#     acc_global = model_global_eval(model_g, test_dataset, task_id, args.task_size, args.device)
#     log_str = 'Task: {}, Round: {} Accuracy = {:.2f}%'.format(task_id, ep_g, acc_global)
#     out_file.write(log_str + '\n')
#     out_file.flush()
#     print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, acc_global))

#     old_task_id = task_id
