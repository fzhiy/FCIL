import torch.nn as nn
import torch
import torch.nn.functional as F

class network_hard(nn.Module):

    def __init__(self, numclass, feature_extractor, class_distribution, class_distribution_real=None, class_distribution_proportion=None, args=None):
        super(network_hard, self).__init__()
        self.args = args
        self.feature = feature_extractor
        
        self.numclass = numclass
        # self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        
        self.fc = nn.Linear(768, numclass, bias=True)
        self.client_fc = nn.Linear(768, self.args.class_per_task, bias=True)
        self.aggregate_weight = torch.nn.Parameter(torch.FloatTensor(int(self.args.numclass/self.args.class_per_task), int(self.args.numclass/self.args.class_per_task)).uniform_(1, 1), requires_grad=True)
        self.class_distribution = class_distribution
        self.class_distribution_real = class_distribution_real
        self.class_distribution_proportion = class_distribution_proportion
        self.task_id = 0
        self.client_index = -1
        #self.client_class_output = []
        self.client_class_min_output = []
        self.global_class_min_output = [] #This is for evaluation of global model
        self.current_class = []
        self.trained_task_id = None
        self.learned_class = None
        self.unlearned_class = None
        self.ep_g = 0

    def forward(self, input):
        # x = self.feature(input)
        client_global_task_id = self.task_id * self.args.num_clients + self.client_index
        # print(f"client_global_task_id: {client_global_task_id}")
        if self.client_index == -1:
            x, _, _, _ = self.feature(input)
            x = x[:,0,:]
            x = self.fc(x)
        else:
            #TODO: 'sharedcodap' in self.args.method to be implemented
            x, _, _, _ = self.feature(input)
            feature = x[:,0,:]
            # print(f'x.shape: {x.shape}\nfeature.shape: {feature.shape}')
            # x.shape: torch.Size([32, 197, 768]) feature.shape: torch.Size([32, 768])
            x = self.fc(feature)
            x[:,self.client_class_min_output] = -float('inf')

        return x

    def Incremental_learning(self, task_id):
        print(f"[func incremental learning] task_id: {task_id}")
        self.task_id = task_id
    # def Incremental_learning(self, numclass):
    #     weight = self.fc.weight.data
    #     bias = self.fc.bias.data
    #     in_feature = self.fc.in_features
    #     out_feature = self.fc.out_features

    #     self.fc = nn.Linear(in_feature, numclass, bias=True)
    #     self.fc.weight.data[:out_feature] = weight
    #     self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        client_global_task_id = self.task_id * self.args.num_clients + self.client_index
        if self.client_index == -1:
            x, _, _, _ = self.feature(input)
            x = x[:,0,:]
        else:
            if "sharedfc" in self.args.method:
                x, _, _, _ = self.feature(input)
                x = x[:,0,:]
            elif "sharedencoder"  in self.args.method:
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x, _, _, _ = self.feature[i](input)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x, _, _, _ = self.feature[i](input)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x, _, _, _ = self.feature[self.task_id * self.args.num_clients + self.client_index](input)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                else:
                    x, _, _, _ = self.feature[-1](input)
                    x = x[:,0,:]
            elif "sharedprompt" in self.args.method:
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x = self.feature.forward_sharedprompt(input, self.prompt, i, client_global_task_id=client_global_task_id)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x = self.feature.forward_sharedprompt(input, self.prompt, i, client_global_task_id=client_global_task_id)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x = self.feature.forward_sharedprompt(input, self.prompt, self.task_id * self.args.num_clients + self.client_index, client_global_task_id=client_global_task_id)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                else:
                    x = self.feature.forward_sharedprompt(input, self.prompt, -1, client_global_task_id=client_global_task_id)
                    x = x[:,0,:]
            elif "sharedcodap" in self.args.method:
                with torch.no_grad():
                    q, _, _, _ = self.feature(input)
                    q = q[:,0,:]
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, i, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, i, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, self.task_id * self.args.num_clients + self.client_index, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                else:
                    x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, -1, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                    x = x[:,0,:]
            else:
                x, _, _, _ = self.feature(input)
                x = x[:,0,:]
        return x

        # if self.client_index == -1:
        #     x, _, _, _ = self.feature(input)
        #     x = x[:,0,:]
        # else:
        #     # TODO: 'sharedcodap' in self.args.method to be implemented
        #     x, _, _, _ = self.feature(input)
        #     x = x[:,0,:]
        # return x
        # return self.feature(inputs)

    def predict(self, fea_input):
        if self.client_index == -1:
            x = self.fc(fea_input)
            x[:,self.global_class_min_output] = -float('inf')
        else:
            # TODO: 'sharedcodap' in self.args.method to be implemented
            x = self.fc(fea_input)
            x[:,self.client_class_min_output] = -float('inf')
        print(f"self.client_class_min_output: {self.client_class_min_output}")
        print(f"x.shape: {x.shape}")
        return x
        # return self.fc(fea_input)

    def set_learned_unlearned_class(self, learned_class):
        self.learned_class = learned_class
        self.unlearned_class = sorted(list(set(list(range(self.numclass))) - set(learned_class)))

class LeNet_hard(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet_hard, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())
