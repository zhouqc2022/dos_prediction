import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import cnn_processing
import cnn_model
##Matdeeplearn imports
# from matdeeplearn import models
# import matdeeplearn.process as process
# import matdeeplearn.training as training
# from matdeeplearn.models.utils import model_summary


def train(model, optimizer, loader, loss_method, loss_features, rank):   #loader为train_loader
    model.train()
    loss = 0
    '''28个数据，batch_size = 10 , 所以loader里有三个data'''
    for data in loader:
        data = data.to(rank)  #确保数据和模型在相同的设备上进行计算
        optimizer.zero_grad()  #将模型参数的梯度归零
        dos_out = model(data)
        dos = torch.sum(dos_out, dim=0)
        real_dos = torch.sum(data.y, dim=0)
        loss_one = getattr(F, loss_method)(dos, real_dos)
        loss += loss_one
    loss = loss / len(loader)
    return loss


##Evaluation step, runs model in eval mode
def evaluate(loader, model, loss_method, loss_features, rank, out=False, subset=None):
    model.eval() #model.train()
    loss = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(rank)
 #指示 PyTorch 不要跟踪在其上下文中发生的操作，以减少内存消耗并加快运行速度
            dos_out = model(data)
            dos = torch.sum(dos_out, dim = 0)
            real_dos = torch.sum(data.y, dim=0)
            loss_one = getattr(F, loss_method)(dos, real_dos)
            loss += loss_one.item()
    loss = loss / len(loader)
    loss = loss / 600
    return loss

'''trainer最后返回的是model'''
def trainer(rank,world_size,model,optimizer,scheduler,loss,features_loss,
    train_loader,val_loader,train_sampler,epochs,verbosity,filename = "my_model_temp.pth",):
    train_error  = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()   #获取当前的时间戳
    best_val_error = 1e10
    model_best = model

    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):
        lr = scheduler.optimizer.param_groups[0]["lr"]  #获取第一个参数组的学习率
        train_error = train(model, optimizer, train_loader, loss, features_loss, rank=rank)
        if rank not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0) #将 train_error 在所有进程中进行归约操作（通常是求和），并将结果发送到指定的目标进程
            train_error = train_error / world_size #取平均值，以确保 train_error 变量中包含的值在每个进程上都被平均分配。
        val_error= evaluate(val_loader, model, loss, features_loss, rank=rank, out=False)

        #remenber the best val error and save model and checkpoint
        if val_error == float("NaN") or val_error < best_val_error:
            model_best = copy.deepcopy(model)  #复制整个模型的组件，包括各个层、激活函数等
            torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'full_model': model,
                }, filename,)  #用于保存模型训练的状态信息以及相关参数，以便稍后能够重新加载并继续训练模型
            best_val_error = min(val_error, best_val_error)
            print('model is updated')
        scheduler.step(train_error) #调整学习率
        print('epoch {} is finished， train_error is {}'.format(epoch, train_error))
    return model_best

##Pytorch model setup
def model_setup(rank,model_name,model_params,dataset,load_model=False,model_path=None,print_model=True,):
    model = getattr(cnn_model, model_name)(
        data = dataset, **(model_params if model_params is not None else {})
    ).to(rank)   #创建了一个模型的实例，该模型来自于 cnn_model 模块，然后将数据集 dataset 作为输入，并应用了模型参数（如果提供的话）。最后，模型被移动到指定的设备 rank 上
    if load_model == "True":
        assert os.path.exists(model_path), "Saved model not found"
        if str(rank) in ("cpu"):
            saved = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            saved = torch.load(model_path)
        model.load_state_dict(saved["state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])
    if print_model == True and rank in (0, "cpu", "cuda"):
        model_summary(model)
    return model
##Pytorch loader setup
def loader_setup(train_ratio,val_ratio,test_ratio,batch_size,dataset,rank,seed,world_size=0,num_workers=0,):
    ##Split datasets
    train_dataset, val_dataset, test_dataset = cnn_processing.split_data(
        dataset, train_ratio, val_ratio, test_ratio, seed)
    train_sampler = None
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # may scale down batch size if memory is an issue

    if len(val_dataset) > 0:
        val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    if len(test_dataset) > 0:
        test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return train_loader, val_loader, test_loader, train_sampler, train_dataset,val_dataset, test_dataset
#return (train_loader, val_loader, test_loader, train_sampler, train_dataset,val_dataset, test_dataset,)

################################################################################
#  Trainers
################################################################################
'''rank在cpu和cuda中则不是分布式训练'''
###Regular training with train, val, test split
def train_regular(rank,world_size, data_path, job_parameters=None, training_parameters=None, model_parameters=None,):
    # model_parameters["lr"] = model_parameters["lr"] * world_size
    print(model_parameters)
    dataset = cnn_processing.get_dataset(data_path, training_parameters["target_index"], False)
    print(dataset)
    train_loader, val_loader, test_loader, train_sampler, train_dataset, val_dataset, test_dataset = loader_setup(
        training_parameters["train_ratio"],
        training_parameters["val_ratio"],
        training_parameters["test_ratio"],
        model_parameters["batch_size"],
        dataset,
        rank,
        job_parameters["seed"],
        world_size,)
    ##Set up model
    model = model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        dataset,
        job_parameters["load_model"],
        job_parameters["model_path"],
        model_parameters.get("print_model", True),    #获得键print_model对应的值，如果不存在就取为True
    )
    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"])
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"])

    ##在trainer中进行训练
    model = trainer(
        rank,
        world_size,
        model,
        optimizer,
        scheduler,
        training_parameters["loss"],
        training_parameters["features_loss"],
        train_loader,
        val_loader,
        train_sampler,
        model_parameters["epochs"],
        training_parameters["verbosity"],
        "my_model_temp.pth",
    )

    if rank in (0, "cpu", "cuda"):
        # train_error = val_error = test_error =  float("NaN")
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=model_parameters["batch_size"],
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=True,)
        train_error = evaluate(train_loader, model, training_parameters["loss"], training_parameters["features_loss"], rank, out=False)
        # Get val error
        val_error = evaluate(val_loader, model, training_parameters["loss"], training_parameters["features_loss"], rank, out=False)
        ##Get test error
        test_error = evaluate(test_loader, model, training_parameters["loss"], training_parameters["features_loss"], rank, out=True,subset=8000)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "full_model": model,
        },
        job_parameters["model_path"],)
    print('train_error is {}, test_error is {}, val_error is {}'.format(float(train_error) ,float(val_error), float(test_error)))


    for data in val_loader:
        data = data.to(rank)
            # 指示 PyTorch 不要跟踪在其上下文中发生的操作，以减少内存消耗并加快运行速度
        dos_out = model(data)
        y = data.y
 # 使用索引作为 x 值
        print(len(y))
        print(y.shape)
        print(dos_out.shape)

    # 绘制图表
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(x_values, dos_out[:200], label='prediction')
    #     plt.plot(x_values, y[:200], label='true')
    #     plt.xlabel('Sample Index')
    #     plt.ylabel('Length')
    #     plt.title('Length Comparison')
    #     plt.legend()
    #     plt.show()





    return test_error

def model_summary(model):
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("--------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


def predict(dataset, loss_method, job_parameters=None):
    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##Loads predict dataset in one go, care needed for large datasets)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ##Load saved model
    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cpu")
        )
    else:
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cuda")
        )
    model = saved["full_model"]
    model = model.to(rank)
    model_summary(model)
    ##Get predictions
    time_start = time.time()
    model.eval()
    count = 0
    for data in loader:
        data = data.to(rank)
        with torch.no_grad():
            dos_out = model(data[0])
            dos = torch.sum(dos_out, dim=0)
            real_dos = torch.sum(data[0].y, dim=0)
            loss = getattr(F, loss_method)(dos, real_dos )
        dos_np = dos.numpy()
        real_dos_np = real_dos.numpy()
        sequence = [i for i in range(len(dos_np))]
        plt.scatter(sequence, real_dos_np, label='Target', s=10)
        plt.scatter(sequence, dos_np, label='Prediction', s=10)
        plt.show()