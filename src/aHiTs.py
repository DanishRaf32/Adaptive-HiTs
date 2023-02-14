import torch
import numpy as np
from scipy import integrate, interpolate
import os
import sys
import time
module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ResNet import *
import ResNet as net
from aHiTs_new import *
criterion = torch.nn.MSELoss(reduction='none')

#Offline aHiTs
def adaptive_multi_scale_forecast(val_data, n_steps, models, best_mse, dt, step_sizes):
    """
    :param x_init: initial state torch array of shape n_test x n_dim
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :return: a torch array of size n_test x n_steps x n_dim,
             a list of indices that are not achieved by interpolations
    """
    # sort models by their step sizes (decreasing order)
    step_sizes = [model.step_size for model in models]
    models_dec = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]
    x_init = torch.tensor(val_data[:, 0 ,:]).float()
    # we assume models are sorted by their step sizes (decreasing order)
    n_test, n_dim = x_init.shape
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    extended_n_steps = n_steps + models_dec[0].step_size
    preds = torch.zeros(n_test, extended_n_steps + 1, n_dim).float().to(device)
    y_preds = torch.zeros(n_test, extended_n_steps+1, n_dim).float().to(device)
    total_step_sizes = n_steps
    preds[:, 0, :] = x_init
    y_prev = preds[:, 0, :]
    y_preds[:, 0, :] = preds[:, 0, :]
    points = int(total_step_sizes)
    stepsize_model = 0
    steps_used=[]
    indices = 0
    index=[]
    index.append(0)
    P=models_dec[len(models_dec)-1].step_size
    while stepsize_model < points:
        for model in models_dec:
            y_next = model(y_prev)
            mse1 = criterion(torch.tensor(y_prev).float(), y_next).mean().item()
            if mse1 < best_mse:
                cur_indices = model.step_size
                steps_used.append(cur_indices)
                indices += cur_indices
                # if indices >= n_steps: #check if indices exceeds the number of time-steps used
                #     break
                index.append(indices)
                y_preds[:, indices, :] = y_next  # .reshape(n_test, -1, n_dim)
                y_prev = y_next
                stepsize_model += model.step_size
                #print(model.step_size)  # Print t
                break
            if model.step_size == P:
                cur_indices = model.step_size
                steps_used.append(cur_indices)
                indices += cur_indices
                # if indices >= n_steps:
                #     break
                index.append(indices)
                y_preds[:,indices, :] = y_next  # .reshape(n_test, -1, n_dim)
                y_prev = y_next
                stepsize_model += model.step_size
    sample_steps = range(0, index[-1])
        # valid_preds = preds[:, indices, :].cpu().detach().numpy()
    # #interpolation
    values=y_preds.cpu().detach().numpy()
    val=np.zeros((n_test, len(index), n_dim))
    j = 0
    for i in index:
        val[:, j, :]=values[:, i, :]
        j=j+1

    x=index
    y=val
    cs = interpolate.interp1d(x, y, kind='linear', axis=1)
    y_preds = torch.tensor(cs(sample_steps)).float()
    return steps_used, index, indices

# Online aHiTs

def adaptive_multi_scale_online(val_data, test_data, n_steps, models, dt, step_sizes, steps_used, index):
    #piece-wise vectorized simulation
    count=1
    iid=[]
    iid.append(0)
    valid_preds=torch.empty(test_data.shape[0],val_data.shape[1],val_data.shape[2])
    valid_preds[:,0,:]=torch.tensor(test_data).float()
    step_update=0
    total_time=0
    k = 0
    def closest(lst, K):
        lst = np.asarray(lst)
        idx = (np.abs(lst - K)).argmin()
        return lst[idx]


    for i in range(0,(len(steps_used)-1)):
        x = steps_used[i]
        if steps_used[i+1] == x:
            count=count+1
        else:
            n_steps_init = (count * steps_used[i-1])-1
            step_update = step_update + n_steps_init
            iid.append(step_update)
            start= step_sizes.index(steps_used[i-1])
            end = step_sizes.index(closest(step_sizes,n_steps_init))+1
            models_init = models[start:end]
            ic = iid[k] + 1
            best_mse=1e+5
            for j in range(len(models_init)):
                y_preds_init = net.vectorized_multi_scale_forecast(torch.tensor(val_data[:, iid[k], :]).float(), n_steps=n_steps_init,
                                                              models=models_init[:len(models_init) - j])
                mse = criterion(torch.tensor(val_data[:, ic:(n_steps_init+ic), :]).float(), y_preds_init).mean().item()
                if mse <= best_mse:
                    end_idx = len(models_init) - j
                    best_mse = mse


            if count == 1:                #check if a step is used only once
                models_init = [models[step_sizes.index(x)]]
            else:
                models_init = models_init[:end_idx]

            start_time= time.time()

            # predictions (online step on testing data)
            y_pred_aHiTs= net.vectorized_multi_scale_forecast(torch.tensor(valid_preds[:, iid[k], :]).float(), n_steps=n_steps_init,
                                                models=models_init)
            online_time = time.time() - start_time


            #update variables
            k=k+1
            count=1

            valid_preds[:,(ic):(n_steps_init+(ic)), :] = y_pred_aHiTs
            total_time = total_time + online_time

        # check for termination
        if i+1 == len(steps_used)-1:
            n_steps_init = (count * steps_used[i])-1
            # step_update = step_update + n_steps_init
            # iid.append(step_update)
            start = step_sizes.index(steps_used[i])
            end = step_sizes.index(closest(step_sizes, n_steps_init))+1
            models_init = models[start:end]
            ic = iid[k]+1
            best_mse = 1e+5
            for j in range(len(models_init)):
                y_preds_init = net.vectorized_multi_scale_forecast(torch.tensor(val_data[:, iid[k], :]).float(),
                                                                   n_steps=((n_steps+1)-ic),
                                                                   models=models_init[:len(models_init) - j])

                mse = criterion(torch.tensor(val_data[:, ic:, :]).float(), y_preds_init).mean().item()
                if mse <= best_mse:
                    end_idx = len(models_init) - j
                    best_mse = mse

            models_init = models_init[:end_idx]
            start_time = time.time()
            # predictions
            y_pred_aHiTs = net.vectorized_multi_scale_forecast(torch.tensor(valid_preds[:, iid[k], :]).float(),
                                                               n_steps=((n_steps+1)-ic),
                                                               models=models_init)
            online_time =  time.time() - start_time


            #update prediction and time
            valid_preds[:,ic:, :] = y_pred_aHiTs
            total_time = total_time + online_time
            break
    return valid_preds, total_time











