1  #!/usr/bin/env python
2  # _*_ coding: utf-8 _*_
3  # @Time : 2020/4/17 10:02 
4  # @Author :Jiwei Tian
5  # @Version：V 0.1
6  # @File : attack.py
7  # @desc :
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import logging
logging.basicConfig(level=logging.DEBUG, filename='test.log', format="%(levelname)s:%(asctime)s:%(message)s")
from itertools import combinations,permutations
from scipy.special import comb, perm
from differential_evolution import differential_evolution
from torch.optim import SGD
from sklearn.metrics import accuracy_score
from numpy import vstack
from main_UAE import evaluate_all_data
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs.to(device))
        yhat= torch.max(yhat,1)[1]
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy().reshape(-1,1)
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

def project_perturbation(data_point,p,perturbation):

    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
    elif p == np.inf:
        perturbation = torch.sign(perturbation) * torch.min(torch.abs(perturbation), data_point)
    return perturbation


def UAE_train(train_dl,model,H,state_length,train_attack_data):
    model.eval()
    model.cuda()
    acc_ori = evaluate_all_data(model, train_attack_data.X, train_attack_data.y)

    H = torch.from_numpy(H).float()
    attack_state = torch.zeros(state_length, 1, requires_grad=True)
    optimizer = optim.Adam([attack_state], lr=0.0001)

    # define the optimization
    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    t0 = time.time()
    acc_best = np.load("14bus_acc_best.npy")
    for epoch in range(1000):
        # enumerate mini batches
        if epoch % 10 == 0:
            print("time:", time.time()-t0)
        print('Epoch: %.0f' % epoch)
        logging.info('Epoch: %.0f', epoch)
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model((inputs+H.mm(attack_state).t()).cuda())
            # calculate loss
            targets = targets.squeeze()
            loss = -torch.mean(criterion(yhat, targets.long().cuda()))
            # credit assignment
            loss.backward(retain_graph=True)
            # update model weights
            optimizer.step()
            #attack_state = project_perturbation(torch.from_numpy(np.array([0.5]).astype(np.float32)), np.inf, attack_state)
            #attack_state = torch.clamp(attack_state, -0.5, 0.5)
            if i%50 == 0:
                final_attack_state = torch.clamp(attack_state, -1, 1)
                final_attack_vector = H.mm(final_attack_state).t().detach().numpy()
                new_attack_data = train_attack_data.X + H.mm(final_attack_state).t().detach().numpy()
                acc = evaluate_all_data(model,new_attack_data, train_attack_data.y)
                print('Accuracy: %.3f' % acc)
                if acc<=acc_best:
                    acc_best = acc
                    np.save("118bus_acc_best.npy",acc_best)
                    np.save("118bus_uni_attack_state.npy",final_attack_state.detach().numpy())
                    np.save("118bus_uni_attack_vector.npy",final_attack_vector)
                    ccc = np.load("118bus_uni_attack_state.npy")
    return attack_state


def UAE_train2(train_dl,model,H,state_length,train_attack_data,bus_number,limit):
    model.eval()
    model.cuda()
    acc_ori = evaluate_all_data(model, train_attack_data.X, train_attack_data.y)

    H = torch.from_numpy(H).float()
    attack_state = torch.zeros(state_length, 1, requires_grad=True)
    optimizer = optim.Adam([attack_state], lr=0.0001)

    # define the optimization
    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    t0 = time.time()
    acc_best = 1.0
    #acc_best = np.load("{}bus_acc_best.npy".format(bus_number))
    for epoch in range(10000):
        # enumerate mini batches
        if epoch % 10 == 0:
            print("time:", time.time()-t0)
            print('Epoch: %.0f' % epoch)
            print('Best acc: %.3f' % acc_best)
        logging.info('Epoch: %.0f', epoch)
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            attack_state.requires_grad = True
            #optimizer.zero_grad()
            # compute the model output
            yhat = model((inputs+H.mm(attack_state).t()).cuda())
            # calculate loss
            targets = targets.squeeze()
            loss = -torch.mean(criterion(yhat, targets.long().cuda()))
            # credit assignment
            loss.backward()
            # update model weights
            #optimizer.step()
            #attack_state = project_perturbation(torch.from_numpy(np.array([0.5]).astype(np.float32)), np.inf, attack_state)
            temp_attack_state = attack_state - 0.001*attack_state.grad.sign()
            attack_state = torch.clamp(temp_attack_state, -limit,limit).detach()
        if epoch%10 == 0:
            final_attack_state = torch.clamp(attack_state, -limit,limit)
            final_attack_vector = H.mm(final_attack_state).t().detach().numpy()
            new_attack_data = train_attack_data.X + H.mm(final_attack_state).t().detach().numpy()
            acc = evaluate_all_data(model,new_attack_data, train_attack_data.y)
            print('Accuracy: %.3f' % acc)

            if acc <= acc_best:
                acc_best = acc
                np.save("{}bus_{}/{}bus_acc_best.npy".format(bus_number, limit, bus_number), acc_best)
                np.save("{}bus_{}/{}bus_uni_attack_state_best.npy".format(bus_number, limit, bus_number),
                        final_attack_state.detach().numpy())
                np.save("{}bus_{}/{}bus_uni_attack_vector_best.npy".format(bus_number, limit, bus_number),
                        final_attack_vector)
                # ccc = np.load("118bus_uni_attack_state.npy")

            if epoch % 100 == 0:
                np.save("{}bus_{}/{}bus_acc_{}.npy".format(bus_number, limit, bus_number, epoch), acc)
                np.save("{}bus_{}/{}bus_uni_attack_state_{}.npy".format(bus_number, limit, bus_number, epoch),
                        final_attack_state.detach().numpy())
                np.save("{}bus_{}/{}bus_uni_attack_vector_{}.npy".format(bus_number, limit, bus_number, epoch),
                        final_attack_vector)
    np.save("{}bus_{}/{}bus_time_10000epoch.npy".format(bus_number, limit, bus_number), time.time() - t0)


    return attack_state

def C_cw_l2_attack(model, inputs, attack_vector,attack_state,H,meter_length, state_length, binary_search=5, kappa=0, max_iter=10000, learning_rate=0.01):
    attack_vector = attack_vector.reshape(meter_length, 1)
    attack_vector = torch.from_numpy(attack_vector)

    inputs= inputs.reshape(meter_length,1)
    inputs = torch.from_numpy(inputs)

    norm_c = np.linalg.norm(attack_state) * 2

    H = torch.from_numpy(H).float()

    attack_state.reshape(state_length, 1)
    attack_state = torch.from_numpy(attack_state).float().resize(state_length,1)

    DMIN = np.inf
    CONST = 0.5
    CONST_max = 100
    CONST_min = 0

    # Define f-function
    def f(x):
        x = x.t()
        outputs = model(x).squeeze()
        return torch.clamp(outputs[1] - outputs[0], min=-kappa)

    PERT_ALL = torch.zeros_like(inputs,requires_grad=False)
    PERT_state = torch.zeros_like(attack_state, requires_grad=False)

    for m in range(binary_search):

        zero = torch.zeros(state_length,1, requires_grad=False)
        aaa = torch.zeros(state_length,1, requires_grad=True)

        alpha = torch.zeros(state_length,1, requires_grad=False)
        alpha[attack_state==0]=0
        alpha[attack_state!=0]=1

        optimizer = optim.Adam([aaa], lr=learning_rate)

        dmin = np.inf
        pert = torch.zeros(meter_length,1,requires_grad=False)
        pert_state = torch.zeros(state_length,1, requires_grad=False)

        for step in range(max_iter):
            # loss1 = nn.MSELoss(reduction='sum')(alpha.mul(aaa), zero)
            loss1 = (nn.MSELoss(reduction='sum')(alpha.mul(aaa)+attack_state, zero) - norm_c).pow(2)
            #loss1 = (nn.MSELoss(reduction='sum')(alpha.mul(aaa), zero)-norm_c).pow(2)
            loss2 = CONST * f(inputs+H.mm(alpha.mul(aaa)))
            cost = loss1 + loss2
            if loss2 == 0 and loss1 <= dmin:
                pert = H.mm(alpha.mul(aaa))
                dmin = loss1
                pert_state = alpha.mul(aaa)

            optimizer.zero_grad()
            cost.backward(retain_graph=True)
            optimizer.step()


        if f(inputs+pert) == 0 and dmin <= DMIN:
            PERT_ALL = pert
            DMIN = dmin
            PERT_state = pert_state

        if f(inputs+pert)==0:
            CONST_max = CONST
        else:
            CONST_min = CONST
        CONST = (CONST_min + CONST_max) / 2


    return PERT_state, PERT_ALL+attack_vector,PERT_ALL, PERT_ALL+inputs, DMIN,f(PERT_ALL+inputs),model((PERT_ALL+inputs).t())

def  T_C_attack(model, inputs, attack_state,H,meter_length,state_length,kappa=0, max_iter=10000,attack_state_number=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    length = state_length - len(np.nonzero(attack_state)[0])
    length = int(comb(length,attack_state_number))
    # length = int(length * (length-1) *(length-2) *(length-3)/24)
    # length = int(length * (length-1) *(length-2)/6)
    # length = int(length * (length - 1) / 2)

    inputs= inputs.reshape(meter_length,1)
    inputs = torch.from_numpy(inputs)

    H = torch.from_numpy(H).float()

    attack_state.reshape(state_length, 1)
    attack_state = torch.from_numpy(attack_state).float().resize(state_length,1)

    # Define f-function
    def f(x):
        x = x.t().to(device)
        outputs = model(x).squeeze()
        return torch.clamp(outputs[:,1] - outputs[:,0], min=-kappa)

    index = np.argwhere(attack_state == 0)
    COM_2 = list(combinations(index[0], attack_state_number))
    # COM_2 = list(combinations(index[0], 3))
    # COM_2 = list(combinations(index[0], 2))
    # COM_2 = list(combinations(index[0], 1))

    alpha = torch.zeros(state_length, length, requires_grad=False)
    for i in range(length):
        # alpha[index[0, i], i] = 1
        alpha[COM_2[i], i] = 1

    aaa = torch.zeros(state_length,length, requires_grad=True)

    optimizer = optim.Adam([aaa], lr=0.1)

    pert_c = np.zeros((state_length,1))
    pert_a = np.zeros((meter_length,1))

    record = list()
    for step in range(max_iter):
        loss1 = torch.norm(alpha.mul(0.5*torch.tanh(aaa)),p=2,dim=0).to(device)
        loss2 = f(inputs+H.mm(alpha.mul(0.5*torch.tanh(aaa)))).to(device)
        cost = torch.sum(loss2)
        record.append(cost.cpu().detach().numpy())

        state = 0
        if torch.sum(loss2!=0) < length:
            where_0 = torch.where(loss2==0)
            for ii in where_0[0].cuda():
                if loss1[ii]<np.inf:
                    state = 1
                    print("success,iteration:", step)
                    pert_c = alpha.mul(0.5*torch.tanh(aaa))[:,ii].detach().numpy()
                    pert_a = H.mm(alpha.mul(0.5*torch.tanh(aaa)))[:,ii].detach().numpy()
            #     break
            # break

        optimizer.zero_grad()
        cost.backward(retain_graph=True)
        optimizer.step()

    # print("final iteration:", step)

    return record,state,pert_c,pert_a

def T_C_cw_l2_attack(model, inputs, attack_vector,attack_state,H,meter_length, state_length, binary_search=1, kappa=0, max_iter=10000, learning_rate=0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attack_vector = attack_vector.reshape(meter_length, 1)
    attack_vector = torch.from_numpy(attack_vector)

    length = state_length - len(np.nonzero(attack_state)[0])

    # length = int(length * (length-1) *(length-2)/6)
    length = int(length * (length - 1) / 2)

    inputs= inputs.reshape(meter_length,1)
    inputs = torch.from_numpy(inputs)

    norm_c = np.linalg.norm(attack_state) * 2

    H = torch.from_numpy(H).float()

    attack_state.reshape(state_length, 1)
    attack_state = torch.from_numpy(attack_state).float().resize(state_length,1)

    DMIN = np.inf*torch.ones((length,1)).to(device)
    CONST = 0.5*torch.ones((length,1)).to(device)
    CONST_max = 100*torch.ones((length,1)).to(device)
    CONST_min = 0*torch.ones((length,1)).to(device)

    # Define f-function
    def f(x):
        x = x.t().to(device)
        outputs = model(x).squeeze()
        return torch.clamp(outputs[:,1] - outputs[:,0], min=-kappa)

    PERT_ALL = torch.zeros_like(inputs,requires_grad=False)
    PERT_state = torch.zeros_like(attack_state, requires_grad=False)

    index = np.argwhere(attack_state == 0)
    # COM_2 = list(combinations(index[0], 3))
    COM_2 = list(combinations(index[0], 2))


    alpha = torch.zeros(state_length, length, requires_grad=False)
    for i in range(length):
        # alpha[index[0, i], i] = 1
        alpha[COM_2[i], i] = 1

    PERT_c = torch.zeros(state_length, length, requires_grad=False)
    PERT_a = torch.zeros(meter_length, length, requires_grad=False)

    for m in range(binary_search):
        t0 = time.time()
        zero = torch.zeros(state_length,1, requires_grad=False)
        aaa = torch.zeros(state_length,length, requires_grad=True)

        pert_c = torch.zeros(state_length, length, requires_grad=False)
        pert_a = torch.zeros(meter_length, length, requires_grad=False)

        optimizer = optim.Adam([aaa], lr=0.01)
        # optimizer = optim.SGD([aaa], lr=learning_rate, momentum=0.9)

        dmin = np.inf*torch.ones((length,1)).to(device)
        pert = torch.zeros(meter_length,1,requires_grad=False)
        pert_state = torch.zeros(state_length,1, requires_grad=False)

        print("binary search:", m)

        for step in range(max_iter):
            # loss1 = torch.sum(torch.abs(alpha.mul(torch.tanh(aaa))),0).to(device)
            loss1 = torch.norm(alpha.mul(torch.tanh(aaa)),p=2,dim=0).to(device)
            # loss1 = nn.MSELoss(reduction='sum')(alpha.mul(aaa), zero)
            # loss1 = (nn.MSELoss(reduction='sum')(alpha.mul(aaa)+attack_state, zero) - norm_c).pow(2)
            #loss1 = (nn.MSELoss(reduction='sum')(alpha.mul(aaa), zero)-norm_c).pow(2)
            loss2 = f(inputs+H.mm(alpha.mul(torch.tanh(aaa)))).to(device)
            # cost = torch.sum(loss1) + torch.sum(loss2.mul(CONST))
            # cost = torch.sum(loss1+loss2.mul(CONST))
            cost = torch.sum(loss2)

            if torch.sum(loss2!=0) < length:
                where_0 = torch.where(loss2==0)
                # for ii in where_0[0].numpy():
                for ii in where_0[0].cuda():
                    if loss1[ii]<dmin[ii]:
                        pert_c[:,ii] = alpha.mul(torch.tanh(aaa))[:,ii]
                        pert_a[:,ii] = H.mm(alpha.mul(torch.tanh(aaa)))[:,ii]
                        dmin[ii] = loss1[ii]

            optimizer.zero_grad()
            cost.backward(retain_graph=True)
            optimizer.step()
            # aaa = torch.clamp(aaa, -1, 1, out=None)

            if step%1000 == 0:
                print("iteration:", step)
        state = False
        for ii in range(length):
            if dmin[ii] < DMIN[ii]:
                state = True
                print("success")
                PERT_c[:,ii] = pert_c[:,ii]
                PERT_a[:,ii] = pert_a[:,ii]
                DMIN[ii] = dmin[ii]
                CONST_max[ii] = CONST[ii]
            else:
                CONST_min[ii] = CONST[ii]


        # if dmin[ii] < np.inf:
        #     CONST_max[ii] = CONST[ii]
        # else:
        #     CONST_min[ii] = CONST[ii]
        print("one binary search time:", time.time()-t0)

        CONST = (CONST_min + CONST_max) / 2

    return state

def DE1_T_C_cw_l2_attack(model, inputs, attack_state,H,meter_length, state_length,pop_size,max_iter,re_combination):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alpha = torch.zeros(state_length, 1, requires_grad=False)
    alpha[attack_state == 0] = 1
    alpha[attack_state != 0] = 0
    alpha = alpha.int()

    popsize = pop_size

    # Define f-function
    def f(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[:,1] - outputs[:,0], min=-kappa)
        return outputs[:,1] - outputs[:,0]

    def f2(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[1] - outputs[0], min=-kappa)
        return outputs[1] - outputs[0]


    def objective_function(xs):
        where = np.around(xs[:,[0]]).astype(int)
        magnitude = xs[:,[1]]
        belta = np.copy(alpha)
        belta = np.repeat(belta, popsize,axis=1)
        vvv = np.zeros([state_length,popsize])
        for i in range(popsize):
            vvv[where[i,:],i] = 1
        uuu = np.zeros([state_length,popsize])
        for i in range(popsize):
            uuu[where[i,:],i] = magnitude[i,:]
        final = belta * vvv * uuu
        aaa = torch.from_numpy(inputs.reshape(meter_length, 1) + np.matmul(H, final))
        nnn = f(aaa).detach().cpu().numpy().reshape(popsize,-1)
        return nnn

    bounds = [(0,state_length-1),(-1,1)]
    popmul = popsize/len(bounds)

    inits = np.zeros([popsize, len(bounds)])
    for init in inits:
        init[0] = np.random.normal(state_length//2,state_length//2-1)
        init[1] = np.random.random()

    def attack_success(x, inputs):
        where = np.around(x[[0]]).astype(int)
        magnitude = x[[1]]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length,1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True

    callback_fn = lambda x, convergence: attack_success(x, inputs)

    def attack_success_output(x, inputs):
        where = np.around(x[[0]]).astype(int)
        magnitude = x[[1]]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length,1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True, final, np.matmul(H, final)

    attack_result = differential_evolution(objective_function, bounds, maxiter=max_iter, popsize=popmul, mutation=1.8,recombination=re_combination, tol=0.01, atol=-1, callback=callback_fn, polish=False, init=inits)

    if attack_success(attack_result.x,inputs) is True:
        state, pert_c, pert_a = attack_success_output(attack_result.x,inputs)
        print("success")
    else:
        state = False
        pert_c = np.zeros([state_length,1])
        pert_a = np.zeros([meter_length,1])
        print("failed")
    return state, pert_c, pert_a

def DE2_T_C_cw_l2_attack(model, inputs, attack_state,H,meter_length, state_length,pop_size,max_iter,re_combination):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alpha = torch.zeros(state_length, 1, requires_grad=False)
    alpha[attack_state == 0] = 1
    alpha[attack_state != 0] = 0
    alpha = alpha.int()

    popsize = pop_size

    # Define f-function
    def f(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[:,1] - outputs[:,0], min=-kappa)
        return outputs[:,1] - outputs[:,0]

    def f2(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[1] - outputs[0], min=-kappa)
        return outputs[1] - outputs[0]


    def objective_function(xs):
        where = np.around(xs[:, [0, 1]]).astype(int)
        # where = xs[:,[0,1]].astype(int)
        magnitude = xs[:,[2,3]]
        belta = np.copy(alpha)
        belta = np.repeat(belta, popsize,axis=1)
        vvv = np.zeros([state_length,popsize])
        for i in range(popsize):
            vvv[where[i,:],i] = 1
        uuu = np.zeros([state_length,popsize])
        for i in range(popsize):
            uuu[where[i,:],i] = magnitude[i,:]
        final = belta * vvv * uuu
        aaa = torch.from_numpy(inputs.reshape(meter_length, 1) + np.matmul(H, final))
        nnn = f(aaa).detach().cpu().numpy().reshape(popsize,-1)
        return nnn

    bounds = [(0,state_length-1),(0,state_length-1),(-1,1),(-1,1)]
    popmul = popsize/len(bounds)

    inits = np.zeros([popsize, len(bounds)])
    for init in inits:
        init[0] = np.random.normal(state_length//2,state_length//2-1)
        init[1] = np.random.normal(state_length//2,state_length//2-1)
        init[2] = np.random.random()
        init[3] = np.random.random()

    def attack_success(x, inputs):
        where = np.around(x[[0, 1]]).astype(int)
        magnitude = x[[2, 3]]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length,1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True

    callback_fn = lambda x, convergence: attack_success(x, inputs)

    def attack_success_output(x, inputs):
        where = np.around(x[[0, 1]]).astype(int)
        magnitude = x[[2, 3]]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length,1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True, final, np.matmul(H, final)

    attack_result = differential_evolution(objective_function, bounds, maxiter=max_iter, popsize=popmul, mutation=1.8,recombination=re_combination, tol=0.01, atol=-1, callback=callback_fn, polish=False, init=inits)

    if attack_success(attack_result.x,inputs) is True:
        state, pert_c, pert_a = attack_success_output(attack_result.x,inputs)
        print("success")
    else:
        state = False
        pert_c = np.zeros([state_length,1])
        pert_a = np.zeros([meter_length,1])
        print("failed")
    return state, pert_c, pert_a


def DE3_T_C_cw_l2_attack(model, inputs, attack_state, H, meter_length, state_length,pop_size,max_iter,re_combination):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alpha = torch.zeros(state_length, 1, requires_grad=False)
    alpha[attack_state == 0] = 1
    alpha[attack_state != 0] = 0
    alpha = alpha.int()

    popsize = pop_size

    # Define f-function
    def f(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[:,1] - outputs[:,0], min=-kappa)
        return outputs[:, 1] - outputs[:, 0]

    def f2(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[1] - outputs[0], min=-kappa)
        return outputs[1] - outputs[0]

    def objective_function(xs):
        where = np.around(xs[:, [0,1,2]]).astype(int)
        magnitude = xs[:, [3,4,5]]
        belta = np.copy(alpha)
        belta = np.repeat(belta, popsize, axis=1)
        vvv = np.zeros([state_length, popsize])
        for i in range(popsize):
            vvv[where[i, :], i] = 1
        uuu = np.zeros([state_length, popsize])
        for i in range(popsize):
            uuu[where[i, :], i] = magnitude[i, :]
        final = belta * vvv * uuu
        aaa = torch.from_numpy(inputs.reshape(meter_length, 1) + np.matmul(H, final))
        nnn = f(aaa).detach().cpu().numpy().reshape(popsize, -1)
        return nnn

    bounds = [(0, state_length - 1), (0, state_length - 1), (0, state_length - 1),(-1, 1), (-1, 1), (-1, 1)]
    popmul = popsize / len(bounds)

    inits = np.zeros([popsize, len(bounds)])
    for init in inits:
        init[0] = np.random.normal(state_length // 2, state_length // 2 - 1)
        init[1] = np.random.normal(state_length // 2, state_length // 2 - 1)
        init[2] = np.random.normal(state_length // 2, state_length // 2 - 1)
        init[3] = np.random.random()
        init[4] = np.random.random()
        init[5] = np.random.random()

    def attack_success(x, inputs):
        where = np.around(x[[0,1,2]]).astype(int)
        magnitude = x[[3,4,5]]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length, 1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True

    callback_fn = lambda x, convergence: attack_success(x, inputs)

    def attack_success_output(x, inputs):
        where = np.around(x[[0,1,2]]).astype(int)
        magnitude = x[[3,4,5]]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length, 1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True, final, np.matmul(H, final)

    attack_result = differential_evolution(objective_function, bounds, maxiter=max_iter, popsize=popmul, mutation=1.8,
                                           recombination=re_combination, tol=0.01, atol=-1, callback=callback_fn, polish=False,
                                           init=inits)

    if attack_success(attack_result.x, inputs) is True:
        state, pert_c, pert_a = attack_success_output(attack_result.x, inputs)
        print("success")
    else:
        state = False
        pert_c = np.zeros([state_length, 1])
        pert_a = np.zeros([meter_length, 1])
        print("failed")
    return state, pert_c, pert_a

def DE4_T_C_cw_l2_attack(model, inputs, attack_state, H, meter_length, state_length,pop_size,max_iter,re_combination):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alpha = torch.zeros(state_length, 1, requires_grad=False)
    alpha[attack_state == 0] = 1
    alpha[attack_state != 0] = 0
    alpha = alpha.int()

    popsize = pop_size

    # Define f-function
    def f(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[:,1] - outputs[:,0], min=-kappa)
        return outputs[:, 1] - outputs[:, 0]

    def f2(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[1] - outputs[0], min=-kappa)
        return outputs[1] - outputs[0]

    def objective_function(xs):
        where = np.around(xs[:, [0,1,2,3]]).astype(int)
        magnitude = xs[:, [4,5,6,7]]
        belta = np.copy(alpha)
        belta = np.repeat(belta, popsize, axis=1)
        vvv = np.zeros([state_length, popsize])
        for i in range(popsize):
            vvv[where[i, :], i] = 1
        uuu = np.zeros([state_length, popsize])
        for i in range(popsize):
            uuu[where[i, :], i] = magnitude[i, :]
        final = belta * vvv * uuu
        aaa = torch.from_numpy(inputs.reshape(meter_length, 1) + np.matmul(H, final))
        nnn = f(aaa).detach().cpu().numpy().reshape(popsize, -1)
        return nnn

    bounds = [(0, state_length - 1), (0, state_length - 1), (0, state_length - 1),(0, state_length - 1),(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    popmul = popsize / len(bounds)

    inits = np.zeros([popsize, len(bounds)])
    for init in inits:
        init[0] = np.random.normal(state_length // 2, state_length // 2 - 1)
        init[1] = np.random.normal(state_length // 2, state_length // 2 - 1)
        init[2] = np.random.normal(state_length // 2, state_length // 2 - 1)
        init[3] = np.random.normal(state_length // 2, state_length // 2 - 1)
        init[4] = np.random.random()
        init[5] = np.random.random()
        init[6] = np.random.random()
        init[7] = np.random.random()

    def attack_success(x, inputs):
        where = np.around(x[[0,1,2,3]]).astype(int)
        magnitude = x[[4,5,6,7]]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length, 1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True

    callback_fn = lambda x, convergence: attack_success(x, inputs)

    def attack_success_output(x, inputs):
        where = np.around(x[[0,1,2,3]]).astype(int)
        magnitude = x[[4,5,6,7]]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length, 1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True, final, np.matmul(H, final)

    attack_result = differential_evolution(objective_function, bounds, maxiter=max_iter, popsize=popmul, mutation=1.8,
                                           recombination=re_combination, tol=0.01, atol=-1, callback=callback_fn, polish=False,
                                           init=inits)

    if attack_success(attack_result.x, inputs) is True:
        state, pert_c, pert_a = attack_success_output(attack_result.x, inputs)
        print("success")
    else:
        state = False
        pert_c = np.zeros([state_length, 1])
        pert_a = np.zeros([meter_length, 1])
        print("failed")
    return state, pert_c, pert_a

def DE_T_C_cw_l2_attack(model, inputs, attack_state, H, meter_length, state_length,pop_size,max_iter,re_combination,target_number):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    where_index = range(target_number)
    magnitude_index = range(target_number,2*target_number)

    alpha = torch.zeros(state_length, 1, requires_grad=False)
    alpha[attack_state == 0] = 1
    alpha[attack_state != 0] = 0
    alpha = alpha.int()

    popsize = pop_size

    # Define f-function
    def f(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[:,1] - outputs[:,0], min=-kappa)
        return outputs[:, 1] - outputs[:, 0]

    def f2(x):
        x = x.t().to(device)
        outputs = model(x.to(torch.float32)).squeeze()
        # return torch.clamp(outputs[1] - outputs[0], min=-kappa)
        return outputs[1] - outputs[0]

    def objective_function(xs):
        where = np.around(xs[:, where_index]).astype(int)
        magnitude = xs[:, magnitude_index]
        belta = np.copy(alpha)
        belta = np.repeat(belta, popsize, axis=1)
        vvv = np.zeros([state_length, popsize])
        for i in range(popsize):
            vvv[where[i, :], i] = 1
        uuu = np.zeros([state_length, popsize])
        for i in range(popsize):
            uuu[where[i, :], i] = magnitude[i, :]
        final = belta * vvv * uuu
        aaa = torch.from_numpy(inputs.reshape(meter_length, 1) + np.matmul(H, final))
        nnn = f(aaa).detach().cpu().numpy().reshape(popsize, -1)
        return nnn

    #bounds = [(0, state_length - 1), (0, state_length - 1), (0, state_length - 1),(0, state_length - 1),(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    bounds = [(0,state_length-1)] * target_number + [(-1,1)] * target_number
    popmul = popsize / len(bounds)

    inits = np.zeros([popsize, len(bounds)])
    for init in inits:
        for ii in where_index:
            init[ii] = np.random.normal(state_length // 2, state_length // 2 - 1)
        for ii in magnitude_index:
            init[ii] = np.random.random()
        # init[0] = np.random.normal(state_length // 2, state_length // 2 - 1)
        # init[1] = np.random.normal(state_length // 2, state_length // 2 - 1)
        # init[2] = np.random.normal(state_length // 2, state_length // 2 - 1)
        # init[3] = np.random.normal(state_length // 2, state_length // 2 - 1)
        # init[4] = np.random.random()
        # init[5] = np.random.random()
        # init[6] = np.random.random()
        # init[7] = np.random.random()

    def attack_success(x, inputs):
        where = np.around(x[where_index]).astype(int)
        magnitude = x[magnitude_index]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length, 1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True

    callback_fn = lambda x, convergence: attack_success(x, inputs)

    def attack_success_output(x, inputs):
        where = np.around(x[where_index]).astype(int)
        magnitude = x[magnitude_index]
        belta = np.copy(alpha)
        vvv = np.zeros([state_length, 1])
        for i in range(1):
            vvv[where] = 1
        uuu = np.zeros([state_length, 1])
        for i in range(1):
            uuu[where] = magnitude[i]
        final = belta * vvv * uuu
        ccc = inputs.reshape(meter_length, 1) + np.matmul(H, final)
        if f2(torch.from_numpy(ccc)).detach().cpu().numpy() <= 0:
            return True, final, np.matmul(H, final)

    attack_result = differential_evolution(objective_function, bounds, maxiter=max_iter, popsize=popmul, mutation=1.0,
                                           recombination=re_combination, tol=0.01, atol=-1, callback=callback_fn, polish=False,
                                           init=inits)

    if attack_success(attack_result.x, inputs) is True:
        state, pert_c, pert_a = attack_success_output(attack_result.x, inputs)
        print("success")
    else:
        state = False
        pert_c = np.zeros([state_length, 1])
        pert_a = np.zeros([meter_length, 1])
        print("failed")
    return state, pert_c, pert_a

def A_cw_l2_attack(model, inputs, attack_vector, meter_length = 54,binary_search=5, kappa=0, max_iter=2000, learning_rate=0.01):
    inputs= inputs.reshape(meter_length,1)
    inputs = torch.from_numpy(inputs)

    attack_vector = attack_vector.reshape(meter_length, 1)
    attack_vector = torch.from_numpy(attack_vector)
    DMIN = np.inf
    CONST = 0.5
    # CONST_max = 100
    CONST_max = 200
    CONST_min = 0
    # Define f-function
    def f(x):
        x = x.t()
        outputs = model(x).squeeze()
        return torch.clamp(outputs[1] - outputs[0], min=-kappa)

    PERT_ALL = torch.zeros_like(inputs,requires_grad=False)

    for m in range(binary_search):
        aaa = torch.zeros_like(inputs, requires_grad=True)

        alpha = torch.zeros(meter_length,1, requires_grad=False)
        alpha[attack_vector==0]=0
        alpha[attack_vector!=0]=1

        optimizer = optim.Adam([aaa], lr=learning_rate)
        dmin = np.inf
        pert = torch.zeros_like(inputs,requires_grad=False)

        for step in range(max_iter):
            loss1 = nn.MSELoss(reduction='sum')(inputs+alpha.mul(aaa), inputs)
            loss2 = CONST * f(inputs+alpha.mul(aaa))
            cost = loss1 + loss2
            if loss2 == 0 and loss1 <= dmin:
                pert = alpha.mul(aaa)
                dmin = loss1

            optimizer.zero_grad()
            cost.backward(retain_graph=True)
            optimizer.step()



        if f(inputs+pert) == 0 and dmin <= DMIN:
            PERT_ALL = pert
            DMIN = dmin

        if f(inputs+pert)==0:
            CONST_max = CONST
        else:
            CONST_min = CONST
        CONST = (CONST_min + CONST_max) / 2

    return PERT_ALL+attack_vector,PERT_ALL, PERT_ALL+inputs, DMIN,f(PERT_ALL+inputs),model((PERT_ALL+inputs).t())