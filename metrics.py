import numpy as np
import torch as th
import pickle as pkl
import matplotlib.pyplot as plt
import clip

def compute_metrics(x, optimistic=True):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    uni,idx = np.unique(ind[0],return_index=True)
    if not optimistic:
        assert(uni.shape[-1] == 1000)
    ind = ind[1][idx]
    test_set_size = x.shape[0]
    print('Recall @ 1', float(np.sum(ind < 1)) / test_set_size)
    print('Recall @ 2', float(np.sum(ind < 2)) / test_set_size)
    print('Recall @ 5', float(np.sum(ind < 5)) / test_set_size)
    print('Recall @ 10', float(np.sum(ind < 10)) / test_set_size)
    print('Median R', np.median(ind) + 1)

def evaluate(model, dataloader, clip_size='base', loss_type='MMS'):
    model.eval()
    
    with th.no_grad():
        if clip_size == 'base':
            mmin1 = th.empty(0,512) # matmul input 1
            mmin2 = th.empty(0,512) # matmul input 2
        else:
            mmin1 = th.empty(0,768) # matmul input 1
            mmin2 = th.empty(0,768) # matmul input 2
        
        b = 0
        for batch in dataloader:
            b += 1
            if loss_type == 'CrossLingual':
                image_out, text_out = model(batch['image'].cuda(), batch['random_caption'].cuda(), langID=batch['langID'].cuda())
            else:
                image_out, text_out = model(batch['image'].cuda(), batch['caption'].cuda(), langID=batch['langID'].cuda()) 
            mmin1 = th.cat((mmin1, image_out.cpu().detach()),dim=0)
            mmin2 = th.cat((mmin2, text_out.cpu().detach()),dim=0)
       
        sets_of_1000 = mmin1.shape[0] // 1000
        if sets_of_1000 == 0:
            print('Only',mmin1.shape[0],'samples in val dataset!')
            all_sim = th.matmul(mmin1, mmin2.t())
            print('I2S')
            compute_metrics(all_sim) 
            print('S2I')
            compute_metrics(all_sim.t())
        else:
            for i in range(sets_of_1000):
                i0 = i * 1000
                all_sim = th.matmul(mmin1[i0:i0+1000],mmin2[i0:i0+1000].t())
                print('Evaluation set', i, 'I2S')
                compute_metrics(all_sim) 
                print('Evaluation set', i, 'S2I')
                compute_metrics(all_sim.t())
