import os
import json
import pickle
import numpy as np
from collections import OrderedDict


datasets = ['davis','kiba']
for dataset in datasets:
    fpath = 'data/' + dataset + '/'
    os.makedirs(os.path.join(fpath,'processed'),exist_ok=True)
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    folds = train_fold + [valid_fold]
    valid_ids = [5,4,3,2,1]
    valid_folds = [folds[vid] for vid in valid_ids]
    train_folds = []
    for i in range(5):
        temp = []
        for j in range(6):
            if j != valid_ids[i]:
                temp += folds[j]
        train_folds.append(temp)
    
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]


    affinity = np.asarray(affinity)
    opts = ['train','test']
    for i in range(5):
        train_fold = train_folds[i]
        valid_fold = valid_folds[i]
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)  
            if opt=='train':
                rows,cols = rows[train_fold], cols[train_fold]
            elif opt=='test':
                rows,cols = rows[valid_fold], cols[valid_fold]
                
            if i == 0:
                # generating standard data
                print('generating standard data')
                with open(os.path.join(fpath,'processed',f'{opt}.txt'), 'w') as f:
                    # f.write('drug_id,protein_id,affinity\n')
                    for pair_ind in range(len(rows)):
                        ls = []
                        ls += [ rows[pair_ind] ] # drug_id
                        ls += [ cols[pair_ind] ] # protein_id
                        ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ] # affinity
                        f.write(' '.join(map(str,ls)) + '\n')
                      
            # 5-fold validation data
            print('generating 5-fold validation data')
            with open(os.path.join(fpath,'processed',f'{opt}_{i}.txt'), 'w') as f:
                # f.write('drug_id,protein_id,affinity\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ rows[pair_ind] ]
                    ls += [ cols[pair_ind] ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    f.write(' '.join(map(str,ls)) + '\n')       

