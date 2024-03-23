import argparse
import os
import pandas as pd

import json
import pandas as pd


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exp-path', type=str, default="file/experiments/imagenet_train_from_scratch")
    p.add_argument('--check', type=str, default="epoch", choices=["epoch", "best_val_top1", "best_test_top1"]) # choices=["epoch", "best_val_top1"]
    p.add_argument('--folder-name', type=str, default="class_subset_")
    p.add_argument('--epoch-num', type=int, default=49)
    p.add_argument('--table', type=str, default="dataset", choices=["dataset", "method"])
    
    args = p.parse_args()

    assert os.path.isdir(args.exp_path), 'exp_path should be path to a folder that contains all the experiments!'
    
    v = ['ffm', 'ffmm', 'cmt', 'cmt_median']
    d = ['dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101']
    # s = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950']
    s = ['950', '900', '850', '800', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '200', '150', '100', '50']
    
    failed_setting = []
    
    if args.table == 'method':
        for vv in v:
            total_data = []
            for dd in d:
                dataset_data = []
                for ss in s:
                    result_path = os.path.join(args.exp_path, vv, dd, args.folder_name + ss, 'log.json')
                    f = open(result_path, 'r')
                    data = json.load(f)
                    dataset_data.append(data[-1][args.check])
                    
                    # Reord the failed setting
                    if args.check == 'epoch' and data[-1][args.check] < args.epoch_num:
                        failed_setting.append(result_path)
                        print(result_path)
                        print(data[-1][args.check])
            
                total_data.append(dataset_data)
                
            # print(failed_setting)
            
            with pd.ExcelWriter(os.path.join(args.exp_path+'/'+vv, args.check+'.xlsx'), mode='w') as writer:
                total_data = pd.DataFrame(total_data, index=d, columns=s)
                total_data.to_excel(writer)
    
    elif args.table == 'dataset':
        for dd in d:
            total_data = []
            for vv in v:
                method_data = []
                for ss in s:
                    result_path = os.path.join(args.exp_path, vv, dd, args.folder_name + ss, 'log.json')
                    f = open(result_path, 'r')
                    data = json.load(f)
                    method_data.append(data[-1][args.check]*100)
                    
                    # Reord the failed setting
                    if args.check == 'epoch' and data[-1][args.check] < args.epoch_num:
                        failed_setting.append(result_path)
                        print(result_path)
                        print(data[-1][args.check])
            
                total_data.append(method_data)
            
            with pd.ExcelWriter(os.path.join(args.exp_path, dd+'_'+args.check+'.xlsx'), mode='w') as writer:
                total_data = pd.DataFrame(total_data, index=v, columns=s)
                total_data.to_excel(writer)
            
            # failed_setting = pd.DataFrame(failed_setting, index=range(len(failed_setting)), columns='dir')
            # failed_setting.to_excel(writer, v)
            
                
            