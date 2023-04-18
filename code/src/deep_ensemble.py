import time
import argparse
import pandas as pd
from catboost import CatBoostRegressor 

from src.utils import Logger, Setting, models_load, ensemble_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.data import cat_data_load, cat_data_split
from src.data import load_ensemble_data, ensemble_data_loader
from src.train import train, test, RMSELoss
from torch.nn import MSELoss
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler
import torch
import os
import pdb
import json
from tqdm import tqdm


def train(args, model, dataloader, logger, setting):
    minimum_loss = 999999999
    if args.loss_fn == 'MSE':
        loss_fn = MSELoss()
    elif args.loss_fn == 'RMSE':
        loss_fn = RMSELoss()
    else:
        pass
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        pass
    if args.scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        batch = 0

        for data in tqdm(dataloader['train_dataloader']):

            x, y, idx = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
            y_hat = model(x, idx)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch +=1

        valid_loss = valid(args, model, dataloader, loss_fn)
        print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
        logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pth')
    logger.close()
    return model

def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for data in tqdm(dataloader['valid_dataloader']):

        x, y, idx = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        y_hat = model(x, idx)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
    valid_loss = total_loss/batch
    return valid_loss


def main(args):
    Setting.seed_everything(args.seed)
    ######################## DATA LOAD
    print(f'--------------- {args.models} Load Data ---------------')
    done = set()
    ensemble_data = {}
    args.batch_size = 1024
    args.shuffle = False
    for model in args.models:
        if model in ('FM', 'FFM','DeepFM', 'FFDCN'):
            data_type = 'context'
            if data_type not in done:
                args.model = model
                data = context_data_load(args)
                data = context_data_split(args, data)
                data = context_data_loader(args, data)
                
        elif model in ('NCF', 'WDN', 'DCN'):
            data_type = 'dl'
            if data_type not in done:
                args.model = model
                data = dl_data_load(args)
                data = dl_data_split(args, data)
                data = dl_data_loader(args, data)

        elif model == 'CNN_FM':
            data_type = 'image'
            if data_type not in done:
                args.model = model
                data = image_data_load(args)
                data = image_data_split(args, data)
                data = image_data_loader(args, data)

        elif model == 'DeepCoNN':
            data_type = 'image'
            if data_type not in done:
                import nltk
                nltk.download('punkt')
                args.model = model
                data = text_data_load(args)
                data = text_data_split(args, data)
                data = text_data_loader(args, data)

        elif model == 'Cat_Boost':
            data_type = 'cat'
            if data_type not in done:
                data = cat_data_load(args)
                data = cat_data_split(args,data)
                #data = cat_data_loader(args, data)

        else: pass

        done.add(data_type)
        ensemble_data[data_type] = data
    
    data_types = list(done)
    ####################### Setting for Log
    setting = Setting()
    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)
    logger = Logger(args, log_path)
    logger.save_args()


    ######################## Load Models For Ensemble

    models = [] ##앙상블할 모델 리스트
    data2model = [] ##모델에 맞는 데이터 타입 순서
    
    for model_type in args.models:
        if model_type in ('FM', 'FFM','DeepFM', 'FFDCN'): data_type = 'context'
        elif model_type in ('NCF', 'WDN', 'DCN'): data_type = 'dl'     
        elif model_type == 'CNN_FM': data_type = 'image'
        elif model_type == 'DeepCoNN': data_type = 'image'
        elif model_type == 'Cat_Boost' : data_type = 'cat'
        args.model = model_type

    ######################## Load Models Paramter and Weight
        print(f'--------------- INIT {model_type} ---------------')
        if model_type != "Cat_Boost":
            param_path = f'/opt/ml/code/src/models/{model_type}/best_params.json'
            if os.path.isfile(param_path):
                with open(param_path, 'r') as file: best_params = json.load(file)
                vars(args).update(best_params)
                args = argparse.Namespace(**vars(args))

            model = models_load(args, ensemble_data[data_type])
                
            model.load_state_dict(torch.load(f'/opt/ml/code/src/models/{model_type}/best_model.pth'))
            for param in model.parameters(): param.requires_grad = False
            models.append(model)
            data2model.append(data_type)
            
        elif model_type == 'Cat_Boost':

            model = CatBoostRegressor(task_type = 'GPU', verbose=50)
            #model = models_load(args, ensemble_data[data_type])
            #model.load_model(f'/opt/ml/code/src/models/{ model_type}/best_model.cbm')
            
            model = model.load_model('/opt/ml/code/saved_models/20230418_103230_Cat_Boost_model.cbm')
            models.append(model)
            data2model.append(data_type)

            

    args.num_models = len(models)
    args.num_users = pd.read_csv(args.data_path + 'users.csv').loc[:,'user_id'].nunique()
    model = ensemble_load(args)

    args.batch_size = 1
    args.shuffle = True
    data= load_ensemble_data(models, data2model, ensemble_data, args)
    data = ensemble_data_loader(args, data)

    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    model = train(args, model, data, logger, setting)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = test(args, model, data, setting)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'FFDCN'):
        submission['rating'] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')

    ############### TRAINING OPTION
    arg('--n_splits', type=int, default=10, help='Stratified K-FOld 여부 개수')
    arg('--batch_size', type=int, default=1, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')

    ############### BASIC OPTION
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--models', type=list, default=['FFDCN','DeepFM','Cat_Boost'], help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--load_all_skf', type=bool, default=True, help='Stratified fold로 학습된 모든 fold를 불러올 것인지')
    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')
    

    ############### FM, FFM, NCF, WDN, DCN Common OPTION
    arg('--embed_dim', type=int, default=16, help='FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2, help='NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=list, default=(16, 16), help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--scheduler', type=bool, default=False, help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')

    ############### DCN
    arg('--num_layers', type=int, default=3, help='에서 Cross Network의 레이어 수를 조정할 수 있습니다.')


    ############### CNN_FM
    arg('--cnn_embed_dim', type=int, default=64, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--cnn_latent_dim', type=int, default=12, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')


    ############### DeepCoNN
    arg('--vector_create', type=bool, default=True, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--deepconn_embed_dim', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--deepconn_latent_dim', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--conv_1d_out_dim', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--kernel_size', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--word_dim', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--out_dim', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')

        ############### Cat_Boost
    arg('--bagging_temperature', type=float, default=75)
    arg('--n_estimators', type=int, default=8492 )
    arg('--max_depth', type=int, default=6 )
    arg('--random_strength', type=int, default=18 )
    arg('--l2_leaf_reg', type=float, default=5.51030125050448e-06)
    arg('--min_child_samples', type=int, default=34)
    arg('--max_bin', type=int, default=34)
    arg('--od_type', type=str, default="ncToDec")


    args = parser.parse_args()
    main(args)
