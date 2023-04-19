import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load, set_args
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.data import cat_data_load, cat_data_split
from src.train import RMSELoss
import pdb
from sklearn.model_selection import StratifiedKFold
import json
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from torch.nn import MSELoss
from torch.optim import SGD, Adam, AdamW
from torch.optim import lr_scheduler
import torch.nn as nn
import pdb
import tqdm
import torch

def train(args, model, dataloader, logger, setting, fold, save = False):
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
    elif args.optimizer == 'ADAMW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        pass
    if args.scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        batch = 0

        for idx, data in enumerate(dataloader['train_dataloader']):
            if args.model == 'CNN_FM':
                x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch +=1
            if args.scheduler: scheduler.step()
            
            
        valid_loss = valid(args, model, dataloader, loss_fn)
        print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
        logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
        if (minimum_loss > valid_loss): 
            minimum_loss = valid_loss
            if save == True:
                minimum_loss = valid_loss
                torch.save(model.state_dict(), f'/opt/ml/code/src/models/{args.model}/best_model{fold}.pth')
    logger.close()

    return model, minimum_loss

def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for idx, data in enumerate(dataloader['valid_dataloader']):
        if args.model == 'CNN_FM':
            x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
    valid_loss = total_loss/batch
    return valid_loss

def test(args, model, dataloader, setting):
    predicts = list()
    model.eval()

    for idx, data in enumerate(dataloader['test_dataloader']):
        if args.model == 'CNN_FM':
            x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts


def objective(trial, args, data):
    option_path = f'/opt/ml/code/src/models/{args.model}/option.json'
    with open(option_path) as f: options = json.load(f)
    args = set_args(args, options, trial)

    ## 기본 실험 세팅
    args.batch_size = trial.suggest_categorical('batch_size',[64, 128, 256, 512, 1024, 2048])
    args.lr = trial.suggest_loguniform('lr',0.0005,0.01)
    args.weight_decay = trial.suggest_loguniform('weight_decay',1e-07,1e-04)
    args.dropout = trial.suggest_categorical("dropout",[0.2,0.25,0.3, 0.4, 0.5])
    #args.seed = trial.suggest_int("seed",21, 42)

    setting = Setting()
    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)
    logger = Logger(args, log_path)
    logger.save_args()

    ################모델 불러오기
    model = models_load(args, data)
    ################모델 학습
    model, val_loss = train(args, model, data, logger, setting, False, None)
    ################학습 결과 보기
    log_score = val_loss

    return log_score

def objective_CatBoost(trial, args, data):
    option_path = f'/opt/ml/code/src/models/{args.model}/option.json'
    with open(option_path) as f: options = json.load(f)
    args = set_args(args, options, trial)

    ## 기본 실험 세팅
    args.lr = trial.suggest_loguniform('lr',0.0005,0.01)
    #args.seed = trial.suggest_int("seed",21, 42)


    ################모델 불러오기
    model = models_load(args, data)
    ################모델 학습
    model.train()
    
    ################학습 결과 보기
    log_score = list( model.predict_train().values())[0]
    return log_score



def main(args):
    Setting.seed_everything(args.seed)
    
    print('--------------- Optuma Basic Mode---------------')
    ######################## DATA LOAD
    
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('FM', 'FFM','DeepFM', 'FFDCN','FFDCN_P'): 
        data = context_data_load(args)
    elif args.model in ('NCF', 'WDN', 'DCN','DCN_P'): 
        data = dl_data_load(args)
    elif args.model == 'CNN_FM': 
        data = image_data_load(args)
    elif args.model == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
    elif args.model == 'Cat_Boost':
        data = cat_data_load(args)
    else: pass

    # 데이터를 불러오고 train/val 나누는 것, main.py와 동일 
    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model in ('FM', 'FFM','DeepFM','FFDCN','FFDCN_P'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)
        
    elif args.model in ('NCF', 'WDN', 'DCN','DCN_P'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.model=='CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.model=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)
    elif args.model == "Cat_Boost":
        data = cat_data_split(args,data)
    else: pass

    ######################## Setting Study

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name = f'{args.model}_parameter_opt',
        direction = 'minimize',
        sampler = sampler,
        )

    if args.model != "Cat_Boost":
        study.optimize(lambda trial: objective(trial, args, data), n_trials=500)
    
    elif args.model == "Cat_Boost":
        study.optimize(lambda trial: objective_CatBoost(trial, args, data), n_trials=1000)

    ###################### Updating Parameter
    best_val = study.best_trial.value
    best_params = study.best_trial.params
    
    #for arg in best_params: setattr(args, arg, best_params[arg])
    vars(args).update(best_params)
    args = argparse.Namespace(**vars(args))

    args_dict = vars(args)
    for param in args_dict:
        if param != 'data': best_params[param] = args_dict[param]

    with open (f'/opt/ml/code/src/models/{args.model}/best_params.json', 'w') as f: json.dump(best_params, f)
    
    ###################### Inference Using Best Param
    if args.model != "Cat_Boost":

        model = models_load(args, data)

        setting = Setting()
        log_path = setting.get_log_path(args)
        setting.make_dir(log_path)
        logger = Logger(args, log_path)
        logger.save_args()

        model, loss = train(args, model, data, logger, setting, '', True)
        print(f'End with RMSE:{loss}')
        print(f'--------------- {args.model} PREDICT ---------------')
        
        state_dict = torch.load(f"/opt/ml/code/src/models/{args.model}/best_model.pth")
        state_dict = state_dict.copy()
        model.load_state_dict(state_dict)

        predicts = test(args, model, data, setting)

        ######################## SAVE PREDICT
        print(f'--------------- SAVE {args.model} PREDICT ---------------')
        submission = pd.read_csv(args.data_path + 'sample_submission.csv')
        if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'FFDCN','DCN_P'):
            submission['rating'] = predicts
        else:
            pass
        
    elif args.model == "Cat_Boost":

        model = models_load(args, data)
        model.train()

        model.save_weight(f"/opt/ml/code/src/models/{args.model}/best_model.cbm")
        predicts = model.predict()
        submission = pd.read_csv(args.data_path + 'sample_submission.csv')
        submission['rating'] = predicts
        ######################## SAVE PREDICT
        print(f'--------------- SAVE {args.model} PREDICT ---------------')
        filename = setting.get_submit_filename(args)
        submission.to_csv(filename, index=False)   

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FFDCN_P','FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM','DCN_P','DeepCoNN','DeepFM','FFDCN', 'Cat_Boost'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')
    arg('--skf', type=bool, default=False, help='Stratified K-FOld 여부.')

    ############### TRAINING OPTION
    arg('--n_splits', type=int, default=10, help='Stratified K-FOld 여부 개수')
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM', 'ADAMW'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')


    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    ############### FM, FFM, NCF, WDN, DCN Common OPTION
    arg('--embed_dim', type=int, default=16, help='FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2, help='NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=list, default=(16, 16), help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--scheduler', type=bool, default=False, help='러닝 스케듈러 설정.')

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
