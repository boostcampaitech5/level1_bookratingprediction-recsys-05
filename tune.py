import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.train import train, test
import pdb
from sklearn.model_selection import StratifiedKFold
import json
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from torch.nn import MSELoss
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler
import torch.nn as nn
import pdb
import tqdm
import torch

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
    logger.close()

    return model, valid_loss

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

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss

# 모델에 옵션 파일을 읽어서 optuma 학습 환경 설정
def set_args(args, options, trial):
    print(f'--------------- Setting Experiement ---------------')
    for option in options:
        if options[option][0] == 'int':
            if option == 'DCN_MLP_DIMS':
                value = [trial.suggest_int(option, options[option][1], options[option][2])]
                value = value * args.DCN_MLP_DIM_LAYERS
            else:
                value = trial.suggest_int(option, options[option][1], options[option][2])
        elif options[option][0] == 'cat':
            value = trial.suggest_categorical(option, options[option][1])
        else: pass
        setattr(args, option, value)
    return args


def objective(trial, args):
    option_path = f'/opt/ml/code/src/models/{args.model}/option.json'
    with open(option_path) as f: options = json.load(f)
    args = set_args(args, options, trial)

    ## 기본 실험 세팅
    args.batch_size = trial.suggest_categorical('BATCH_SIZE',[128, 256, 512, 1024])
    args.epochs = 20  #trial.suggest_int('EPOCH',5,10)
    args.lr = trial.suggest_loguniform('LR',0.001,0.01)
    args.weight_decay = trial.suggest_loguniform('WEIGHT_DECAY',1e-07,5e-06)
    args.dropout = trial.suggest_categorical("DCN_DROPOUT",[0.2,0.25,0.3])


    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('FM', 'FFM','DeepFM', 'FFDCN'): 
        data = context_data_load(args)
    elif args.model in ('NCF', 'WDN', 'DCN'): 
        data = dl_data_load(args)
    elif args.model == 'CNN_FM': 
        data = image_data_load(args)
    elif args.model == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
    else: pass

    # 데이터를 불러오고 train/val 나누는 것, main.py와 동일 
    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model in ('FM', 'FFM','DeepFM','FFDCN'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)
        
    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        #data = dl_data_loader(args, data)

    elif args.model=='CNN_FM':
        data = image_data_split(args, data)
        #data = image_data_loader(args, data)

    elif args.model=='DeepCoNN':
        data = text_data_split(args, data)
        #data = text_data_loader(args, data)
    else: pass

    setting = Setting()
    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)
    logger = Logger(args, log_path)
    logger.save_args()

    ################모델 불러오기
    model = models_load(args, data)
    ################모델 학습
    model, val_loss = train(args, model, data, logger, setting)
    ################학습 결과 보기
    log_score = val_loss

    return log_score


def objective_skf(trial, args):
    option_path = f'/opt/ml/code/src/models/{args.model}/option.json'
    with open(option_path) as f: options = json.load(f)
    args = set_args(args, options, trial)

    ## 기본 실험 세팅
    args.batch_size = trial.suggest_categorical('BATCH_SIZE',[128, 256, 512, 1024])
    args.epochs = 20  #trial.suggest_int('EPOCH',5,10)
    args.lr = trial.suggest_loguniform('LR',0.001,0.01)
    args.weight_decay = trial.suggest_loguniform('WEIGHT_DECAY',1e-07,5e-06)
    args.dropout = trial.suggest_categorical("DCN_DROPOUT",[0.2,0.25,0.3])

    setting = Setting()
    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)
    logger = Logger(args, log_path)
    logger.save_args()

    ################모델 불러오기
    model = models_load(args, data)
    ################모델 학습
    model, val_loss = train(args, model, data, logger, setting)
    ################학습 결과 보기
    log_score = val_loss

    return log_score

def main(args):
    Setting.seed_everything(args.seed)

    if args.skf:
            ######################## DATA LOAD
        print(f'--------------- {args.model} Load Data ---------------')
        if args.model in ('FM', 'FFM','DeepFM', 'FFDCN'): 
            data = context_data_load(args)
        elif args.model in ('NCF', 'WDN', 'DCN'): 
            data = dl_data_load(args)
        elif args.model == 'CNN_FM': 
            data = image_data_load(args)
        elif args.model == 'DeepCoNN':
            import nltk
            nltk.download('punkt')
            data = text_data_load(args)
        else: pass

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=arg.seed)
        folds = []
        for train_idx, valid_idx in skf.split(data['train'].drop(['rating'], axis=1), data['train']['rating']):
            folds.append((train_idx, valid_idx))

        for fold in range(0,10):
        print(f'===================================={fold+1}============================================')
        train_idx, valid_idx = folds[fold]
        X_train = train_ratings.drop(['rating'],axis = 1).iloc[train_idx]
        X_valid = train_ratings.drop(['rating'],axis = 1).iloc[valid_idx]
        y_train = train_ratings['rating'].iloc[train_idx]
        y_valid = train_ratings['rating'].iloc[valid_idx]

        if args.model in ('FM', 'FFM','DeepFM','FFDCN'):
            fold_data = {
                    'X_train':X_train,
                    'X_valid': X_valid,
                    'y_train': y_train,
                    'y_valid': y_valid,
                    'field_dims':data['field_dims'],
                    'users':data['users'],
                    'books':data['books'],
                    'sub':data['sub'],
                    'idx2user': data['idx2user'],
                    'idx2isbn':data['idx2isbn'],
                    'user2idx':data['user2idx'],
                    'isbn2idx':data['isbn2idx'],
                    }
            data = context_data_loader(args, data)   
        elif args.model in ('NCF', 'WDN', 'DCN'):
            

            data = dl_data_loader(args, data)
        elif args.model=='CNN_FM':
            data = image_data_loader(args, data)
        elif args.model=='DeepCoNN':
            data = text_data_loader(args, data)
        else: pass


        fold_data = context_data_loader(args,fold_data)

    fold_data = context_data_loader(args,fold_data)
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name = 'cat_parameter_opt',
            direction = 'minimize',
            sampler = sampler,
        )
        study.optimize(lambda trial: objective_skf(trial, args), n_trials=10)

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            study_name = f'{args.model}_parameter_opt',
            direction = 'minimize',
            sampler = sampler,
        )
        print("Best Score:", study.best_value)
        print("Best trial", study.best_trial.params)

    else:
        study.optimize(lambda trial: objective(trial, args), n_trials=10)

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            study_name = f'{args.model}_parameter_opt',
            direction = 'minimize',
            sampler = sampler,
        )
        
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
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN','DeepFM','FFDCN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')


    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=20, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')


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


    args = parser.parse_args()
    main(args)
