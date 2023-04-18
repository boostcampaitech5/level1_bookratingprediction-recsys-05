import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.data import cat_data_load, cat_data_split
import pdb
from sklearn.model_selection import StratifiedKFold
import os
import tqdm
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import SGD, Adam

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss

def train(args, model, dataloader, logger, setting,fold):
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
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        pass

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
        valid_loss = valid(args, model, dataloader, loss_fn)
        print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
        logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_{fold}_fold_model.pt')
    logger.close()
    return model


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


def test(args, model, dataloader, setting,fold):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_{fold}_fold_model.pt'))
    else:
        pass
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

def main(args):
    Setting.seed_everything(args.seed)

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
    elif args.model == 'Cat_Boost':
        data = cat_data_load(args)
    else:
        pass
    
    stk = StratifiedKFold(n_splits = args.n_splits, shuffle = True, random_state = args.seed)
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    
    for fold, (train_idx, val_idx) in enumerate(stk.split(data['train'],data['train']['rating'])):
        print(f'--------------- {args.model} {fold}th Split ---------------')
        
        X_train = data['train'].iloc[train_idx].drop(['rating'], axis=1)
        y_train = data['train']['rating'].iloc[train_idx]
        
        X_valid = data['train'].iloc[val_idx].drop(['rating'], axis=1)
        y_valid = data['train']['rating'].iloc[val_idx]
        
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

        if args.model in ('FM', 'FFM','DeepFM','FFDCN'):
            data = context_data_loader(args, data)
            
        elif args.model in ('NCF', 'WDN', 'DCN'):
            data = dl_data_loader(args, data)        
            
        ####################### Setting for Log
        setting = Setting()

        log_path = setting.get_log_path(args)
        setting.make_dir(log_path)

        logger = Logger(args, log_path)
        logger.save_args()
        
        ######################## Model
        print(f'--------------- INIT {args.model} ---------------')
        model = models_load(args,data)
    
        ######################## TRAIN
        print(f'--------------- {args.model} TRAINING ---------------')
        model = train(args, model, data, logger, setting,fold)


        ######################## INFERENCE
        print(f'--------------- {args.model} PREDICT ---------------')
        predicts = test(args, model, data, setting,fold)
        submission['rating'] += predicts


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission['rating'] = submission['rating'] / args.n_splits

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN','DeepFM','FFDCN','Cat_Boost'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')
    arg('--n_splits', type=int, default=5, help='K-Fold를 조정할 수 있습니다.')


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

        ############### Cat_Boost
    arg('--bagging_temperature', type=float, default=75)
    arg('--n_estimators', type=int, default=8492 )
    arg('--max_depth', type=int, default=6 )
    arg('--random_strength', type=int, default=18 )
    arg('--l2_leaf_reg', type=float, default=5.51030125050448e-06)
    arg('--min_child_samples', type=int, default=34)
    arg('--max_bin', type=int, default=34)
    arg('--od_type', type=str, default="IncToDec")

    args = parser.parse_args()
    main(args)
