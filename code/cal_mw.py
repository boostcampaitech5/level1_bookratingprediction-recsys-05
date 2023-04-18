import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.data import cat_data_load, cat_data_split
from src.train import train, test
from main import main as m
import pdb
import os
import random


def main(args):
    Setting.seed_everything(args.seed)
    
    ######################## DIR CREATE
    print(f'--------------- directory create ---------------')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    train = train.sample(frac=1,random_state = args.cal)
    train.reset_index(drop=True, inplace=True)
    
    k_fold = 5
    fold_len=len(train)//k_fold
    
    directory = '/opt/ml/cal/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory += str(args.cal)+"_"+str(k_fold)
    if not os.path.exists(directory):
        os.mkdir(directory)
        os.mkdir(directory+'/data/')
        for i in range(k_fold):
            os.mkdir(directory+f'/data/{i}')
            start, end = i*fold_len, (i+1)*fold_len
            if i==k_fold-1:
                end+=len(train)%k_fold
            train.iloc[start:end].to_csv(directory+f'/data/{i}/test_ratings.csv', index=False)
            train.copy().drop(range(start,end)).to_csv(directory+f'/data/{i}/train_ratings.csv', index=False)
        os.mkdir(directory+'/csv/')

    args.cal_path = directory+'/data/0/'
    m(args)
    
    
    




if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### CALCURATE.PY OPTION
    arg('--cal', type=int, default=-100, help='디폴트값이 아닌 수를 부여하면 모델의 weight를 계산합니다.')

    ############### BASIC OPTION
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN','DeepFM','FFDCN','Catboost'],
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