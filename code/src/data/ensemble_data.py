import pdb
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
class EnsembleDataset(Dataset):
    def __init__(self, X, y, user_idx):
        self.X = torch.from_numpy(X)
        self.X = self.X.type(torch.float32)
        self.y = torch.from_numpy(y)
        self.y = self.y.type(torch.float32)
        self.user_idx = user_idx

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        idx = self.user_idx[index]
        return x, y, idx
    
def load_ensemble_data(ensemble_models, data2model, ensemble_data, args):
    X_train_ensemble = []
    X_valid_ensemble = []
    y_train_ensemble = ensemble_data[data2model[0]]['y_train']
    y_valid_ensemble = ensemble_data[data2model[0]]['y_valid']
    user_idx_train = ensemble_data[data2model[0]]['X_train'].iloc[:,0] 
    user_idx_valid = ensemble_data[data2model[0]]['X_valid'].iloc[:,0]  

 ###################Predict X_train
    for ensemble_model, data_type in zip(ensemble_models, data2model):
        if data_type != 'cat':
            predicts = []
    
            dataloader = ensemble_data[data_type]['train_dataloader']
            ensemble_model.eval()
            for data in tqdm(dataloader):

                if data_type == 'img':
                    x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
                elif data_type == 'text':
                    x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
                else:
                    x = data[0].to(args.device)
                y_hat = ensemble_model(x)
                predicts.extend(y_hat.tolist())
            X_train_ensemble.append(predicts)

        elif data_type == 'cat':
            data = ensemble_data[data_type]['X_train']
            X_train_ensemble.append(ensemble_model.predict(data))


###################Predict X_Valid
    for ensemble_model, data_type in zip(ensemble_models, data2model):
        if data_type != 'cat':
            predicts = []
            dataloader = ensemble_data[data_type]['valid_dataloader']
            ensemble_model.eval()
            for data in tqdm(dataloader):
                if data_type == 'img':
                    x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
                elif data_type == 'text':
                    x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
                else:
                    x = data[0].to(args.device)
                y_hat = ensemble_model(x)
                predicts.extend(y_hat.tolist())
            X_valid_ensemble.append(predicts)

        elif data_type == 'cat':
            data = ensemble_data[data_type]['X_valid']
            X_valid_ensemble.append(ensemble_model.predict(data))

    data = {}

    X_train_ensemble = np.array(X_train_ensemble).T
    X_valid_ensemble = np.array(X_valid_ensemble).T
    y_train_ensemble = np.array(y_train_ensemble)
    y_valid_ensemble = np.array(y_valid_ensemble)
    data['user_idx_train'] = np.array(user_idx_train)
    data['user_idx_valid'] = np.array(user_idx_valid)
    data['X_train'] = X_train_ensemble
    data['X_valid'] = X_valid_ensemble
    data['y_train'] = y_train_ensemble
    data['y_valid'] = y_valid_ensemble
    return data

def ensemble_data_loader(args, data):

    train_dataset = EnsembleDataset(data['X_train'], data['y_train'], data['user_idx_train'])
    valid_dataset = EnsembleDataset(data['X_valid'],  data['y_valid'], data['user_idx_valid'])
    #test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'] = train_dataloader, valid_dataloader
    return data

