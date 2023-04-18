from catboost import CatBoostRegressor 
from ...train.trainer import RMSELoss

class Cat_Boost():
    def __init__(self,args,data):
        super().__init__()
        
        self.criterion = RMSELoss()
        
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_valid = data['X_valid']
        self.y_valid = data['y_valid']
        self.sub = data['sub']
        self.cat_features = list(range(0, self.X_train.shape[1]))
        self.param = {
            "random_state":args.seed,
            'learning_rate' : args.lr,
            'bagging_temperature' :args.bagging_temperature,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            'random_strength' : args.random_strength,
            "l2_leaf_reg": args.l2_leaf_reg,
            "min_child_samples": args.min_child_samples,
            "max_bin": args.max_bin,
            'od_type': args.od_type,
            }
        
        self.model = CatBoostRegressor(**self.param, task_type = 'GPU', verbose=50)
        
    def train(self):
        self.model.fit(
            self.X_train,
            self.y_train,
            cat_features = self.cat_features,
            eval_set=(self.X_valid, self.y_valid)
            )
        
    def predict_train(self):
        return self.model.get_best_score()
    
    def predict(self):
        predicts = self.model.predict(self.sub)
        return predicts
    
    def predict_data(self, x):
        return  self.model.predict(x)
    
    def save_weight(self, path):
        self.model.save_model(path)

    def load_weight(self,path):
        self.model.load_model(path)
        return self.model