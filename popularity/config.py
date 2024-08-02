class Config(object):
    def __init__(self, args):
        self.alpha = args.alpha
        self.batch_size = args.batch_size 
        self.lr = args.lr 
        self.dataset = args.dataset
        self.data_preprocessed = args.data_preprocessed
        self.test_only = args.test_only
        self.num_epochs = args.num_epochs
        self.embedding_dim = args.embedding_dim

        # weight of each component
        self.wt_pop = args.wt_pop 
        self.wt_time = args.wt_time 
        self.wt_side = args.wt_side 

        self.model_save_path = f'./model/{self.dataset}/'
        self.dataset_path = f'../dataset/{self.dataset}/'
        self.raw_review_file_path = f'{self.dataset_path}reviews_{self.dataset}.json.gz'
        self.raw_meta_file_path = f'{self.dataset_path}meta_{self.dataset}.json.gz'
        self.review_file_path = f'{self.dataset_path}{self.dataset}.pkl'

        self.processed_path = f'{self.dataset_path}preprocessed/'        
        self.pop_train_path = f'{self.processed_path}train_df_pop.pkl'
        self.pop_valid_path = f'{self.processed_path}valid_df_pop.pkl'
        self.pop_test_path = f'{self.processed_path}test_df_pop.pkl'
        self.dice_pop_path = f'{self.processed_path}dice_pop_dict.pkl' 
        self.pd_pop_path = f'{self.processed_path}pd_pop_dict.pkl'

        self.cuda_device = args.cuda_device
        

        