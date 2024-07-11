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

        self.model_save_path = f'../model/{self.dataset}/'
        self.dataset_path = f'../dataset/{self.dataset}/'
        self.review_file_path = f'{self.dataset_path}{self.dataset}.pkl'
        self.pop_file_path = f'{self.dataset_path}pop_{self.dataset}.pkl'

        self.processed_path = f'{self.dataset_path}preprocessed/'        
        self.split_path = f'{self.processed_path}split_df_{self.data_type}.txt'

        self.pos_train_path = f'{self.processed_path}pos_train_df_{self.data_type}.txt'
        self.pos_valid_path = f'{self.processed_path}pos_valid_df_{self.data_type}.txt'
        self.pos_test_path = f'{self.processed_path}pos_test_df_{self.data_type}.txt'
        
        self.train_path = f'{self.processed_path}train_df_{self.data_type}.txt'
        self.valid_path = f'{self.processed_path}valid_df_{self.data_type}.txt'
        self.test_path = f'{self.processed_path}test_df_{self.data_type}.txt'

        self.cuda_device = args.cuda_device
        

        