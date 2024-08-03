class Config(object):
    def __init__(self, args):        
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size 
        self.dropout_rate = args.dropout_rate
        
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim

        self.gamma = args.gamma
        self.PD_gamma = args.PD_gamma
        self.k = args.k

        self.train_num_samples = 4
        self.valid_num_samples = 4
        self.test_num_samples = 99

        self.method = args.method
        self.dataset = args.dataset
        self.data_type = args.data_type
        self.df_preprocessed = args.df_preprocessed
        self.test_only = args.test_only

        self.int_weight = args.int_weight
        self.pop_weight = args.pop_weight
        self.discrepancy_weight = args.discrepancy_weight
        self.reg_weight = args.reg_weight
        self.loss_decay = args.loss_decay
        self.discrepancy_type = args.discrepancy_type
        

        self.wo_con = args.wo_con
        self.wo_qlt = args.wo_qlt

        self.model_save_path = f'./model/{self.dataset}/'
        self.dataset_path = f'../dataset/{self.dataset}/'
        self.review_file_path = f'{self.dataset_path}{self.dataset}.pkl'
        self.pop_file_path = f'{self.dataset_path}pop_{self.dataset}.pkl'

        self.processed_path = f'{self.dataset_path}preprocessed/'   
        self.dice_pop_path = f'{self.processed_path}dice_pop_dict.pkl'  
        self.pd_pop_path = f'{self.processed_path}pd_pop_dict.pkl'     
        self.split_path = f'{self.processed_path}split_df_{self.data_type}.txt'

        self.pos_train_path = f'{self.processed_path}pos_train_df_{self.data_type}.txt'
        self.pos_valid_path = f'{self.processed_path}pos_valid_df_{self.data_type}.txt'
        self.pos_test_path = f'{self.processed_path}pos_test_df_{self.data_type}.txt'
        
        self.train_path = f'{self.processed_path}train_df_{self.data_type}.txt'
        self.valid_path = f'{self.processed_path}valid_df_{self.data_type}.txt'
        self.test_path = f'{self.processed_path}test_df_{self.data_type}.txt'

        self.dice_train_path = f'{self.processed_path}dice_train_df_{self.data_type}.txt'
        self.dice_valid_path = f'{self.processed_path}dice_valid_df_{self.data_type}.txt'
        self.dice_test_path = f'{self.processed_path}dice_test_df_{self.data_type}.txt'

        self.cuda_device = args.cuda_device
        
        