class Config(object):
    def __init__(self, args):        
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size 
        self.dropout_rate = args.dropout_rate
        
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim

        self.k_m = args.k_m
        self.k_s = args.k_s
        self.k = args.k

        self.train_num_samples = 4
        self.valid_num_samples = 4
        self.test_num_samples = 99

        self.dataset = args.dataset
        self.data_type = args.data_type
        self.df_preprocessed = args.df_preprocessed
        self.test_only = args.test_only

        self.regularization_weight = args.regularization_weight
        self.discrepancy_loss_weight = args.discrepancy_loss_weight

        self.wo_mid = args.wo_mid
        self.wo_con = args.wo_con
        self.wo_qlt = args.wo_qlt

        self.model_path = f'../../model/'

        self.cuda_device = args.cuda_device
        
        