class Config(object):
    def __init__(self, args):        
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size 
        self.dropout_rate = args.dropout_rate
        
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim

        self.time_unit = args.time_unit
        self.pop_time_unit = args.pop_time_unit * self.time_unit
        self.k_m = args.k_m * self.time_unit
        self.k_s = args.k_s * self.time_unit
        self.k = args.k

        self.train_num_samples = 4
        self.valid_num_samples = 4
        self.test_num_samples = 49

        self.dataset = args.dataset
        self.df_preprocessed = args.df_preprocessed
        self.test_only = args.test_only

        self.regularization_weight = args.regularization_weight
        self.discrepancy_loss_weight = args.discrepancy_loss_weight

        self.no_mid = args.no_mid
        self.no_con = args.no_con
        self.no_qlt = args.no_qlt

        self.inv = args.inv

        self.model_path = f'../../model/'

        self.cuda_device = args.cuda_device
        
        