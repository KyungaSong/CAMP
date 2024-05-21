class Config(object):
    def __init__(self, args):        
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size 
        self.dropout_rate = args.dropout_rate
        
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim

        self.time_unit = 60*60*24*1000 # a day
        self.time_range = 30 * self.time_unit # a month
        self.k_m = args.k_m * self.time_range
        self.k_s = args.k_s * self.time_range
        self.k = args.k

        self.train_num_samples = 4
        self.valid_num_samples = 4
        self.test_num_samples = 49

        self.dataset = args.dataset
        self.data_preprocessed = args.data_preprocessed
        self.test_only = args.test_only

        self.regularization_weight = args.regularization_weight
        self.discrepancy_loss_weight = args.discrepancy_loss_weight

        self.has_mid = args.has_mid

        self.model_path = f'../../model/'
        
        