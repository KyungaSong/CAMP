class Config:
    time_unit = 60*60*24*1000 # a day
    time_range = 30 * time_unit # a month
    k_m = 3 * 12 * time_range # three year
    k_s = 6 * time_range # six month
    
    train_num_samples = 4
    valid_num_samples = 4
    test_num_samples = 49

    # Instantiate the model
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 1
    num_epochs = 30
    batch_size = 256
    
    data_preprocessed = True
    k = 20