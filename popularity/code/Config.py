import os.path
import torch
from datetime import datetime


class Config(object):
    def __init__(self, args):
        data_path = './dataset/'
        self.is_douban = False
        self.ori_dataset = args.dataset
        # Modification required depending on the type of dataset
        self.main_path = data_path + 'Amazon14/'
        self.side_info_path = self.main_path + \
                                'meta_' + '_'.join(args.dataset.split('_')[1:-1]) + '.jsonl.gz'

        self.dataset = args.dataset + '.csv'
        print('dataset: {}'.format(self.dataset))

        self.processed_path = self.main_path + 'processed/'
        self.fig_path = './fig/'
        self.result_file = './result/'
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
        if not os.path.exists(self.result_file):
            os.mkdir(self.result_file)

        self.train_path = self.processed_path + '{}_pop_train.csv' \
            .format(self.dataset.split('.')[0])
        self.valid_path = self.processed_path + '{}_pop_valid.csv' \
            .format(self.dataset.split('.')[0])
        self.test_path = self.processed_path + '{}_pop_test.csv' \
            .format(self.dataset.split('.')[0])

        self.info_path = f"{self.processed_path}{self.dataset.split('.')[0]}_infos_pop.json"
        self.model_path = './saved_model/{}_model_{}.pth'\
            .format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'), self.dataset)
        self.result_path = '{}new_{}_alpha_{}_a_{}'\
            .format(self.result_file, args.dataset, args.alpha, datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')[:19])


        print('train_path: {}'.format(self.train_path))
        print('valid_path: {}'.format(self.valid_path))
        print('test_path: {}'.format(self.test_path))
        print('model_path: {}'.format(self.model_path))

        # dataset
        self.time_unit = args.time_unit  # a day
        self.pop_time_unit = args.pop_time_unit  # a month
        self.test_time_range = args.test_time_range
        self.user_limit = args.user_limit
        self.item_limit = args.item_limit
        if 'Video' in self.dataset:
            self.pop_history_length = 12
        else:
            self.pop_history_length = 5
        self.pos_item_pop_limit = args.pos_item_pop_limit
        self.neg_item_num = args.neg_item_num

        self.douban_rate_limit = args.douban_rate_limit
        self.do_sta = args.do_sta

        # training
        self.embed_size = args.embed_size
        print('embed_size: {}'.format(self.embed_size))
        self.alpha = args.alpha
        print('alpha: {}'.format(self.alpha))
        self.beta = args.beta
        print('beta: {}'.format(self.beta))
        self.lr = args.lr
        print('lr: {}'.format(self.lr))
        self.loss = args.loss
        print('loss: {}'.format(self.loss))
        self.batch_size = args.batch_size
        print('batch_size: {}'.format(self.batch_size))
        self.epochs = args.epochs
        print('epochs: {}'.format(self.epochs))
        self.dropout = args.dropout
        print('dropout: {}'.format(self.dropout))
        self.top_k = args.top_k
        print('top_k: {}'.format(self.top_k))
        self.add_pop = 1
        print('add_pop: {}'.format(self.add_pop))
        self.max_pop = args.max_pop
        print('max_pop: {}'.format(self.max_pop))
        self.num_eval_count = args.num_eval_count
        self.eval_limit = args.eval_limit

        # processing parameters

        # num_item, max_time, dict_item_idx, dict_item_pop, dict_item_time_release, num_side_info, dict_side_info
        self.json_loaded = False
        self.num_item = -1
        self.max_time = -1
        self.dict_item_idx = {}
        self.dict_idx_item = {}
        self.dict_item_pop = {}
        self.dict_item_time_release = {}
        self.num_side_info = []
        self.dict_side_info = {}

        # running
        self.device = torch.device(args.device)
        print('device: {}'.format(self.device))
        self.save_model = False if args.save_model == 0 else True
        self.load_model = False if args.load_model == 0 else True

        # figure parameter
        self.fig_name_pre = '{}{}-{}'.format(self.fig_path, datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'), self.dataset)
