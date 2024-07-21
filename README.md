# CAMP

## 0. Dataset Download

### Amazon 2014 Dataset

1. Go to the [Amazon 2014 dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html){:target="_blank"} link.
2. Download the reviews and metadata files from the **Per-category files** section.
3. Save the downloaded files under the `dataset/dataset_name` directory.

## 1. Popularity Module

### Hyperparameter Descriptions

- `--alpha`: Parameter for balancing pop_history and time
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--num_epochs`: Number of training epochs
- `--embedding_dim`: Embedding size for embedding vectors
- `--wt_pop`: Weight parameter for balancing the loss contribution of pop_history
- `--wt_time`: Weight parameter for balancing the loss contribution of release_time
- `--wt_side`: Weight parameter for balancing the loss contribution of side_information
- `--dataset`: Dataset file name
- `--data_preprocessed`: Flag to indicate if the input data has already been preprocessed
- `--test_only`: Flag to indicate if only testing should be performed
- `--cuda_device`: CUDA device to use

### Usage Example

```bash
python main.py --dataset Toys_and_Games
```

## 2. Interest Module

### Hyperparameter Descriptions

- `--lr`: Learning rate
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--dropout_rate`: Dropout rate for the model
- `--embedding_dim`: Embedding size for embedding vectors
- `--hidden_dim`: Size of the hidden layer embeddings
- `--output_dim`: Size of the output layer embeddings
- `--gamma`: Discount factor
- `--k`: Value of k for evaluation metrics
- `--discrepancy_loss_weight`: Loss weight for discrepancy between long and short term user embedding
- `--regularization_weight`: Weight for L2 regularization applied to model parameters
- `--wo_con`: Flag to indicate if model has no conformity module
- `--wo_qlt`: Flag to indicate if model has no quality module
- `--dataset`: Dataset file name
- `--data_type`: Dataset split type (reg, unif, seq)
- `--df_preprocessed`: Flag to indicate if the dataframe has already been preprocessed
- `--test_only`: Flag to indicate if only testing should be performed
- `--cuda_device`: CUDA device to use

### Usage Example

```bash
python main.py --dataset Toys_and_Games --data_type unif
```

### Important Note
- The `interest` module should be executed after the `popularity` module to ensure proper training.

