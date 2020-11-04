local cuda_device = -1;
local num_epochs = 100;
local patience = 10;
local batch_size = 64;
local shuffle_data = true;
local train_data_path = './data/train_data_aug3.0.tsv';
local validation_data_path = './data/new_test_data.tsv';

// Hyperparameters
local embedding_dim = 64;  // std.parseInt(std.extVar('embedding_dim'))
local lr = 0.001;  // std.parseJson(std.extVar('lr'));

{
    dataset_reader: {
        type: 'tsv-reader',
        token_indexers: {
            tokens: {
                type: 'single_id',
            },
        },
        tokenizer: {
            type: "whitespace",
        },
    },
    model: {
        type: 'simple_classifier',
        embedder: {
            token_embedders: {
                tokens: {
                    type: 'embedding',
                    embedding_dim: embedding_dim,
                },
            },
        },
        encoder: {
            type: 'boe',
            embedding_dim: embedding_dim,
        },
    },
    train_data_path: train_data_path,
    validation_data_path: validation_data_path,
    trainer: {
        cuda_device: cuda_device,
        num_epochs: num_epochs,
        optimizer: {
            type: 'adamw',
            lr: lr,
        },
        patience: patience,
    },
    data_loader: {
        batch_size: batch_size,
        shuffle: shuffle_data,
    },
}
