local cuda_device = 0;
local num_epochs = 100;
local patience = 10;
local batch_size = 64;
local shuffle_data = true;
local train_data_path = './data/train_data.tsv';
local validation_data_path = './data/test_data.tsv';

// Hyperparameters
local embedding_dim = std.parseInt(std.extVar('embedding_dim')); // 64
local lr = std.parseJson(std.extVar('lr'));  // 0.01
local dropout = std.parseJson(std.extVar('dropout'));  // 0

{
    dataset_reader: {
        type: 'architecture.dataset_reader.TSVDatasetReader',
        token_indexers: {
            tokens: {
                type: 'single_id',
            },
        },
        tokenizer: {
            type: "pretrained_transformer",
            model_name: "bert-base-uncased"
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
        dropout: dropout,
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
