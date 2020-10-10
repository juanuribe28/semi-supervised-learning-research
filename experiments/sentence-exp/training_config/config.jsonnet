local cuda_device = -1;
local num_epochs = 100;
local patience = 10;
local batch_size = 64;
local shuffle_data = true;
local train_data_path = './data/new_train_data.tsv';
local validation_data_path = './data/new_test_data.tsv';

// Hyperparameters
local embedding_dim = 64;  // std.parseInt(std.extVar('embedding_dim'))
local lr = 0.001;  // std.parseJson(std.extVar('lr'));

{
    dataset_reader: {
        type: 'tsv-reader',
        token_indexers: {
            tokens: {
                type: "pretrained_transformer",
                model_name: "bert-base-uncased",
            },
        },
        tokenizer: {
            type: "pretrained_transformer",
            model_name: "bert-base-uncased",
        },
    },
    model: {
        type: 'simple_classifier',
        embedder: {
            token_embedders: {
                tokens: {
                    type: "pretrained_transformer",
                    model_name: "bert-base-uncased",
                }
            }
        },
        encoder: {
            type: 'bert_pooler',
            pretrained_model: "bert-base-uncased",
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
