local cuda_device = -1;
local num_epochs = 100;
local patience = 10;
local batch_size = 64;
local shuffle_data = true;
local train_data_path = './data/train_data.tsv';
local validation_data_path = './data/test_data.tsv';

// Hyperparameters
local embedding_dim = 64; 
local dropout = 0.8;
local s_embedding_dim = std.parseInt(std.extVar('s_embedding_dim'));
local s_dropout = std.parseJson(std.extVar('s_dropout'));
local v_embedding_dim = std.parseInt(std.extVar('v_embedding_dim'));
local v_dropout = std.parseJson(std.extVar('v_dropout'));
local lr = std.parseJson(std.extVar('lr'));
// local s_embedding_dim = 64;
// local s_dropout = 0.8;
// local v_embedding_dim = 64;
// local v_dropout = 0.8;
// local lr = 0.01;

{
    dataset_reader: {
        type: 'architecture.dataset_reader.TSVDatasetReader',
        sentence_token_indexers: {
            tokens: {
                type: 'single_id',
            },
        },
        verb_token_indexers: {
            tokens: {
                type: 'single_id',
            },
        },
        sentence_tokenizer: {
            type: 'whitespace',
        },
        verb_tokenizer: {
            type: 'character',
        },
    },
    model: {
        type: 'dual_simple_classifier',
        sentence_embedder: {
            token_embedders: {
                tokens: {
                    type: 'embedding',
                    embedding_dim: s_embedding_dim,
                },
            },
        },
        verb_embedder: {
            token_embedders: {
                tokens: {
                    type: 'embedding',
                    embedding_dim: v_embedding_dim,
                },
            },
        },
        sentence_encoder: {
            type: 'boe',
            embedding_dim: s_embedding_dim,
        },
        verb_encoder: {
            type: 'boe',
            embedding_dim: v_embedding_dim,
        },
        sentence_dropout: s_dropout,
        verb_dropout: v_dropout,
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
        epoch_callbacks: [
            {
                type: 'optuna_pruner',
            },
        ],
    },
    data_loader: {
        batch_size: batch_size,
        shuffle: shuffle_data,
    },
}
