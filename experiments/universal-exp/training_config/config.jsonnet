local cuda_device = 0;
local num_epochs = 100;
local patience = 10;
local batch_size = 64;
local shuffle_data = true;
local train_data_path = '../data/train_data.tsv';
local validation_data_path = '../data/test_data.tsv';

// Var Hyperparameters
local s_weight = std.parseJson(std.extVar('s_weight'));
local s_embedding_dim = if s_weight == 0 then 64 else std.parseInt(std.extVar('s_embedding_dim'));
local s_dropout = if s_weight == 0 then 0 else std.parseJson(std.extVar('s_dropout'));
local v_embedding_dim = if s_weight == 1 then 64 else std.parseInt(std.extVar('v_embedding_dim'));
local v_dropout = if s_weight == 1 then 0 else std.parseJson(std.extVar('v_dropout'));
local lr = std.parseJson(std.extVar('lr'));

// Default Hyperparameters
// local s_weight = 0.5;
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
            type: "pretrained_transformer",
            model_name: "bert-base-uncased"
        },
        verb_tokenizer: {
            type: "pretrained_transformer",
            model_name: "bert-base-uncased"
        },
    },
    model: {
        type: 'universal_dual_classifier',
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
        sentence_weight: s_weight,
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
        callbacks: [
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
