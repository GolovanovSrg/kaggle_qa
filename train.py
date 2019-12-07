import argparse
import json

from google_quest_qa_labeling.utils import set_seed, get_vocab, get_model, get_trainer, get_train_val_datasets


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='config.json', help="Path of config")

    return parser


def main(args):
    with open(args.config_path, 'r') as json_file:
        config = json.load(json_file)

    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    test_batch_size = config['test_batch_size']
    best_checkpoint_path = config['best_checkpoint_path']
    linear_scheduler = config['linear_scheduler']

    set_seed(config['seed'])

    vocab = get_vocab(config)
    model = get_model(config, vocab)
    train_dataset, val_dataset = get_train_val_datasets(config, vocab, model.n_pos_embeddings)

    if linear_scheduler:
        config['lr_decay'] = (1 - 1e-9) / (n_epochs * (len(train_dataset) + batch_size - 1) // batch_size)
    else:
        config['lr_decay'] = 0

    trainer = get_trainer(config, model)

    trainer.train(train_data=train_dataset,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  test_data=val_dataset,
                  test_batch_size=test_batch_size,
                  best_checkpoint_path=best_checkpoint_path)


if __name__ == "__main__":
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]
    main(args)
