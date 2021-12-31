import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import random
import pandas as pd
import os
import numpy as np
import random
import torch
import math
import json


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def replace_nans(texts, original):
    """
    replaces the nans in the texts with other random texts
    :param texts: the texts

    :return: the replaced texts
    """
    n = len(texts)
    texts = [text for text in texts if str(text) != 'nan' and text != '']
    if len(texts) == 0: 
        return [original for i in range(n)]
    if len(texts) < n:
        texts = random.choices(texts, k=n)
    return texts


def generate_samples(df, anchor_column="question", positive_cols=[], negative_cols=[], use_inbatch=False, max_triplets_per_sample=-1):
    """
    generates the triplets from the dataframe
    :param df: the dataframe
    :param anchor_column: the column name of the anchor column
    :param positive_cols: the columns that are used to generate the positive samples
    :param negative_cols: the columns that are used to generate the negative samples
    :param use_inbatch: if true, the samples are generated in batches, otherwise, they are present in the negative_cols
    :param max_triplets_per_sample: the maximum number of triplets per sample

    :return: a list of the triplets
    """
    if not use_inbatch and len(negative_cols) == 0: raise ValueError("if use_inbatch is false, negative_cols must be specified")
    if use_inbatch and len(negative_cols) > 0: raise ValueError("if use_inbatch is true, negative_cols must not be specified")

    train_samples = []
    num_triplets_per_sample = max(len(positive_cols), len(negative_cols))
    # if max_triplets_per_sample is specified, we limit the number of triplets per sample
    if max_triplets_per_sample > 0: num_triplets_per_sample = min(num_triplets_per_sample, max_triplets_per_sample) 

    for idx, row in df.iterrows():
        anchor_text = row[anchor_column]
        positives, negatives = [], []

        # generate the positive samples
        if len(positive_cols) >= num_triplets_per_sample:
            positive_cols_chosen = random.sample(positive_cols, num_triplets_per_sample)
        else:
            positive_cols_chosen = random.choices(positive_cols, k=num_triplets_per_sample)
        positives = [row[col] for col in positive_cols_chosen]
        positives = replace_nans(positives, anchor_text)

        # generate the negative samples
        if use_inbatch:
            negative_idxs_list = list(df.index.difference([idx]))
            negative_idxs = random.sample(negative_idxs_list, num_triplets_per_sample)
            negatives = [df.loc[idx, anchor_column] for idx in negative_idxs]
        else:
            if len(negative_cols) >= num_triplets_per_sample:
                negative_cols_chosen = random.sample(negative_cols, num_triplets_per_sample)
            else:
                negative_cols_chosen = random.choices(negative_cols, k=num_triplets_per_sample)
            negatives = [row[col] for col in negative_cols_chosen]
            negatives = replace_nans(negatives, " ".join(anchor_text.split()[:-5]))

        for positive, negative in zip(positives, negatives):
            train_samples.append(InputExample(texts = [anchor_text, positive, negative]))

    return train_samples


def train_with_seed(seed, train_samples, val_samples, model_path='sentence-transformers/paraphrase-mpnet-base-v2', stop_after = 0.2, 
            num_epochs=1, batch_size=16, output_dir='output', verbose=True, save_loss=False, loss_fn="triplet"):
    """
    Trains for one epoch with stop_after fraction of the training data for a given seed
    """
    set_seed(seed)

    model = SentenceTransformer(model_path)
    train_dataset = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    evaluator = evaluation.TripletEvaluator.from_input_examples(val_samples)

    warmup_steps = int(len(train_dataset) * num_epochs * 0.1)
    steps_per_epoch = math.ceil( len(train_dataset) / batch_size * stop_after)

    if loss_fn == "triplet":
        train_loss = losses.TripletLoss(model=model)
    elif loss_fn == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        raise NotImplementedError

    model_save_path = os.path.join(output_dir, 'seed_{}'.format(seed))

    model.fit(train_objectives=[(train_dataset, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=150,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        output_path=model_save_path)

    if verbose:
        print(f"Seed {seed} finished!")


def read_eval_file(eval_path):
    """
    reads the eval file from the evaluation path
    returns the mean of the last three cosine accuracies
    """
    eval_file = [f for f in os.listdir(eval_path) if f.endswith('.csv')][0]
    eval_df = pd.read_csv(os.path.join(eval_path, eval_file))
    cosine_accuracies = eval_df["accuracy_cosinus"].values
    # Take the mean of the last three values of the Series
    cosine_acc = np.mean(cosine_accuracies[-3:])
    return cosine_acc


def get_best_seed(model_dir):
    """
    given the model dir, returns the best seed by reading the seed results
    """

    model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith('seed_')]
    seeds = [int(f.split('_')[-1].split('.')[0]) for f in model_paths]
    eval_paths = [os.path.join(model_path, "eval") for model_path in model_paths]

    best_seed = -1; cosine_acc = -1
    for seed, eval_path in zip(seeds, eval_paths):
        cosine_results = read_eval_file(eval_path)

        if cosine_results > cosine_acc:
            best_seed = seed
            cosine_acc = cosine_results
    
    return best_seed


def train(train_samples, val_samples, model_path='sentence-transformers/paraphrase-mpnet-base-v2', num_epochs=10, 
        batch_size=16, output_dir='output', verbose=True, save_loss=False, loss_fn="triplet"):
    """
    train the biencoder
    :param train_samples: the training samples
    :param val_samples: the validation samples
    :param model_path: the path to the pretrained model
    :param num_epochs: the number of epochs
    :param batch_size: the batch size
    :param output_dir: the output directory
    :param verbose: if true, the training is printed

    :return: the trained model
    """
    if verbose:
        print(f"[INFO] Training Set Size: {len(train_samples)}")
        print(f"[INFO] Validation Set Size: {len(val_samples)}")

    # load the model
    model = SentenceTransformer(model_path)

    # Make the dataloaders
    train_dataloader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    evaluator = evaluation.TripletEvaluator.from_input_examples(val_samples)

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    if loss_fn == "triplet":
        train_loss = losses.TripletLoss(model=model)
    elif loss_fn == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        raise NotImplementedError

    if verbose:
        print(f"[INFO] Training for {num_epochs} epochs")

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs, warmup_steps=warmup_steps, evaluator=evaluator,
            output_path=output_dir)

    if verbose:
        print(f"[INFO] Training finished!")
        print(f"[INFO] Saving model to {output_dir}")

    if save_loss:
        torch.save(train_loss, os.path.join(output_dir, 'loss', 'pytorch_loss.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", "-t", type=str, default="data/train.csv", help="path to the training data")
    parser.add_argument("--val_path", "-v", type=str, default="data/val.csv", help="path to the validation data")
    parser.add_argument("--model_path", "-m", type=str, default="sentence-transformers/paraphrase-mpnet-base-v2", help="path to the pretrained model")
    parser.add_argument("--num_epochs", "-ne", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="batch size")
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="output directory")
    parser.add_argument("--verbose", "-vb", action="store_true", help="if true, the training is printed")
    parser.add_argument("--anchor_column", "-a", type=str, default="question", help="the column name of the anchor column")
    parser.add_argument("--positive_cols", "-p", type=str, nargs='+', default=[], help="the columns that are used to generate the positive samples")
    parser.add_argument("--negative_cols", "-n", type=str, nargs='*', default=[], help="the columns that are used to generate the negative samples")
    parser.add_argument("--use_inbatch", "-u", action="store_true", help="if true, the samples are generated in batches, otherwise, \
        they are present in the negative_cols")
    parser.add_argument("--max_triplets_per_sample", "-mt", type=int, default=-1, help="the maximum number of triplets per sample")
    parser.add_argument("--save_loss", "-sl", action="store_true", help="if true, the loss is saved")
    parser.add_argument("--loss_fn", "-lf", type=str, default="triplet", help="the loss function to use")
    parser.add_argument("--seed", "-s", type=int, default=-1, help="the random seed")
    parser.add_argument("--num_seeds", "-ns", type=int, default=-1, help="the number of seeds")
    parser.add_argument("--stop_after", "-sa", type=int, default=0.2, help="the amount of batches to stop after")
    parser.add_argument("--continue_training", "-ct", action="store_true", help="if true, the training with best seed is continued")

    args = parser.parse_args()

    if args.seed == -1 and args.num_seeds == -1:
        num_seed = 5
    elif args.seed != -1 and args.num_seeds != -1:
        raise ValueError("You can either specify a seed or the number of seeds, not both")
    
    seed_list = [args.seed] if args.seed != -1 else [i for i in range(args.num_seeds)]
    

    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    for seed in seed_list:
        set_seed(seed)

        train_samples = generate_samples(df=train_df, anchor_column=args.anchor_column, positive_cols=args.positive_cols, 
                                        negative_cols=args.negative_cols, use_inbatch=args.use_inbatch, 
                                        max_triplets_per_sample=args.max_triplets_per_sample)

        val_samples = generate_samples(df=val_df, anchor_column=args.anchor_column, positive_cols=args.positive_cols,
                                        negative_cols=args.negative_cols, use_inbatch=args.use_inbatch,
                                        max_triplets_per_sample=args.max_triplets_per_sample)

        model = train_with_seed(seed=seed, train_samples=train_samples, val_samples=val_samples, model_path=args.model_path, num_epochs=1,
                    batch_size=args.batch_size, output_dir=args.output_dir, verbose=args.verbose, save_loss=args.save_loss, 
                    loss_fn=args.loss_fn, stop_after=args.stop_after)

    best_seed = get_best_seed(output_dir=args.output_dir)
    with open(os.path.join(args.output_dir, 'best_seed.json'), 'w') as f:
        info = {'best_seed': best_seed}
        json.dump(info, f)

    if args.continue_training:
        set_seed(best_seed)
        train_samples = generate_samples(df=train_df, anchor_column=args.anchor_column, positive_cols=args.positive_cols, 
                                            negative_cols=args.negative_cols, use_inbatch=args.use_inbatch, 
                                            max_triplets_per_sample=args.max_triplets_per_sample)

        val_samples = generate_samples(df=val_df, anchor_column=args.anchor_column, positive_cols=args.positive_cols,
                                            negative_cols=args.negative_cols, use_inbatch=args.use_inbatch,
                                            max_triplets_per_sample=args.max_triplets_per_sample)

        model = train(train_samples=train_samples, val_samples=val_samples, model_path=args.model_path, num_epochs=args.num_epochs,
                        batch_size=args.batch_size, output_dir=args.output_dir, verbose=args.verbose, save_loss=args.save_loss, 
                        loss_fn=args.loss_fn)

    """
    To train the model, run the following command:
    python train_biencoder.py -t data/train.csv -v data/val.csv -a question -p aug1 aug2 aug3 -n aug4 aug5 aug6 -ne 10 -b 16 -o output
    """