from operators import DataGenerator
import argparse
import pandas as pd

"""
Available Operators:

positive_ops=["BackTranslate", "SameSentence", "Num2Words", "UnitExpansion"] 
negative_ops=["MostImportantPhraseRemover", "DeleteLastSentence", "ReplaceNamedEntities",
            "ReplaceNumericalEntities", "Pegasus", "ReplaceUnits", "NegateQuestion"]
"""

def augment(df, input_col, data_generator, positive_ops=[], negative_ops=[], output_dir=None):
    """"
    augment the dataset using the given ops
    :param: df: the dataframe to augment
    :param: input_col: the column to augment
    :param: data_generator: the data generator
    :param: positive_ops: the list of positive ops
    :param: negative_ops: the list of negative ops
    """
    positive_cols = [f"positive_{i}" for i in range(len(positive_ops))]
    negative_cols = [f"negative_{i}" for i in range(len(negative_ops))]

    for col in positive_cols.extend(negative_cols):
        if col in df.columns:
            raise ValueError(f"column {col} already exists in the dataframe")

    df[:, positive_cols] = ""
    df[:, negative_cols] = ""

    # generate the samples
    for idx in range(len(df)):
        text = df.loc[idx, input_col]
        positives, negatives = data_generator.generate(text, positive_ops, negative_ops)
        df.loc[idx, positive_cols] = positives
        df.loc[idx, negative_cols] = negatives
        
    if output_dir is not None:
        df.to_csv(output_dir, index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", "-i", type=str, default="data/train.csv", help="path to the input data")
    parser.add_argument("--input_col", "-ic", type=str, default="question", help="the column to augment")
    parser.add_argument("--positive_ops", "-p", type=str, nargs="+", required=True, help="The positive operators to use")
    parser.add_argument("--negative_ops", "-n", type=str, nargs="+", required=True, help="The negative operators to use")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="The output directory")

    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    data_generator = DataGenerator()

    df = augment(df, args.input_col, data_generator, args.positive_ops, args.negative_ops, args.output_dir)


