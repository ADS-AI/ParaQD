import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import os, json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


class Metrics():

    @staticmethod
    def get_confusion_matrix(predicted, actual):
        conf_matrix = np.zeros((2, 2))
        for pred,act in zip(predicted, actual):
            conf_matrix[act, pred] += 1
        return conf_matrix


    @staticmethod     
    def get_TP(confusion_matrix, label):
        tp = confusion_matrix[label][label]
        return tp


    @staticmethod
    def get_FN(confusion_matrix,label):
        row = confusion_matrix[label,]
        row_truepositives = row[label]
        fn = row.sum() - row_truepositives
        return fn


    @staticmethod
    def get_FP(confusion_matrix,tag):
        col = confusion_matrix[:,tag]
        col_tp = col[tag]
        fp = col.sum() - col_tp  # sum of all values in column except tp
        return fp


    @staticmethod
    def macro_precision(conf_matrix):
        precision = 0.0
        for label in [0,1]:
            dividor= Metrics.get_TP(conf_matrix, label) + \
                    Metrics.get_FP(conf_matrix, label)
            if dividor != 0.0:
                precision += (Metrics.get_TP(conf_matrix, label))/dividor
        return (precision / 2)


    @staticmethod
    def macro_recall(conf_matrix):
        recall = 0.0
        for label in [0,1]:
            dividor = Metrics.get_TP(conf_matrix, label) + \
                    Metrics.get_FN(conf_matrix, label)
            if dividor != 0.0:
                recall += (Metrics.get_TP(conf_matrix,label)) / dividor
        return (recall / 2)


    @staticmethod
    def F1(precision,recall):
        return (2*precision*recall)/(precision+recall)

            
    @staticmethod
    def get_macro_metrics(predictions, test_labels):
        conf_matrix = Metrics.get_confusion_matrix(predictions, test_labels)
        precision = Metrics.macro_precision(conf_matrix)
        recall = Metrics.macro_recall(conf_matrix)
        f1_score = Metrics.F1(precision,recall)
        return (precision, recall, f1_score)


    @staticmethod
    def weighted_precision(conf_matrix, test_samples):
        accum =0
        for label in [0,1]:
            true_sample = [sample for sample in test_samples if sample==label]
            if (Metrics.get_TP(conf_matrix,label) + Metrics.get_FP(conf_matrix,label)) !=0:
                accum += float(len(true_sample)) *(Metrics.get_TP(conf_matrix,label)/ \
                            (Metrics.get_TP(conf_matrix,label) + Metrics.get_FP(conf_matrix,label)))
        precision =  accum / len(test_samples)
        return precision


    @staticmethod
    def weighted_recall(conf_matrix, test_samples):
        accum =0
        for label in [0,1]:
            true_sample = [sample for sample in test_samples if sample==label ]
            if (Metrics.get_TP(conf_matrix,label) + Metrics.get_FN(conf_matrix,label)) != 0:
                accum += float(len(true_sample)) * (Metrics.get_TP(conf_matrix,label) / \
                        (Metrics.get_TP(conf_matrix,label) + Metrics.get_FN(conf_matrix,label)))
        recall =  accum / len(test_samples)
        return recall


    @staticmethod
    def get_weighted_metrics(predictions,test_labels):
        conf_matrix = Metrics.get_confusion_matrix(predictions, test_labels)
        precision  = Metrics.weighted_precision(conf_matrix, test_labels)
        recall = Metrics.weighted_recall(conf_matrix, test_labels)
        f1_score = Metrics.F1(precision, recall)
        return (precision, recall, f1_score)


    @staticmethod
    def compute(df, column, thresh=0.5):
        metrics = {}
        preds = list(df[column].apply(lambda x: 1 if x>thresh else 0).values)
        true = list(df['label'].values)

        # Confusion matrix
        conf_matrix = Metrics.get_confusion_matrix(preds, true)

        # Calculating the macro metrics
        macro_precision, macro_recall, macro_f1 = Metrics.get_macro_metrics(preds, true)
        metrics["macro_precision"] = macro_precision; metrics["macro_recall"] = macro_recall
        metrics["macro_f1"] = macro_f1
        
        # Calculating the weighted metrics
        weighted_precision, weighted_recall, weighted_f1 = Metrics.get_weighted_metrics(preds, true)
        metrics["weighted_precision"] = weighted_precision; metrics["weighted_recall"] = weighted_recall
        metrics["weighted_f1"] = weighted_f1

        # Calculating the mean difference
        pos_mean = df[df["label"]==1][column].mean()
        neg_mean = df[df["label"]==0][column].mean()
        diff = pos_mean - neg_mean
        metrics["mean_difference"] = diff
        metrics["pos_mean"] = pos_mean; metrics["neg_mean"] = neg_mean

        return metrics, conf_matrix


class Evaluator():
    def __init__(self, model_path, dataset_path, out_path, method, column_1="question", 
                column_2="paraphrase", verbose=False, **kwargs):
        """
        Initialize the evaluator.

        @param model_path: Path to the sentence embedding model.
        @param dataset_path: Path to the dataset.
        @param out_path: Path to the output file.
        @param method: Name of the current method.
        @param column_1: Name of the column containing the first sentence.
        @param column_2: Name of the column containing the second sentence.
        @param verbose: Whether to print the progress.
        """
        self.verbose = verbose
        self.data = pd.read_csv(dataset_path)
        self.out_path = out_path
        self.method = method
        self.column_1 = column_1; self.column_2 = column_2
        if verbose:
            print("[INFO] Loading model...")
        self.model = SentenceTransformer(model_path)
        self.threshold = 0.5
        if "threshold" in kwargs: self.threshold = kwargs["threshold"]
        self.dataset_name = kwargs.get("dataset_name", "unknown")


    def __score_biencoder(self, sentence_1, sentence_2):
        """
        Helper function to find the cosine distance between two sentences.
        """
        emb_1 = self.model.encode(sentence_1,  convert_to_numpy = True)
        emb_2 = self.model.encode(sentence_2,  convert_to_numpy = True)
        return (1-cdist(emb_1.reshape(1,-1),emb_2.reshape(1,-1),'cosine'))[0][0]


    def __get_score(self):
        """
        Adds a column in data with the cosine scores for all pairs
        """
        if self.verbose: print("[INFO] Calculating the scores...")
        self.data[self.method] = self.data.apply(lambda x: self.__score_biencoder( \
                                            x[self.column_1], x[self.column_2]), axis=1)


    def __round_metrics(self, metrics):
        """
        rounds all the numbers to 3 decimal places
        """
        for key in metrics:
            metrics[key] = round(metrics[key], 3)

        return metrics


    def get_metrics(self):
        self.__get_score()

        if self.verbose: print("[INFO] Computing metrics...")
        metrics, conf_matrix = Metrics.compute(self.data, self.method, self.threshold)
        metrics = self.__round_metrics(metrics)
        metrics["method"] = self.method
        metrics["threshold"] = self.threshold
        return metrics, conf_matrix

    
    def __save_conf_matrix(self, conf_matrix):
        """
        Saves the confusion matrix in a csv file.
        """
        if self.verbose: print("[INFO] Saving the confusion matrix...")
        ax = sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, fmt=".2f")

        ax.set_title(f'{self.method}');
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['Invalid','Valid'])
        ax.yaxis.set_ticklabels(['Invalid','Valid'])

        directory = os.path.dirname(self.out_path)
        out_path = os.path.join(directory, f"{self.method}_{self.dataset_name}.png")
        plt.savefig(out_path)


    def __write_metrics_json(self):
        info = []

        metrics, conf_matrix = self.get_metrics()
        self.__save_conf_matrix(conf_matrix)

        if os.path.exists(self.out_path):
            with open(self.out_path, "r") as f:
                info = json.load(f)
            if type(info) != list:
                info = [info]

        info.append(metrics)

        if self.verbose: print("[INFO] Writing metrics to file...")
        with open(self.out_path, "w") as f:
            json.dump(info, f)


    def __write_metrics_csv(self):
        metrics, conf_matrix = self.get_metrics()
        self.__save_conf_matrix(conf_matrix)

        if os.path.exists(self.out_path):
            df = pd.read_csv(self.out_path)
            columns = set(df.columns); metrics_keys = set(metrics.keys())
            if columns != metrics_keys:
                raise Exception("[ERROR] Metrics columns do not match")
            metrics_df = pd.DataFrame(metrics, index=[0])
            df = pd.concat([df, metrics_df], ignore_index=True).reset_index(drop=True)

        else:
            df = pd.DataFrame(metrics, index=[0])

        if self.verbose: print("[INFO] Writing metrics to file...")
        df.to_csv(self.out_path, index=False)


    def save_metrics(self):
        if self.out_path.endswith(".json"):
            self.__write_metrics_json()

        elif self.out_path.endswith(".csv"):
            self.__write_metrics_csv()

        else:
            raise Exception("[ERROR] Output file must be a .json or .csv file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="Path to the sentence embedding model.")
    parser.add_argument("--dataset_path", "-d", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--out_path", "-o", type=str, required=True,
                        help="Path to the output file.")
    parser.add_argument("--method", "-mth", type=str, default="ParaQD",
                        help="Name of the current method.")
    parser.add_argument("--column_1", "-c1", type=str, default="question",
                        help="Name of the column containing the first sentence.")
    parser.add_argument("--column_2", "-c2", type=str, default="paraphrase",
                        help="Name of the column containing the second sentence.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Whether to print the progress.")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Threshold to use for the evaluation.")
    parser.add_argument("--dataset_name", "-n", type=str, default="unknown",
                        help="Name of the dataset.")


    args = parser.parse_args()

    evaluator = Evaluator(args.model_path, args.dataset_path, args.out_path, args.method,
                            args.column_1, args.column_2, args.verbose, threshold=args.threshold,
                            dataset_name=args.dataset_name)
    evaluator.save_metrics()
