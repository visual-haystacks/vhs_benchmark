import os
import json
import nltk
import shutil
import logging
import numpy as np
import copy


class CostMeter:
    def __init__(self, model_name, costs=None) -> None:
        self._prompt_tokens_used = 0
        self._completion_tokens_used = 0
        if costs is not None:
            self._prompt_tokens_cost = costs["input_token_cost"]
            self._completion_tokens_cost = costs["output_token_cost"]
        else:
            # cost per 1M tokens
            if "gpt-4o" in model_name:
                self._prompt_tokens_cost = 5
                self._completion_tokens_cost = 15
            elif "gemini-1.5-pro" in model_name:
                self._prompt_tokens_cost = 3.5
                self._completion_tokens_cost = 7
            elif "gemini-1.0" in model_name:
                self._prompt_tokens_cost = 0.5
                self._completion_tokens_cost = 1.5
            elif "claude-3-haiku" in model_name:
                self._prompt_tokens_cost = 0.25
                self._completion_tokens_cost = 1.25

    def update(self, usage) -> None:
        self._prompt_tokens_used += usage.prompt_tokens if usage else 0
        self._completion_tokens_used += usage.completion_tokens if usage else 0

    @property
    def cost(self) -> float:
        input_cost = self._prompt_tokens_used * self._prompt_tokens_cost * 1e-6
        output_cost = self._completion_tokens_used * self._completion_tokens_cost * 1e-6
        return input_cost + output_cost


def create_directory(directory):
    # Check if the directory already exists
    if os.path.exists(directory):
        # Prompt the user for a decision
        response = (
            input(
                f"The directory '{directory}' already exists. Do you want to delete it? (y/n): "
            )
            .strip()
            .lower()
        )
        if response == "y":
            # Delete the directory if the user confirms
            shutil.rmtree(directory)  # This removes an empty directory
            logging.info(f"Deleted the directory: {directory}")
            # Re-create the directory
            os.makedirs(directory)
            logging.info(f"Created the directory: {directory}")
        elif response == "n":
            logging.info("Continuing without deleting the directory.")
        else:
            logging.info("Invalid input. Exiting.")
            return
    else:
        # Create the directory if it does not exist
        os.makedirs(directory)
        logging.info(f"Created the directory: {directory}")


def report_number(bucket_result, bucket_total, acc_stat, comp_stat, mode):
    table_str = "Result\n"
    if mode == "single_needle":
        # Header
        table_str += f"{mode} -- Needle Position Analysis\n"
        table_str += "Needle Position | Bucket Stat. (Accuracy / Compliance) | Bucket Total | Average Acc / Comp \n"
        table_str += "-" * 100 + "\n"  # Adjusted length for the new column

        # Rows
        N = len(bucket_total)
        bucket_correct = bucket_result[0]
        bucket_compliance = bucket_result[1]
        for i in range(N):
            acc = bucket_correct[i] / bucket_total[i] if bucket_total[i] > 0 else 0
            comp = bucket_compliance[i] / bucket_total[i] if bucket_total[i] > 0 else 0
            table_str += f"{i:^17} | {bucket_correct[i]:^16}/{bucket_compliance[i]:^17} | {bucket_total[i]:^12} | {acc:^10.2%}/{comp:^10.2%}\n"

        # Summary
        table_str += "-" * 100 + "\n"  # Adjusted length for the new column
    # For both modes, we need to report the average accuracy and standard deviation
    table_str += f"- Accuracy: {acc_stat[0]:.2%} ± {acc_stat[1]:.2%}\n"
    table_str += f"- Compliance: {comp_stat[0]:.2%} ± {comp_stat[1]:.2%}\n"
    return table_str


def run_eval(dirname, needle_mode):
    all_files = [f for f in os.listdir(dirname) if f.endswith(".json")]
    data = [json.load(open(os.path.join(dirname, f), "r")) for f in all_files]
    num_images = len(data[0]["result"]["image_paths"])
    if needle_mode == "single_needle":
        interval = max(int(num_images // 10), 1)
        bucket_correct = [0] * interval
        bucket_compliance = [0] * interval
        bucket_total = [0] * interval
    else:
        bucket_correct = []
        bucket_compliance = []
        bucket_total = []
    acc_avgs = []
    comp_avgs = []
    N = len(data)
    all_indices = list(range(N))

    # Bootstraping Evaluation
    for _ in range(100):
        correct = 0
        compliance = 0
        total = 0
        indices = np.random.choice(all_indices, N, replace=True)
        for i in indices:
            gt = data[i]["conversations"][1]["value"]
            pred = data[i]["result"]["response"]
            if needle_mode == "single_needle":
                pos_idx = data[i]["result"]["image_paths"].index(
                    data[i]["pos_image"][0]
                )
                bucket_idx = pos_idx // interval
                if "yes" in nltk.word_tokenize(
                    pred.lower()
                ) or "no" in nltk.word_tokenize(pred.lower()):
                    compliance += 1
                    bucket_compliance[bucket_idx] += 1
                if gt in nltk.word_tokenize(pred.lower()):
                    correct += 1
                    bucket_correct[bucket_idx] += 1
                total += 1
                bucket_total[bucket_idx] += 1
            else:
                if "yes" in nltk.word_tokenize(
                    pred.lower()
                ) or "no" in nltk.word_tokenize(pred.lower()):
                    compliance += 1
                if gt in nltk.word_tokenize(pred.lower()):
                    correct += 1
                total += 1

        acc_avg = correct / total
        comp_avg = compliance / total
        acc_avgs.append(acc_avg)
        comp_avgs.append(comp_avg)
    acc_avg = np.mean(acc_avgs)
    acc_std = np.std(acc_avgs)
    comp_avg = np.mean(comp_avgs)
    comp_std = np.std(comp_avgs)
    logging.info(
        report_number(
            (bucket_correct, bucket_compliance),
            bucket_total,
            (acc_avg, acc_std),
            (comp_avg, comp_std),
            needle_mode,
        )
    )


def run_detailed_eval(dirname, needle_mode):
    all_files = [f for f in os.listdir(dirname) if f.endswith(".json")]
    data = [json.load(open(os.path.join(dirname, f), "r")) for f in all_files]
    num_responses = len(data[0]["response"])
    assert needle_mode == "single_needle"

    corrects = []
    corrects_avgs = []

    compliances = []
    compliance_avgs = []
    totals = []
    N = len(data)
    all_indices = list(range(N))

    # Bootstraping Evaluation
    for _ in range(100):
        correct = [0] * num_responses
        compliance = [0] * num_responses
        total = [0] * num_responses

        indices = np.random.choice(all_indices, N, replace=True)
        for i in indices:
            gt = data[i]["conversations"][1]["value"]
            preds = data[i]["response"]
            for j, pred in enumerate(preds):
                if len(pred) == 0:
                    total[j] += 1
                    continue
                if "yes" in nltk.word_tokenize(
                    pred.lower()
                ) or "no" in nltk.word_tokenize(pred.lower()):
                    compliance[j] += 1
                if gt in nltk.word_tokenize(pred.lower()):
                    correct[j] += 1
                total[j] += 1

        avg = np.sum(correct) / np.sum(total)
        corrects_avgs.append(avg)
        corrects.append(correct)

        compliance_avg = np.sum(compliance) / np.sum(total)
        compliance_avgs.append(compliance_avg)
        compliances.append(compliance)
        totals.append(total)
    acc_avg = np.mean(corrects_avgs)
    acc_std = np.std(corrects_avgs)
    corrects = np.array(corrects)

    comp_avg = np.mean(compliance_avgs)
    comp_std = np.std(compliance_avgs)
    compliances = np.array(compliances)
    totals = np.array(totals)
    bucket_correct = np.sum(corrects, axis=0)
    bucket_total = np.sum(totals, axis=0)

    bucket_compliance = np.sum(compliances, axis=0)
    logging.info(
        report_number(
            (bucket_correct, bucket_compliance),
            bucket_total,
            (acc_avg, acc_std),
            (comp_avg, comp_std),
            needle_mode,
        )
    )
