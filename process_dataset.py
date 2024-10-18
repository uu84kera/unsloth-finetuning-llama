""" this script is used to process the dataset """
import json
from glob import glob

def read_data(paths):
    """ read the data from json and combine it into single one """
    questions, answers = [], []
    for path in paths:
        with open("{}".format(path), "rb") as f:
            data_ = json.load(f)
            for qa in data_:
                for key in qa:
                    if key.startswith('annotated_result'):
                        questions.append(qa["original_data"])
                        answers.append(qa[key])
    return questions, answers

if __name__ == "__main__":
    data_path = "cross_validation_datasets"
    json_paths = glob("{}/train_fold_1.json".format(data_path))
    print(json_paths)
    questions, answers = read_data(json_paths)
    new_json_format = []
    for question, answer in zip(questions, answers):
        qa_pairs = {}
        qa_pairs["instruction"] = "请替我解答以下句子的真实意义，并且以尽可能简洁的语言回答."
        qa_pairs["input"] = question
        qa_pairs["output"] = answer
        new_json_format.append(qa_pairs)
    with open("ruozhiba_processed_train1.json", "w") as file:
        json.dump(new_json_format, file)
