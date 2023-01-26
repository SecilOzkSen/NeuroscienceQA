import json
import random, string
import os

list_of_files_train = [
    "/home/secilsen/PycharmProjects/NeuroscienceQA/neuroscience_test_fixture/raw/train.json",
]
list_of_files_dev = ["/home/secilsen/PycharmProjects/NeuroscienceQA/neuroscience_test_fixture/raw/dev.json"]

dev_save = "/home/secilsen/PycharmProjects/NeuroscienceQA/policyqa_text_fixture/dataset/dev/dev.json"
train_save = "/home/secilsen/PycharmProjects/NeuroscienceQA/policyqa_text_fixture/dataset/train/train.json"

file_names = [
    "PolicyQA train - "
]
file_name_dev = ["PolicyQA dev - "]

def id_generator(N=24):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))

def squad_qa_formatter_list_only(file_path, file_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    data_list = []
    for i,data in enumerate(raw_data["data"]):
        dict_i = {"title" : f"{file_name}{i+1}",
        "paragraphs": []}
        for paragraph in data["paragraphs"]:
            sub_dict = {
                "context": paragraph["context"],
                "qas": []
            }
            for qa in paragraph["qas"]:
                if len(qa["generated_questions"]) == 0:
                    continue
                ssub_dict = {
                    "answers" : [],
                    "question" : qa["generated_questions"][0],
                    "id" : id_generator()
                }
                ssub_dict["answers"].append(dict(start=qa["qa_answers"][0]["handle_impossible_answer_True"]["start"],
                                            text=qa["qa_answers"][0]["handle_impossible_answer_True"]["answer"]))
                sub_dict["qas"].append(ssub_dict)
            dict_i["paragraphs"].append(sub_dict)
        data_list.append(dict_i)
    return data_list

def squad_formatter(list_of_files, titles):
    whole_list = []
    for file, title in zip(list_of_files, titles):
        data_list = squad_qa_formatter_list_only(file, title)
        whole_list.extend(data_list)
    return {
        "data" : whole_list
    }

def add_end_of_sentence(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    paragraphs = raw_data["data"][0]["paragraphs"]
    for paragraph in paragraphs:
        qas = paragraph['qas']
        if qas is None or len(qas) == 0:
            continue
        for qa in qas:
            answers = qa["answers"]
            if answers is None or len(answers) == 0:
                continue
            for answer in answers:
                answer['text'] = f"{answer['text']}."

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f)
def split_dev_train(file_path:str, train_percentage=0.8):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    data = raw_data["data"][0]["paragraphs"]
    training_size = int(len(data) * train_percentage)
    index_list = [i for i in range(len(data))]
    random.shuffle(index_list)
    training_indexes = index_list[0:training_size]
    dev_indexes = index_list[training_size+1:]
    training_json = {
        "data": [{"paragraphs": [data[i] for i in training_indexes]}]
    }
    dev_json = {
        "data": [{"paragraphs": [data[i] for i in dev_indexes]}]
    }
    return training_json, dev_json

def split_dev_train_binary_policy(file_path:str, train_percentage=0.8):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    training_size = int(len(data) * train_percentage)
    index_list = [i for i in range(len(data))]
    random.shuffle(index_list)
    training_indexes = index_list[0:training_size]
    dev_indexes = index_list[training_size+1:]
    training_json =  [data[i] for i in training_indexes]
    dev_json = [data[i] for i in dev_indexes]
    return training_json, dev_json


if __name__ == "__main__":
    #dev_list = squad_formatter(list_of_files_dev, file_name_dev)
#    add_end_of_sentence("/home/secilsen/PycharmProjects/SquadOperations/policyQA/bsbs_squad/full_sentence.json")
 #   training_json, dev_json = split_dev_train("/home/secilsen/PycharmProjects/SquadOperations/policyQA/bsbs_squad/full_sentence.json")
    training_json, dev_json = split_dev_train_binary_policy("/policyqa_text_fixture/dataset/binary_bsbs_full.json")

    with open(train_save, 'w', encoding='utf-8') as f:
        json.dump(training_json, f)

    with open(dev_save, 'w', encoding='utf-8') as f:
        json.dump(dev_json, f)





