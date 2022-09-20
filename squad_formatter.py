import json
import random, string

list_of_files_train = [
    "/home/secilsen/PycharmProjects/NeuroscienceQA/neuroscience_test_fixture/dataset/raw/the_brain_squad.json",
    "/home/secilsen/PycharmProjects/NeuroscienceQA/neuroscience_test_fixture/dataset/raw/the_neuroscience_of_intelligence_squad.json"
]
list_of_files_dev = ["/home/secilsen/PycharmProjects/NeuroscienceQA/neuroscience_test_fixture/dataset/raw/incognito_squad.json"]

dev_save = "/home/secilsen/PycharmProjects/NeuroscienceQA/neuroscience_test_fixture/dataset/dev/dev.json"
train_save = "/home/secilsen/PycharmProjects/NeuroscienceQA/neuroscience_test_fixture/dataset/train/train.json"

file_names = [
    "The Brain - ",
    "The Neuroscience of Intelligence - "
]
file_name_dev = ["The Incognito - "]


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

if __name__ == "__main__":
    dev_list = squad_formatter(list_of_files_dev, file_name_dev)
    with open(dev_save, 'w', encoding='utf-8') as f:
        json.dump(dev_list, f)





