from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM, TrainingArguments, Trainer
import math
import os

model_checkpoint = "facebook/contriever-msmarco"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
chunk_size = 128
EPOCH = 50
os.environ["WANDB_DISABLED"] = "true"

def csv_to_dataset_dict(path):
    full_df = pd.read_csv(
        path, delimiter='|')
    full_df = full_df.drop(labels=['answer'], axis=1)
    indexes = list(range(len(full_df.index)))

    train_index, valid_index, train_question, valid_question = train_test_split(indexes,
                                                                                full_df['question'].tolist(),
                                                                                test_size=0.1, random_state=8)
    context_list = full_df['context'].tolist()
    train_context = [context_list[i] for i in train_index]
    valid_context = [context_list[i] for i in valid_index]

    train_dataframe = pd.DataFrame({"text": train_context})
    validation_dataframe = pd.DataFrame({"text": valid_context})

    train_dataset = Dataset.from_pandas(train_dataframe)
    valid_dataset = Dataset.from_pandas(validation_dataframe)

    ds = DatasetDict()

    ds['train'] = train_dataset
    ds['validation'] = valid_dataset

    return ds

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if iword_idst's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

basecamp_dataset = csv_to_dataset_dict("/home/secilsen/PycharmProjects/NeuroscienceQA/data/basecamp.csv")
tokenized_datasets = basecamp_dataset.map(
    tokenize_function, batched=True,
    remove_columns=["text"]
)
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
training_args = TrainingArguments(
    num_train_epochs=EPOCH,
    output_dir="domain-adapted-contriever",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=5,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    disable_tqdm=False,
    warmup_steps=500,
    do_eval=True,
    fp16=True,
    lr_scheduler_type="linear",
    optim='adamw_hf',
    logging_dir="domain-adapted-contriever" + '/logging',
    run_name='obss-contriever-finetuned',
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)
trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")