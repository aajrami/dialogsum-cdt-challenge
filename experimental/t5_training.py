import numpy as np
import os
import os.path as op
import pandas as pd
from argparse import ArgumentParser
from rich.table import Column, Table
from rich import box
from rich.console import Console
from transformers import T5Tokenizer, T5ForConditionalGeneration


import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from t5_dataset import T5Dataset


model_classes = {"T5": T5ForConditionalGeneration, "Bart": BartForConditionalGeneration}
tokenizer_classes = {"T5": T5Tokenizer, "Bart": BartTokenizer}

console = Console(record=True)

parser = ArgumentParser()
parser.add_argument("--MODEL",  default="t5-small", type=str)   
parser.add_argument("--TRAIN_EPOCHS", default=1, type=int)  
parser.add_argument("--PRETRAIN_EPOCHS", default=0, type=int)  
parser.add_argument("--LEARNING_RATE", default=1e-3, type=float)  
parser.add_argument("--RUN_NAME", default=None, type=str)
parser.add_argument("--BATCH_SIZE", default=8, type=int)
parser.add_argument("--MODEL_TYPE", default="T5", type=str)

args = parser.parse_args()

if args.RUN_NAME != None:
    RUN_NAME = args.RUN_NAME
else:
    RUN_NAME = f"unnamed_run_{args.MODEL}"


# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

torch.cuda.empty_cache()

def get_scores_df(outputs_csv_path, epoch):
    from nlgeval import NLGEval
    nlg_eval = NLGEval()  # loads the models
    results_df = pd.read_csv(outputs_csv_path)

    references = [list(results_df["Actual Text"])]
    hypotheses = list(results_df["Generated Text"])

    metrics_dict = nlg_eval.compute_metrics(references, hypotheses)
    metrics_dict["epoch"] = epoch
    metrics_df = pd.DataFrame([metrics_dict])
    return(metrics_df)


def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

def validate(epoch, tokenizer, model, device, loader):

    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
    return predictions, actuals


def load_model_and_tokenizer(model_class, tokenizer_class, model_name):
    
    if not op.exists("transformers_cache"):
        os.mkdir("transformers_cache")
    cache_path = op.join("transformers_cache", model_name)

    if op.exists(cache_path):
        model = model_class.from_pretrained(cache_path)
        tokenizer = tokenizer_class.from_pretrained(cache_path + "-tokenizer")
    else: 
        model = model_class.from_pretrained(model_name)
        tokenizer = tokenizer_class.from_pretrained(model_name)
        
        model.save_pretrained(cache_path)
        tokenizer.save_pretrained(cache_path + "-tokenizer")

    
    return model, tokenizer


def load_tokenizer(tokenizer_class, model_name):
    
    if not op.exists("transformers_cache"):
        os.mkdir("transformers_cache")
    cache_path = op.join("transformers_cache", model_name)

    if op.exists(cache_path):
        tokenizer = tokenizer_class.from_pretrained(cache_path + "-tokenizer")
    else: 
        tokenizer = tokenizer_class.from_pretrained(model_name)
      
    
    return tokenizer




def T5Trainer(
    train_dataset, val_dataset, 
    source_text, target_text, 
    model_params,
    experiment_name, 
    model=None, pretrain=False, 
    model_class=T5ForConditionalGeneration, tokenizer_class=T5Tokenizer
):

    """
    T5 trainer

    """

    output_dir = os.path.join("experiments", experiment_name)
    os.makedirs(output_dir, exist_ok=True) 



    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
  
    if model is None:
        model, tokenizer = load_model_and_tokenizer(model_class=model_class,
                                                tokenizer_class=tokenizer_class,
                                                model_name=model_params["MODEL"])
    else:
        print("USING QUESTION PRETRAINED MODEL")
        tokenizer = load_tokenizer(tokenizer_class=tokenizer_class,
                                    model_name=model_params["MODEL"])
 
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    train_dataset = train_dataset[[source_text, target_text]]
    val_dataset = val_dataset[[source_text, target_text]]
    display_df(train_dataset.head(2))


    #console.print(f"FULL Dataset: {dataset.shape}")
    # console.print(f"TRAIN Dataset: {train_dataset.shape}")
    # console.print(f"VAL Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = A2SDataset(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = A2SDataset(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    scores_path = op.join(output_dir, "scores.csv")

    import time

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        loop_start_time = time.time()
        #clear_output()
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        outputs_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        if pretrain==True:
            csv_path = op.join(output_dir, f"pretrain_epoch_{epoch+1}.csv")
        else:
            csv_path = op.join(output_dir, f"{epoch+1}_epochs.csv")
        outputs_df.to_csv(csv_path)
        train_finish_time = time.time()
    
        console.log(f"epoch training time: {(train_finish_time - loop_start_time)/60} minutes")

        #console.log(f"Initiating metrics")
        #scores_df = get_scores_df(csv_path, epoch+1)
        #if not epoch == 0:
        #    existing_scores = pd.read_csv(scores_path)
        #    scores_df = pd.concat([existing_scores, scores_df])
        #scores_df.to_csv(scores_path)
        #console.log(f"Scores saved at {scores_path}")
        #metrics_finished_time = time.time()
        #console.log(f"Metric calculation time : {(metrics_finished_time - train_finish_time)/60} minutes")

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, f"model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    for epoch in range(model_params["VAL_EPOCHS"]):
        #clear_output()
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)


    console.save_text(os.path.join(output_dir, f"logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )

    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

    return model



# let's define model parameters specific to T5
pretrain_params = {
    "MODEL": args.MODEL,  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": args.BATCH_SIZE,  # training batch size
    "VALID_BATCH_SIZE": args.BATCH_SIZE,  # validation batch size
    "TRAIN_EPOCHS": args.PRETRAIN_EPOCHS,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": args.LEARNING_RATE,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}


# T5 accepts prefix of the task to be performed:
# Since we are summarizing, let's add summarize to source text as a prefix
# train_df = pd.read_csv(op.join("data_augmented", "train_augmented.csv"))
# val_df = pd.read_csv(op.join("data_augmented", "val_augmented.csv"))
# train_df_small = train_df.truncate(after=30)

# train_df["argument"] = "summarize: " + train_df["argument"]


model_params = {
    "MODEL": args.MODEL,  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": args.BATCH_SIZE,  # training batch size
    "VALID_BATCH_SIZE": args.BATCH_SIZE,  # validation batch size
    "TRAIN_EPOCHS": args.TRAIN_EPOCHS,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": args.LEARNING_RATE,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}



T5Trainer(
    train_dataset=train_df,
    val_dataset=val_df,
    source_text="argument",
    target_text="questions",
    model_params=model_params,
    experiment_name=RUN_NAME,
    model=model,
    model_class=model_classes[args.MODEL_TYPE],
    tokenizer_class=tokenizer_classes[args.MODEL_TYPE]
)

