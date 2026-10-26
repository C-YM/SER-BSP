import os
from dataclasses import dataclass
from glob import glob
from os import makedirs
from os.path import join
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
import torchaudio
import yaml
import evaluate
import torch.nn.functional as F
from datasets import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
)
from transformers.file_utils import ModelOutput
from transformers.models.wavlm.modeling_wavlm import (
   WavLMModel,
   WavLMPreTrainedModel,
)


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_checkpoint = "facebook/hubert-large-ls960-ft"
# model_checkpoint = "microsoft/wavlm-large"
processor = Wav2Vec2Processor.from_pretrained(model_checkpoint, ignore_mismatched_sizes=True)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        # d_type = torch.long if isinstance(label_features[0], int) else torch.float
        d_type = torch.long

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "labels" in features[0]:
            label_features = [feature["labels"] for feature in features]
            batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    # print("path:", path, "shape:", speech_array.shape)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech = resampler(speech_array)
    if len(speech.shape) == 2:
        speech = speech.mean(0)
    speech = speech.squeeze().numpy()
    return speech


def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples["filename_full"]]
    result = processor(speech_list, sampling_rate=16000)
    if "label" in examples:
        target_list = [
            int(label) for label in examples["label"]
        ]  # Do any preprocessing on your float/integer data
        result["labels"] = list(target_list)
    return result


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class WavLMClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class WavLMForSpeechClassification(WavLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.hubert =WavLMModel(config)
        self.classifier =WavLMClassificationHead(config)
        self.layer_weights = torch.nn.Parameter(torch.ones(24))
        self.init_weights()

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
            labels=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states[1:]
        hidden_states = torch.stack(hidden_states, dim=1)
        # # print(hidden_states.shape)
        weights = torch.nn.functional.softmax(self.layer_weights, dim=-1)
        weights = weights.view(-1, 24, 1, 1)
        hidden_states = (hidden_states * weights).sum(dim=1)
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    print("regression")
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    print("single_label_classification")
                    self.config.problem_type = "single_label_classification"
                else:
                    print("multi_label_classification")
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = F.mse_loss
                loss = loss_fct(logits.view(-1, self.num_labels), labels.unsqueeze(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )


def compute_metrics(eval_pred):
    """Computes spearman and pearson on predictions"""
    recall = recall_metric.compute(
        predictions=np.argmax(eval_pred.predictions.squeeze(), axis=-1),
        references=eval_pred.label_ids,
        average="macro",
    )
    f1 = f1_metric.compute(
        predictions=np.argmax(eval_pred.predictions.squeeze(), axis=-1),
        references=eval_pred.label_ids,
        average="macro",
    )
    # f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    return {**recall, **f1}


if __name__ == "__main__":
    lr = 0.0000001

    with open("params.yaml") as f:
        _params = yaml.safe_load(f)
        params = _params["wav2vec"]
        target = _params["target"]

    label_dir = "E:/work/USERNAME/DEMOS/lab/diff"
    result_dir = f"F:/work/USERNAME/DEMOS/result/diff/fine_tune/24_weight_hubert/{lr}_4"
    logging_dir = "./tb_logs/wav2vec"

    model_checkpoint = params["model"]

    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sr = 16000
    model_name = model_checkpoint.split("/")[-1]

    all_files = list(glob(r"E:/work/USERNAME/DEMOS/wav_all/diff/wav/*.wav", recursive=True))
    label_list = [0]

    makedirs(result_dir, exist_ok=True)
    makedirs(logging_dir, exist_ok=True)

    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    df_train = pd.read_csv(f"{label_dir}/train.csv", encoding="utf-8")
    df_dev = pd.read_csv(f"{label_dir}/devel.csv", encoding="utf-8")

    df_train["filename_full"] = "E:/work/USERNAME/DEMOS/wav_all/diff/wav\\" + df_train["filename"]
    df_train = df_train[df_train["filename_full"].isin(all_files)]
    df_dev["filename_full"] = "E:/work/USERNAME/DEMOS/wav_all/diff/wav\\" + df_dev["filename"]
    df_dev = df_dev[df_dev["filename_full"].isin(all_files)]

    label_encoder = LabelEncoder()
    df_train["label"] = label_encoder.fit_transform(df_train[target])
    df_dev["label"] = label_encoder.transform(df_dev[target])

    train_dataset = Dataset.from_pandas(df_train)

    dev_dataset = Dataset.from_pandas(df_dev)

    config = AutoConfig.from_pretrained(
        model_checkpoint,
        num_labels=len(label_encoder.classes_),
        problem_type=None,
        ignore_mismatched_sizes=True
    )
    setattr(config, "pooling_mode", params["pooling"])

    processor = Wav2Vec2Processor.from_pretrained(model_checkpoint, ignore_mismatched_sizes=True)
    target_sampling_rate = processor.feature_extractor.sampling_rate

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True, max_length=2 * target_sampling_rate
    )

    # print("train", train_dataset)
    train_dataset = train_dataset.map(
        function=preprocess_function, batch_size=100, batched=True, num_proc=1
    )

    dev_dataset = dev_dataset.map(
        function=preprocess_function, batch_size=100, batched=True, num_proc=1
    )

    model =WavLMForSpeechClassification.from_pretrained(
        model_checkpoint,
        config=config,
    )
    if params["freezeExtractor"]:
        model.freeze_feature_extractor()
    if params["freezeTransformer"]:
        model.freeze_transformer()

    args = TrainingArguments(
        result_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=params["epochs"],
        warmup_ratio=0.1,
        logging_steps=10,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="recall",
        push_to_hub=False,
        gradient_checkpointing=True,
        save_total_limit=2,
        disable_tqdm=True,
        logging_dir=f"{logging_dir}/{datetime.datetime.now().strftime('%h%d_%Y-%H-%M-%S')}",
    )
    print(args.device)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    dev_predictions = trainer.predict(dev_dataset)
    dev_df = pd.DataFrame(
        {
            "filename": dev_dataset["filename"],
            "prediction": label_encoder.inverse_transform(
                np.argmax(dev_predictions.predictions.squeeze(), axis=-1)
            ),
            "true": label_encoder.inverse_transform(
                dev_predictions.label_ids.squeeze()
            ),
        }
    )
    dev_df.to_csv(join(result_dir, str(lr) + "_predictions_devel_4.csv"), index=False)
    print(compute_metrics(dev_predictions))
