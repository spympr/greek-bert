import pytorch_wrapper as pw
import torch
import os
import uuid

from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from itertools import product
from transformers import AutoTokenizer, AutoModel, AdamW
from functools import partial

from ...utils import loss_wrappers, evaluators
from .model import NERBERTModel
from .dataset import NERBERTDataset


class NERBERTSystemWrapper:

    def __init__(self, pretrained_bert_name, preprocessing_function, bert_like_special_tokens, model_params):

        self._pretrained_bert_name = pretrained_bert_name
        bert_model = AutoModel.from_pretrained(pretrained_bert_name)
        model = NERBERTModel(bert_model, **model_params)
        self._preprocessing_function = preprocessing_function
        self._bert_like_special_tokens = bert_like_special_tokens

        if torch.cuda.is_available():
            print("\nRunning on cuda with model:", pretrained_bert_name,"\n")
            self._system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cuda'))
        else:
            print("\nRunning on cpu with model:", pretrained_bert_name,"\n")
            self._system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cpu'))


    def train(self,
              train_dataset_file,
              val_dataset_file,
              lr,
              batch_size,
              grad_accumulation_steps,
              run_on_multi_gpus,
              verbose=True,
              seed=0):
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(self._pretrained_bert_name)

        train_dataset = NERBERTDataset(
            train_dataset_file,
            tokenizer,
            self._bert_like_special_tokens,
            self._preprocessing_function
        )

        val_dataset = NERBERTDataset(
            val_dataset_file,
            tokenizer,
            self._bert_like_special_tokens,
            self._preprocessing_function
        )

        self._train_impl(
            train_dataset,
            val_dataset,
            lr,
            batch_size,
            grad_accumulation_steps,
            run_on_multi_gpus,
            tokenizer.pad_token_id,
            verbose
        )

    def _train_impl(self,
                    train_dataset,
                    val_dataset,
                    lr,
                    batch_size,
                    grad_accumulation_steps,
                    run_on_multi_gpus,
                    pad_value,
                    verbose=True):

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size,
            collate_fn=partial(NERBERTDataset.collate_fn, pad_value=pad_value)
        )

        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size,
            collate_fn=partial(NERBERTDataset.collate_fn, pad_value=pad_value)
        )

        loss_wrapper = loss_wrappers.MaskedTokenLabelingGenericPointWiseLossWrapper(nn.CrossEntropyLoss())
        optimizer = AdamW(self._system.model.parameters(), lr=lr)

        base_es_path = f'/tmp/{uuid.uuid4().hex[:30]}/'
        os.makedirs(base_es_path, exist_ok=True)

        train_method = self._system.train_on_multi_gpus if run_on_multi_gpus else self._system.train

        _ = train_method(
            loss_wrapper,
            optimizer,
            train_data_loader=train_dataloader,
            evaluation_data_loaders={'val': val_dataloader},
            evaluators={
                'macro-f1': evaluators.MultiClassF1EvaluatorMaskedTokenEntityLabelingEvaluator(train_dataset.I2L)
            },
            gradient_accumulation_steps=grad_accumulation_steps,
            callbacks=[
                pw.training_callbacks.EarlyStoppingCriterionCallback(
                    patience=3,
                    evaluation_data_loader_key='val',
                    evaluator_key='macro-f1',
                    tmp_best_state_filepath=f'{base_es_path}/temp.es.weights'
                )
            ],
            verbose=verbose
        )

    def evaluate(self, eval_dataset_file, batch_size, run_on_multi_gpus, verbose=True):
        tokenizer = AutoTokenizer.from_pretrained(self._pretrained_bert_name)
        eval_dataset = NERBERTDataset(
            eval_dataset_file,
            tokenizer,
            self._bert_like_special_tokens,
            self._preprocessing_function
        )
        return self._evaluate_impl(eval_dataset, batch_size, run_on_multi_gpus, tokenizer.pad_token_id, verbose)

    def _evaluate_impl(self, eval_dataset, batch_size, run_on_multi_gpus, pad_value, verbose=True):

        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=batch_size,
            collate_fn=partial(NERBERTDataset.collate_fn, pad_value=pad_value)
        )

        evals = {
            'macro-prec': evaluators.MultiClassPrecisionEvaluatorMaskedTokenEntityLabelingEvaluator(eval_dataset.I2L),
            'macro-rec': evaluators.MultiClassRecallEvaluatorMaskedTokenEntityLabelingEvaluator(eval_dataset.I2L),
            'macro-f1': evaluators.MultiClassF1EvaluatorMaskedTokenEntityLabelingEvaluator(eval_dataset.I2L),
            'micro-prec': evaluators.MultiClassPrecisionEvaluatorMaskedTokenEntityLabelingEvaluator(
                eval_dataset.I2L,
                average='micro'
            ),
            'micro-rec': evaluators.MultiClassRecallEvaluatorMaskedTokenEntityLabelingEvaluator(
                eval_dataset.I2L,
                average='micro'
            ),
            'micro-f1': evaluators.MultiClassF1EvaluatorMaskedTokenEntityLabelingEvaluator(
                eval_dataset.I2L,
                average='micro'
            ),
        }

        from seqeval.metrics import classification_report

        print(type(eval_dataloader))
        true_labels = []
        for batch_idx, samples in enumerate(eval_dataloader):
            # print((samples['target']))
            true_labels.append(samples['target'].tolist())

        print()
        true_labels = [item for sublist in true_labels for item in sublist]
        print(len(true_labels))
        print(len(true_labels[1]))
        print((true_labels[1]))

        predictions = self._system.predict(eval_dataloader)

        print()
        total_predictions = []
        for k,i in enumerate(predictions['outputs']):
            i = torch.FloatTensor(i)
            if k==1:
                print(i)
                print(i.argmax().numpy())
            total_predictions.append(i.argmax().numpy())

        print()
        print(len(total_predictions))
        print((total_predictions[1]))
        # print(predictions)
        # print(type(predictions['outputs'][0]))
        # print(len(predictions['outputs']))
        # print(len(predictions['outputs'][0]))
        # print((predictions['outputs'][0]))
        # print(predictions)
        # for i in predictions['outputs']:
        # print((i))

        
        print(classification_report(true_labels,total_predictions),target_names=eval_dataset.I2L)
        print(set(true_labels)-set(total_predictions))

        if run_on_multi_gpus:
            return self._system.evaluate_on_multi_gpus(eval_dataloader, evals, verbose=verbose)
        else:
            return self._system.evaluate(eval_dataloader, evals, verbose=verbose)

    def save_model_state(self, path):
        self._system.save_model_state(path)

    @staticmethod
    def tune(pretrained_bert_name,
             preprocessing_function,
             bert_like_special_tokens,
             train_dataset_file,
             val_dataset_file,
             run_on_multi_gpus):
        lrs = [5e-5, 3e-5, 2e-5]
        dp = [0, 0.1, 0.2]
        grad_accumulation_steps = [2, 4]
        batch_size = 8
        params = list(product(lrs, dp, grad_accumulation_steps))

        tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)

        print("On Train")
        train_dataset = NERBERTDataset(
            train_dataset_file,
            tokenizer,
            bert_like_special_tokens,
            preprocessing_function
        )

        print("On Val")
        val_dataset = NERBERTDataset(
            val_dataset_file,
            tokenizer,
            bert_like_special_tokens,
            preprocessing_function
        )

        results = []
        for i, (lr, dp, grad_accumulation_steps) in enumerate(params):
            print(f'{i + 1}/{len(params)}')
            torch.manual_seed(0)
            current_system_wrapper = NERBERTSystemWrapper(
                pretrained_bert_name,
                preprocessing_function,
                bert_like_special_tokens,
                {'dp': dp}
            )   

            current_system_wrapper._train_impl(
                train_dataset,
                val_dataset,
                lr,
                batch_size,
                grad_accumulation_steps,
                run_on_multi_gpus,
                tokenizer.pad_token_id
            )

            current_results = current_system_wrapper._evaluate_impl(
                val_dataset,
                batch_size,
                run_on_multi_gpus,
                tokenizer.pad_token_id
            )
            results.append([current_results['macro-f1'].score, (lr, dp, grad_accumulation_steps)])

        return results
