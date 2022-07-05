#sst2_dataset this is revised from composer's example nlp notebook, to try to have something that works

# note: for gpt-2, the appropriate acceleration is to replace attention with alibi, and perform sequence length warmup

# NOTE NOTE: tokenizers can pad to 'longest' instead of 'max_length', not implemented yet

# Define a Composer Model
import torchmetrics
import composer.models.base
import composer.metrics
class ComposerBERT(composer.models.base.ComposerModel):
    def __init__(self, model):
        super().__init__()
        self.module = model

        # Metrics
        self.train_loss = composer.metrics.CrossEntropy()
        self.val_loss = composer.metrics.CrossEntropy()
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, batch):
        output = self.module(**batch)
        return output

    def loss(self, outputs, batch):
        return outputs['loss']

    def validate(self, batch):
        labels = batch.pop('labels')
        output = self.forward(batch)
        output = output['logits']
        return (output, labels)

    def metrics(self, train: bool = False):
        return torchmetrics.collections.MetricCollection([self.train_loss, self.train_acc]) if train else torchmetrics.collections.MetricCollection([self.val_loss, self.val_acc])

class ComposerGPT(composer.models.base.ComposerModel):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.module = model
        self.tokenizer = tokenizer

        # Metrics
        self.train_loss = composer.metrics.CrossEntropy()
        self.val_loss = composer.metrics.CrossEntropy()
        
        self.train_acc = torchmetrics.Accuracy(ignore_index = self.train_loss.ignore_index)
        self.val_acc = torchmetrics.Accuracy(ignore_index = self.val_loss.ignore_index)

    def forward(self, batch):
        output = self.module(**batch)
        return output

    def loss(self, outputs, batch):
        return outputs['loss']

    def validate(self, batch):
        labels = batch.pop('labels')
        output = self.forward(batch)
        output = output['logits']

        # flatten multitoken outputs
        output = output.view(-1, output.shape[-1])
        labels = labels.view(-1)

        return (output, labels)

    def metrics(self, train: bool = False):
        return torchmetrics.collections.MetricCollection([self.train_loss, self.train_acc]) if train else torchmetrics.collections.MetricCollection([self.val_loss, self.val_acc])


import torch, transformers

torch.cuda.is_available = lambda: False

# Create a BERT sequence classification model using HuggingFace transformers
#model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # in BERT hparams
#model = transformers.AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2) # in BERT hparams
model = transformers.AutoModelForCausalLM.from_pretrained('facebook/opt-125m')

# Create BERT tokenizer
#tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased') # from transfomer_shared
tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/opt-125m')
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id

# Package as a composer model
composer_model = ComposerGPT(model, tokenizer)

import datasets
import multiprocessing
import transformers
import composer
import composer.core

class ComposerTrainer:
    one_time = True
    def __init__(self, model, tokenizer, optimizer, *lr_schedule, batch_size = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size

    def dataset_tokenize(self, dataset, input_col, label_col=None, remove_cols=[], **kwparams):
        kwparams = {'ignore_index': self.model.train_loss.ignore_index}
        kwparams = {'padding':'max_length', 'max_length':256, **kwparams}
        remove_cols = [*remove_cols, input_col]
        if label_col is None:
            tokenizer = self._tokenize_input
        elif label_col == input_col:
            remove_cols = [*remove_cols]
            tokenizer = self._tokenize_input_as_labels
        else:
            remove_cols = [*remove_cols, label_col]
            tokenizer = self._tokenize_input_concat_labels
            kwparams = {'label_col':label_col,**kwparams}
        return dataset.map(tokenizer, batched=True, num_proc=multiprocessing.cpu_count(), batch_size=1000, remove_columns=remove_cols, fn_kwargs={'tokenizer':self.tokenizer, 'input_col':input_col, **kwparams})

    # these are static methods because the datasets library performs caching based on hashes of them
    @staticmethod
    def _tokenize_input(sample, *, tokenizer, input_col, ignore_index='ignored', **kwparams):
        return tokenizer(text = sample[input_col], **kwparams)
    @staticmethod
    def _tokenize_input_as_labels(sample, *, tokenizer, input_col, ignore_index=-100, **kwparams):
        kwparams = {'return_tensors':'np', 'max_length':kwparams['max_length']+1,**kwparams}
        tokenized = tokenizer(text = sample[input_col], **kwparams)
        tokenized['label_ids'] = tokenized['input_ids'][...,1:].copy()
        tokenized['label_ids'][tokenized['attention_mask'][...,1:] == 0] = ignore_index
        tokenized['input_ids'] = tokenized['input_ids'][...,:-1]
        tokenized['attention_mask'] = tokenized['attention_mask'][...,:-1]
        return tokenized
    @staticmethod
    def _tokenize_input_concat_labels(sample, *, tokenizer, input_col, label_col, ignore_index=-100, **kwparams):
        # here we concatenate the input and label ids as if labels follow input, such that only the labels are trained on
        max_length = kwparams['max_length']
        kwparams = {**kwparams, 'padding':'do_not_pad', 'return_attention_mask': False}
        #kwparams = {'return_tensors':'np', **kwparams, 'padding':'longest'}
        input_ids = tokenizer(text = sample[input_col], **kwparams)['input_ids']
        kwparams = {**kwparams, 'add_special_tokens':False}
        label_ids = tokenizer(text = sample[label_col], **kwparams)['input_ids']
        if ComposerTrainer.one_time:
            ComposerTrainer.one_time = False
            print('input and label concatenation has a lot of room for speed optimization')
        input_and_label_ids = [*zip(input_ids, label_ids)]
        for input_ids, label_ids in input_and_label_ids:
            if len(input_ids) >= max_length:
        #if len(input_tok[0]) >= max_length:
                print('warning: item length leaves no room for generation within max length')
        tokenized_input_ids = [
            (input_ids + label_ids[:-1] + [tokenizer.pad_token_id] * max(max_length + 1 - len(input_ids) - len(label_ids), 0))[:max_length]
            for input_ids, label_ids in input_and_label_ids
        ]
        tokenized_attention_mask = [
            ([1] * (len(input_ids) + len(label_ids) - 1) + [0] * max(max_length + 1 - len(input_ids) - len(label_ids), 0))[:max_length]
            for input_ids, label_ids in input_and_label_ids
        ]
        tokenized_label_ids = [
            ([ignore_index] * (len(input_ids) - 1) + label_ids + [ignore_index] * max(max_length + 1 - len(input_ids) - len(label_ids), 0))[:max_length]
            for input_ids, label_ids in input_and_label_ids
        ]
        return {
            'input_ids': tokenized_input_ids,
            'attention_mask': tokenized_attention_mask,
            'label_ids': tokenized_label_ids
        }

    def process_data(self, dataset):
        data_collator = transformers.data.data_collator.default_data_collator
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=data_collator)
        dataspec = composer.core.DataSpec(dataloader=dataloader, split_batch=self._split_batch_dict)
        return dataspec

    def fit(self, train_data, eval_data, duration='1ep', num_batches=150, num_eval_batches=-1, seed=17, precision='fp32'):
        # Create Trainer Object
        trainer = composer.Trainer(
            model=self.model,
            train_dataloader=train_data,
            eval_dataloader=eval_data,
            max_duration=duration,
            optimizers=self.optimizer,
            schedulers=self.lr_schedule,
            device='gpu' if torch.cuda.is_available() else 'cpu',
            train_subset_num_batches=num_batches,
            eval_subset_num_batches=num_eval_batches,
            precision=precision,
            seed=seed
        )
        # Start training
        trainer.fit()
        return trainer

    def predict(self, eval_batch, logits=False):
        # Move batch to gpu
        eval_batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in eval_batch.items()}
        with torch.no_grad():
            predictions = self.model(eval_batch)['logits']
            if not logits:
                predictions = predictions.argmax(dim=-1)
        return predictions
        
    def _split_batch_dict(self, batch, n_microbatches: int):
        chunked = {k: v.chunk(n_microbatches) for k, v in batch.items()}
        num_chunks = len(list(chunked.values())[0])
        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]

import torch.optim
import torch.optim.lr_scheduler

optimizer = torch.optim.AdamW(
    params=composer_model.parameters(),
    lr=3e-5, betas=(0.9, 0.98),
    eps=1e-6, weight_decay=1e-6
)
linear_lr_decay = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0,
    end_factor=0, total_iters=55#150
)

my_trainer = ComposerTrainer(composer_model, tokenizer, optimizer, linear_lr_decay, batch_size=16)

# Tokenize SST-2
# Split dataset into train and validation sets
sst2_dataset = datasets.load_dataset('glue', 'sst2')
def int2str(sample, features, col):
    feature = features[col]
    names = feature.names
    return {
        col: [
            '' if val == -1 else names[val]
            for val in sample[col]
        ]
    }
sst2_dataset = sst2_dataset.map(int2str, batched=True, num_proc=multiprocessing.cpu_count(), batch_size=1000, fn_kwargs={'features':sst2_dataset['train'].features, 'col':'label'})
tokenized_sst2 = my_trainer.dataset_tokenize(sst2_dataset, 'sentence', 'label', remove_cols=['idx'])
train_dataspec, eval_dataspec = my_trainer.process_data(tokenized_sst2['train']), my_trainer.process_data(tokenized_sst2['validation'])

fit_trainer = my_trainer.fit(train_dataspec, eval_dataspec, num_batches=150)

eval_batch = next(iter(eval_dataspec.dataloader))
predictions = my_trainer.predict(eval_batch)

# Visualize only 5 samples
predictions = predictions[:6]

#label = ['negative', 'positive']
for i, prediction in enumerate(predictions):
    sentence = sst2_dataset['validation'][i]['sentence']
    correct_label = sst2_dataset['validation'][i]['label']
    # use attention mask to get label offset, last 1 location
    # NOTE: should change to left-padding, then prediction is at end
    #prediction_label = label[prediction]
    label_offset = eval_batch['attention_mask'][i].nonzero(as_tuple=True)[0][-1]
    prediction_label_id = prediction[label_offset]
    prediction_label = tokenizer.decode(prediction_label_id)
    print(f'Sample: {sentence}')
    print(f'Label: {correct_label}')
    print(f'Prediction: {prediction_label}')
    print()
