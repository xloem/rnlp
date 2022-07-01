#sst2_dataset this is revised from composer's example nlp notebook, to try to have something that works

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


import torch, transformers

# Create a BERT sequence classification model using HuggingFace transformers
#model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # in BERT hparams
model = transformers.AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2) # in BERT hparams

# Create BERT tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased') # from transfomer_shared

# Package as a composer moel
composer_model = ComposerBERT(model)

import datasets
import multiprocessing
import transformers
import composer
import composer.core

class ComposerTrainer:
    def __init__(self, model, tokenizer, optimizer, *lr_schedule, batch_size = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size

    def dataset_tokenize_col(self, dataset, col, **kwparams):
        tokenizer = self.tokenizer
        kwparams = {'padding':'max_length', **kwparams}
        def tokenize(sample):
            return tokenizer(
                text=sample[col],
                **kwparams
            )
        return dataset.map(tokenize, batched=True, num_proc=multiprocessing.cpu_count(), batch_size=1000, remove_columns=['idx', 'sentence'])

    def process_data(self, dataset):
        data_collator = transformers.data.data_collator.default_data_collator
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=data_collator)
        dataspec = composer.core.DataSpec(dataloader=dataloader, split_batch=self._split_batch_dict)
        return dataspec

    def fit(self, train_data, eval_data, duration='1ep', num_batches=150, seed=17, precision='fp32'):
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
            precision=precision,
            seed=seed
        )
        # Start training
        trainer.fit()

    def predict(self, eval_batch, logits=False):
        # Move batch to gpu
        eval_batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in eval_batch.items()}
        with torch.no_grad():
            predictions = self.model(eval_batch)['logits']
            if not logits:
                predictions = predictions.argmax(dim=1)
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
    end_factor=0, total_iters=150
)

my_trainer = ComposerTrainer(composer_model, tokenizer, optimizer, linear_lr_decay, batch_size=16)

# Tokenize SST-2
# Split dataset into train and validation sets
sst2_dataset = datasets.load_dataset('glue', 'sst2')
tokenized_sst2 = my_trainer.dataset_tokenize_col(sst2_dataset, 'sentence')
train_dataspec, eval_dataspec = my_trainer.process_data(tokenized_sst2['train']), my_trainer.process_data(tokenized_sst2['validation'])

my_trainer.fit(train_dataspec, eval_dataspec)

eval_batch = next(iter(eval_dataspec.dataloader))
predictions = my_trainer.predict(eval_batch)
#
## Move batch to gpu
#eval_batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in eval_batch.items()}
#with torch.no_grad():
#    predictions = my_trainer.model(eval_batch)['logits'].argmax(dim=1)

# Visualize only 5 samples
predictions = predictions[:6]

label = ['negative', 'positive']
for i, prediction in enumerate(predictions[:6]):
    sentence = sst2_dataset['validation'][i]['sentence']
    correct_label = label[sst2_dataset['validation'][i]['label']]
    prediction_label = label[prediction]
    print(f'Sample: {sentence}')
    print(f'Label: {correct_label}')
    print(f'Prediction: {prediction_label}')
    print()
