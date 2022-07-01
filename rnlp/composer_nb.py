# this is a typeover of composer's example nlp notebook, so as to have something that works


import transformers
from torchmetrics import Accuracy
from torchmetrics.collections import MetricCollection
from composer.models.base import ComposerModel
from composer.metrics import CrossEntropy

# Define a Composer Model
class ComposerBERT(ComposerModel):
    def __init__(self, model):
        super().__init__()
        self.module = model

        # Metrics
        self.train_loss = CrossEntropy()
        self.val_loss = CrossEntropy()
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

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
        return MetricCollection([self.train_loss, self.train_acc]) if train else MetricCollection([self.val_loss, self.val_acc])

# Create a BERT sequence classification model using HuggingFace transformers
#model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # in BERT hparams
model = transformers.AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-small', num_labels=2) # in BERT hparams

# Package as a composer moel
composer_model = ComposerBERT(model)

import datasets
from multiprocessing import cpu_count

# Create BERT tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased') # from transfomer_shared
def tokenize_function(sample):
    return tokenizer(
        text=sample['sentence'],
        padding='max_length',
        max_length=256,
        truncation=True
    )

# Tokenize SST-2
sst2_dataset = datasets.load_dataset('glue', 'sst2')
tokenized_sst2_dataset = sst2_dataset.map(tokenize_function, batched=True, num_proc=cpu_count(), batch_size=1000, remove_columns=['idx', 'sentence'])

# Split dataset into train and validation sets
train_dataset = tokenized_sst2_dataset['train']
eval_dataset = tokenized_sst2_dataset['validation']

from torch.utils.data import DataLoader
data_collator = transformers.data.data_collator.default_data_collator
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=data_collator)
#train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=data_collator)
#eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=data_collator)

from composer.core import DataSpec

def split_batch_dict(batch, n_microbatches: int):
    chunked = {k: v.chunk(n_microbatches) for k, v in batch.items()}
    num_chunks = len(list(chunked.values())[0])
    return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]

train_dataspec = DataSpec(dataloader=train_dataloader, split_batch=split_batch_dict)
eval_dataspec = DataSpec(dataloader=eval_dataloader, split_batch=split_batch_dict)

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

optimizer = AdamW(
    params=composer_model.parameters(),
    lr=3e-5, betas=(0.9, 0.98),
    eps=1e-6, weight_decay=1e-6
)
linear_lr_decay = LinearLR(
    optimizer, start_factor=1.0,
    end_factor=0, total_iters=150
)

import torch
from composer import Trainer

# Create Trainer Object
trainer = Trainer(
    model=composer_model,
    train_dataloader=train_dataspec,
    eval_dataloader=eval_dataspec,
    max_duration="1ep",
    optimizers=optimizer,
    schedulers=[linear_lr_decay],
    device='gpu' if torch.cuda.is_available() else 'cpu',
    train_subset_num_batches=150,
    precision='fp32',
    seed=17
)
# Start training
trainer.fit()

eval_batch = next(iter(eval_dataloader))

# Move batch to gpu
eval_batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in eval_batch.items()}
with torch.no_grad():
    predictions = composer_model(eval_batch)['logits'].argmax(dim=1)

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
