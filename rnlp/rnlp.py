# === this was the first draft, before i started working with the composer notebook.
#     i ran into an issue so it seemed most reliable to work off of something that already functioned.

# a dataset is roughly a vector of dicts where the dict properties can be batched
# datasets are commonly split into train, test, etc

def commonsense_qa(*keys):
    import datasets
    if not len(keys):
        keys = ['train', 'validation', 'test']
    dataset = datasets.load_dataset('commonsense_qa', split=keys)
    dataset = datasets.interleave_datasets(dataset)
    # question, question_concept, choices: { label, text }, answerKey
    return dataset

def cqa2str(dataset, key='str'):
    def combine(entry):
        return {'str':(
            entry['question'] + '\n\n' +
            '\n'.join([
                f'{label} {text}'
                for label, text in zip(entry['choices']['label'], entry['choices']['text'])
            ]) + '\n' + entry['answerKey']
        )}
    return dataset.map(combine, batched=False, remove_columns=['question', 'choices', 'answerKey'])

def tokenize(dataset, tokenizer, key):
    import multiprocessing
    keys_to_remove = dataset.features.keys()
    def tokenize(entry):
        return tokenizer(entry[key], padding='max_length')
    return dataset.map(tokenize, batched=True, num_proc=multiprocessing.cpu_count(), remove_columns=keys_to_remove)

######

#import composer
#class ComposerGPT2(composer.ComposerModel):
#    def __init__(self, model):
#        super().__init__()
### the ComposerModel is subclassed in the nlp example, and wraps forward, loss,validate, and metrics, providing train_loss, val_loss, train_acc, val_acc properties

def composer_tokenize(tokenizer, sample):
    return tokenizer(
        text = sample,
        padding = 'max_length',
        #max_length = 256,
        #truncation = True
    )

if __name__ == '__main__':
    import composer
    import transformers
    from composer.models import GPT2Model

    model = GPT2Model(module=transformers.AutoModelForCausalLM.from_pretrained("gpt2"),
                      config=transformers.GPT2Config.from_pretrained("gpt2"),
                      #tokenizer=transformers.AutoTokenizer.from_pretrained("gpt2")
    )
    from torch.utils.data import DataLoader

    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    train_dataset = tokenize(cqa2str(commonsense_qa('train', 'test')), tokenizer, 'str')
    eval_dataset = tokenize(cqa2str(commonsense_qa('validate')), tokenizer, 'str')

    train_dataloader = DataLoader(train_dataset, batch_size=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    def split_batch_dict(batch, n_microbatches: int):
        chunked = {k: v.chunk(n_microbatches) for k, v in batch.items()}
        num_chunks = len(list(chunked.values())[0])
        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]

    train_dataspec = composer.core.DataSpec(dataloader=train_dataloader, split_batch=split_batch_dict)
    eval_dataspec = composer.core.DataSpec(dataloader=eval_dataloader, split_batch=split_batch_dict)

    import torch.optim
    import torch.optim.lr_scheduler
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr=3e-5, betas=(0.9, 0.98),
        eps=1e-6, weight_decay=3e-6,
    )
    linear_lr_decay = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0,
        end_factor=0, total_iters=150
    )

    trainer = composer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration="2ep", # can be other kindds of units i think

        optimizers=optimizer,
        schedulers=[linear_lr_decay],
        device='gpu' if torch.cuda.is_available() else 'cpu',
        train_subset_num_batches=150,
        precision='fp32',
        seed=17,

        #algorithms=[
   #         BlurPool(replace_convs=True, replace_maxpools=True, blur_first=True),
   #         ChannelsLast(),
   #         CutMix(alpha=1.0),
   #         LabelSmoothing(smoothing=0.1),
        #]
    )
    trainer.fit()
