import transformers

def init(model_name, task_name, adapter_config, labels=None, config_name = None, tokenizer_name=None, revision=None, adapter_non_linearity = None, adapter_reduction_factor = None, load_adapter=None, extra_adapters={}):
    config = transformers.AutoConfig.from_pretrained(config_name if config_name else model_name, num_labels=len(labels) if labels else None, revision=revision, finetuning_task=task_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name, revision=revision, use_fast=True)

    model = transformers.AutoAdapterModel.from_pretrained(model_name, config=config, revision=revision)

    if labels:
        model.add_classification_head(
            task_name or '',
            num_labels=len(labels)
            id2label={i: v for i, v in enumerate(labels)} if len(labels) > 0 else None,
        )

    if task_name not in model.config.adapters:
        adapter_config = transformers.AdapterConfig.load(adapter_config, non_linearity=adapter_non_linearity, reduction_factor = adapter_reduction_factor)
        if load_adapter:
            model.load_adapter(load_adapter, config=adapter_config, load_as=task_name)
        else:
            model.add_adapter(task_name, config=adapter_config)
        for extra_task, (extra_adapter, extra_config, extra_non_linearity, extra_reduction_factor) in [*extra_adapters.items()]:
            extra_config = transformers.AdapterConfig.load(extra_config, non_linearity=extra_non_linearity, reduction_factor=extra_reduction_factor)
            extra_adapter = model.load_adapter(extra_adapter, config=extra_config, load_as=extra_task)
            extra_adapters[extra_task] = extra_adapter

        # freezes non-adapter weights
        model.train_adapter([task_name])

        model.set_active_adapters(ac.Stack([*extra_adapters.keys(), task_name])
        return model
    else:
        if load_adapter or len(extra_adapters):
            raise ValueError('this code was copied from code that does not handle loading adapters when not training a task')
    

model = init()
optim = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9,0.999), eps=4e-9)

while True:
    loss = model(input_ids, labels)
    model.zero_grad()
    loss.backward()
    optim.step()

