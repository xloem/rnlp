import torch
from transformers import BertTokenizer
from transformers.adapters import BertAdapterModel

# Load pre-trained BERT tokenizer from Huggingface.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# An input sentence.
sentence = "It's also, clearly, great fun."

# Tokenize the input sentence and create a PyTorch input tensor.
input_data = tokenizer(sentence, return_tensors='pt')

# Load pre-trained BERT model from HuggingFace Hub.
# The `BertAdapterModel` class is specifically designed for working with adapters.
# It can be used with different prediction heads.
model = BertAdapterModel.from_pretrained('bert-base-uncased')

# load pre-trained task adapter from Adapter Hub
# this method call will also load a pre-trained classification head for the adapter task
adapter_name = model.load_adapter('sst-2@ukp', config='pfeiffer')

# activate the adapter we just loaded, so that it is used in every forward pass
model.set_active_adapters(adapter_name)

# predict output tensor
outputs = model(**input_data)

# retrieve the predicted class label
predicted = torch.argmax(outputs[0]).item()
assert predicted == 1
print(sentence,'->',['negative','positive'][predicted]) # added this line

# save model
model.save_pretrained('./path/to/model/directory/')
# save adapter
model.save_adapter('./path/to/adapter/directory/', 'sst-2')

# load model
model = AutoAdapterModel.from_pretrained('./path/to/model/directory/')
model.load_adapter('./path/to/adapter/directory/')

# deactivate all adapters
model.set_active_adapters(None)
# delete the added adapter
model.delete_adapter('sst-2')
