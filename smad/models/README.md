# Using saved models

Once you've trained a model, you can load it back in using the model folder path and stored state dictionary:

```python

from smad.models.utils import load_model_package, create_model

data_path = 'path/to/model'

mdl_package = load_model_package(data_path)
model = create_model(mdl_package['training_info']['cfg'])
model.load_state_dict(mdl_package['state_dict'])

# or simply

mdl_package = load_model_package(data_path,load_model=True)
model = mdl_package['model']
```

Don't forget that the model will now be loaded on the CPU by default, so if you want to test the model or perform any function that require data be on the GPU, you'll need to run:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # this assumes you have cuda installed, otherwise it will just default to the CPU

model.to(device)
```