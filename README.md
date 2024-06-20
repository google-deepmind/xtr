# XTR: Rethinking the Role of Token Retrieval in Multi-Vector Retrieval

In this repository, we provide how you can run XTR (conteXtualized Token Retriever) for document retrieval. Please refer to our NeurIPS 2023 paper ([Lee et al., 2023](https://arxiv.org/abs/2304.01982)) for technical details.

## Usage

XTR is available through [Kaggle Models](https://www.kaggle.com/models/deepmind/xtr/). For instance, you can load XTR checkpoints as follows:

```python
## Model Usage
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text  # Registers the ops.

hub_url = "/kaggle/input/xtr/tensorflow2/base-en/2/" # if using Kaggle Notebooks, otherwise:
hub_url = "https://www.kaggle.com/models/deepmind/xtr/frameworks/tensorFlow2/variations/base-en/versions/2"
encoder = hub.KerasLayer(hub_url, signature="serving_default", signature_outputs_as_dict=True)

# Sample texts to encode.
sample_texts = tf.constant(["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])
sample_embeds = encoder(sample_texts)

# This returns token-level representations from XTR.
encodings = sample_embeds["encodings"].numpy()
mask = sample_embeds["mask"].numpy()
print(f"encodings: {encodings.shape}, mask: {mask.shape}")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb)

Please check out our Notebook above, which contains the full inference for running document retrieval with XTR.

XTR is also available in [Huggingface](https://huggingface.co/google/xtr-base-en) thanks to [Mujeen Sung](https://github.com/mjeensung).

## Citing this work

```bibtex
@article{lee2024rethinking,
  title={Rethinking the role of token retrieval in multi-vector retrieval},
  author={Lee, Jinhyuk and Dai, Zhuyun and Duddu, Sai Meher Karthik and Lei, Tao and Naim, Iftekhar and Chang, Ming-Wei and Zhao, Vincent},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
