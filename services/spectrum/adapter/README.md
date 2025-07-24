---
library_name: peft
license: other
base_model: /data/group/project1/models/Qwen3-32B
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: qm9s_300k
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qm9s_300k

This model is a fine-tuned version of [/data/group/project1/models/Qwen3-32B](https://huggingface.co//data/group/project1/models/Qwen3-32B) on the QM9S_train dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 6
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 48
- total_eval_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.15.1
- Transformers 4.51.3
- Pytorch 2.7.0+cu126
- Datasets 3.5.0
- Tokenizers 0.21.1