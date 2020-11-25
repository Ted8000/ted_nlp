# 预训练
from transformers import *
import torch

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
dataset_train = LineByLineTextDataset(tokenizer, '/home/lupinsu001/data/raw_data/vakan_1000', block_size=32)

print(len(dataset_train))

config = RobertaConfig(
    vocab_size=21128,
    max_position_embeddings=32,
    num_attention_heads=6,
    num_hidden_layers=4,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)
print('num_parameters:', model.num_parameters())

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=128,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    prediction_loss_only=True,
)

trainer.train()

# save model
trainer.save_model('data/models/pretrained_model')
torch.save(model, 'pre_train_model.pt')