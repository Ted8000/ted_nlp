from transformers import *
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# data load
msgs = []
labels = []
with open('source_BIO_2014_cropus.txt', 'r') as f:
    for i in f:
        msgs.append(list(i.split()))
with open('target_BIO_2014_cropus.txt', 'r') as f:
    for i in f:
        labels.append(list(i.split()))


# 数据集划分
train_texts, val_texts, train_tags, val_tags = train_test_split(msgs, labels, test_size=.2)

unique_tags = set(tag for doc in labels for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

train_texts = list(' '.join(x) for x in train_texts)
val_texts = list(' '.join(x) for x in val_texts)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, train_x, train_y, max_len):
        self.tokenizer = tokenizer
        self.train_x = train_x
        self.train_y = train_y
        self.max_len = max_len

    def __len__(self):
        return len(self.train_x)
  
    def __getitem__(self, index):
        batch = self.tokenizer(self.train_x[index], padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        # [CLS], [SEP] 赋0
        batch['labels'] = torch.cat((torch.cat((torch.tensor([-100]),torch.tensor(self.train_y[index]),torch.tensor([-100]*self.max_len)))[:self.max_len-1], torch.tensor([-100])))
        
        return batch
    
class BERTClass(torch.nn.Module):
    def __init__(self, pretrain_name, num_class, cache_dir=None):
        super(BERTClass, self).__init__()
        if cache_dir == None:
            self.l1 = BertModel.from_pretrained(pretrain_name)
        else:
            self.l1 = BertModel.from_pretrained(pretrain_name, cache_dir=cache_dir)
        hidden = self.l1.pooler.dense.out_features
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(hidden, num_class)
    
    def forward(self, input_ids, attention_mask):
        sequence_output, output_1= self.l1(input_ids, attention_mask = attention_mask)
        output_2 = self.l2(sequence_output)
        output = self.l3(output_2)
        return output

def tag2label(data):
    tmp2 = []
    try:
        for i in data:
            tmp = []
            for j in i:
                tmp.append(tag2id[j])
            tmp2.append(tmp)
    except:
        print(i)
    return tmp2
train_tags_label = tag2label(train_tags)
val_tags_label = tag2label(val_tags)


# model load
num_class = len(tag2id)

tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
model = BERTClass('voidful/albert_chinese_tiny', num_class)


# 超参数
max_len=128
batch_sieze = 128
epochs = 3
batch_size = 64
lr = 5e-5

train_dataset = CustomDataset(tokenizer, train_texts, train_tags_label, max_len)
valid_dataset = CustomDataset(tokenizer, val_texts, val_tags_label, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

loss_fct = torch.nn.CrossEntropyLoss()
optim = AdamW(model.parameters(), lr=lr)

step_ = len(train_loader)//3
total_step = len(train_loader)


# train
for epoch in range(epochs):
    for i,batch in enumerate(tqdm(valid_loader)):
        optim.zero_grad()
        
        input_ids = batch['input_ids'].to(device).squeeze(1)
        attention_mask = batch['attention_mask'].to(device).squeeze(1)
        labels = batch['labels'].to(device)
        logits = model(input_ids, attention_mask=attention_mask)
        
        loss = None
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, num_class)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels.long())
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        loss.backward()
        optim.step()
        
        if (i+1) % step_ == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, epochs, i+1, total_step, loss.item()))

        
# predict
model.eval()

pred = []
real = []

for batch in tqdm(valid_loader):
    input_ids = batch['input_ids'].to(device).squeeze(1)
    attention_mask = batch['attention_mask'].to(device).squeeze(1)
    labels = batch['labels'].to(device)
    logits = model(input_ids, attention_mask=attention_mask)

    for i in range(len(logits)):
        pred.append(torch.max(logits[i][attention_mask[i]==1][1:-1], 1)[1].tolist())
        real.append(labels[i][attention_mask[i]==1][1:-1].tolist())    
        
a = [id2tag[i] for item in real for i in item]
b = [id2tag[i] for item in pred for i in item]
report = classification_report(a, b)
print(report)


#model save
