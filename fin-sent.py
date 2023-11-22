import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

df = pd.read_csv("cnbc_headlines.csv", nrows=50)
df_array = np.array(df)
df_list = list(df_array[:,0]) 
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
inputs = tokenizer(df_list, padding = True, truncation = True, return_tensors='pt') #tokenize text to be sent to model
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
model.config.id2label
positive = predictions[:, 0].tolist()
negative = predictions[:, 1].tolist()
neutral = predictions[:, 2].tolist()
table = {'Headline':df_list, "Positive":positive, "Negative":negative, "Neutral":neutral}      
df2 = pd.DataFrame(table, columns = ["Headline", "Positive", "Negative", "Neutral"])
df2.to_csv("fin-sentiments.csv", index=False)