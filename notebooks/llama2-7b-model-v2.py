#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datetime import datetime
import json
import torch as t
import pandas as pd


# In[ ]:


# Suppress warning messages
from transformers.utils import logging
logging.set_verbosity(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[ ]:


# Program variables
#max_iterations = 10
conversation_history = list()
model_id = "codellama/CodeLlama-7b-Instruct-hf"
filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"


# In[ ]:


device = "cuda:0" if t.cuda.is_available() else "cpu"


# In[ ]:


device


# In[ ]:


cache_dir ="../datax/models"


# In[ ]:


# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir, device_map=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, padding_side="left")
streamer = TextStreamer(tokenizer, skip_prompt=True)


# ## Prompt template configuration

# In[ ]:


prompt_templates = {}
prompt_templates['p1']={'role': None,
    'example':"The following snippet is a confirmed vulnerable code with a severity score of {}:  {}",
          'question':"What is the vulnerability severity score for the following snippet?   {}",
        "field":"func_before"}
          
prompt_templates['p2']={'role':'You are a software code vulnerability inspector, CVSS score is a number between 0.0 and 10.0',
        'example':"The following snippet is a vulnerable code with a CVSS score of {}:  {}",
        'question':"What is the CVSS score for the following snippet?  {}",
        "field":"func_before"}
        
prompt_templates['p3']={'role':'You are a software code vulnerability inspector and you should to provide a CVSS score depending on the severity. a CVSS score is a number between 0.0 and 10.0',
        'example':"The following vulnerability description has a escore of {}:  {}",
        'question':"What is the CVSS score from the following description?  {}",
        "field":"func_before"}

  


# ### Examples

# In[ ]:


prompt1 = '''The following snippet is a exploitable code with a score of 4.5
	dev = usb_get_intfdata(interface);
	if (!dev) {
		retval = -ENODEV;
		goto exit;
	}

	/* increment our usage count for the device */
	kref_get(&dev->kref);

	/* save our object in the file's private structure */
	mutex_lock(&dev->io_mutex);
	file->private_data = dev;
	mutex_unlock(&dev->io_mutex);'''


# In[ ]:


prompt2 = '''The following snippet has a score of 7.5: int i;
char inLine[64];
cin >> inLine;
i = atoi (inLine);
sleep(i);'''


# In[ ]:


prompt3 = '''What is the score for the following snippet? 
int main(int argc, char *argv[])
{
	rc = SQLConnect(Example.ConHandle, argv[0], SQL_NTS,
	(SQLCHAR *) "", SQL_NTS, (SQLCHAR *) "", SQL_NTS);
} '''


# In[ ]:


model.to(device)


# In[ ]:


N_EXAMPLES= 3


# In[ ]:


max_trials = 30


# # Data loading and filtering

# In[ ]:


val_data = pd.read_csv("../datax/big-vul/train.csv")


# In[ ]:


val_data.head()


# ## Filtering data by length

# 100 to 300 code length

# In[ ]:


filtered_val_300 = val_data[val_data['func_before'].str.len().between(100,300)]
filtered_val_300 = filtered_val_300[filtered_val_300['Score'].notna()]
filtered_val_300.shape


# less than 100 words

# In[ ]:


filtered_val_100 = val_data[val_data['func_before'].str.len()<100]
filtered_val_100 = filtered_val_100[filtered_val_100['Score'].notna()]
filtered_val_100.shape


# ### Load testing data

# In[ ]:


test_data = pd.read_csv("../datax/big-vul/test.csv")


# In[ ]:


filtered_test_300 = test_data[test_data['func_before'].str.len().between(100,300)]
filtered_test_300 = filtered_test_300[filtered_test_300['Score'].notna()]
filtered_test_300.shape


# In[ ]:


filtered_test_100 = test_data[test_data['func_before'].str.len()< 100]
filtered_test_100 = filtered_test_100[filtered_test_100['Score'].notna()]
filtered_test_100.shape


# In[ ]:


test_data.head()


# In[ ]:


def build_messages(filtered_val,  filtered_test, prompt_template):
    messages = []
    indexes = []
    gt = None
    if (role_template := prompt_template['role']) :
        messages.append(role_template)
    p1_template = prompt_template['example']
    p2_template = prompt_template['question']
    for i in range(N_EXAMPLES):
        random_row = filtered_val.sample(n=1)
        text = random_row['func_before'].values[0]
        score = random_row['Score'].values[0]
        message = p1_template.format(score, text)
        indexes.append(int(random_row['index'].values[0]))
        messages.append(message)
    random_row = filtered_test.sample(n=1)
    text = random_row['func_before'].values[0]
    gt = random_row['Score'].values[0]
    message = p2_template.format(text)
    indexes.append(int(random_row['index'].values[0]))
    messages.append(message)
    return messages, indexes, gt


# In[ ]:


messages, indexes, gt_score = build_messages(filtered_val_100, filtered_test_100, prompt_templates['p1'])


# In[ ]:


indexes


# In[ ]:


# Limit maximum iterations for conversation
def generate_prediction(messages):
    conversation_history = list()

    for message in messages:

        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "system", "content": ""})
        # Convert conversational history into chat template and tokenize
        inputs = tokenizer.apply_chat_template(conversation_history, return_tensors="pt", return_attention_mask=False).to(device)

        # Generate output
        generated_ids = model.generate(inputs,
            #streamer=streamer,
            max_new_tokens=20,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature= 0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        # Get complete output from model including input prompt
        output = tokenizer.batch_decode(generated_ids)[0]

        # Filter only new output information using '</s>' delimiter, then strip starting and trailing whitespace
        output_filtered = output.split('[/INST]')[-1].strip()

        # Update conversation history with the latest output
        conversation_history[-1]["content"] = output_filtered

    return conversation_history

        # Capture input before start of next iteration
        #capture_input()


# In[ ]:


def generate_trials():
    conversations = list()
    for i in range(max_trials):
        conversation = generate_prediction(messages)
        conversations.append(conversation)
    return conversations
        
        
    


# In[ ]:


conversations= generate_trials()


# In[ ]:


def save_conversations(conversation_history):
    # Save entire conversation history to text file for debugging or use for loading conversational context
    with open(filename, 'w') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)


# In[ ]:


N_EXAMPLES= 0


# In[ ]:


SAMPLES = 300


# In[ ]:


import logging
import time
logging.basicConfig(filename='my_script.log', level=logging.INFO, format='%(asctime)s %(message)s')


# In[ ]:


results = []
for i in range(SAMPLES):
    print("Executing...")
    logging.info(f"Logging message {i}")
    result = dict()
    messages, indexes, gt_score = build_messages(filtered_val_100, filtered_test_100, prompt_templates['p2'])
    if not gt_score:
        continue #TODO: DRC filter data with gt_score only
    result["indexes"] = indexes
    result["gt_score"] = gt_score
    result["chats"] = generate_trials()
    results.append(result)


# In[ ]:


import json
json_data = json.dumps(results, ensure_ascii=False)


# In[ ]:


def save_conversations(conversation_history):
    # Save entire conversation history to text file for debugging or use for loading conversational context
    with open(filename, 'w') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)


# In[ ]:


filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_100_30_0_codeLlama7b_P2.txt"
save_conversations(results)

