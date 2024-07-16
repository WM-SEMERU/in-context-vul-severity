#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datetime import datetime
import json
import torch as t
import pandas as pd


# ## The transofrmer model initialization

# On this notebook we aim to evaluate Mistral model for predicting vulnerability CVSS by providing a set of examples on the in-context learning coniguration. We will introduce three kind of variables in the prompt:
# 1. Code with a context length less than 100 words.
# 2. Code with a context lenth between 100 and 300 words.
# 3. Only CVSS text description.
# 
# Since the model supports a limited number of tokens/words we truncate some descriptions

# In[4]:


# Suppress warning messages
from transformers.utils import logging
logging.set_verbosity(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[5]:


# Program variables
#max_iterations = 30
conversation_history = list()
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"


# In[6]:


device = "cuda:0" if t.cuda.is_available() else "cpu"


# In[7]:


device


# In[8]:


cache_dir ="../datax/models"


# In[9]:


# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, cache_dir=cache_dir, device_map=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, padding_side="left")
streamer = TextStreamer(tokenizer, skip_prompt=True)


# In[10]:


model.to(device)


# ## Step 1: Prompt template configuration

# This experiment considers three type of prompts:
# 1. Only code with out any context abour the score, we just want the model to classify and guess the score just providing some examples with the real ground truth score
# 2. Only code with some CVSS context, explaining first that CVSS consist on a score from 1.0 to 10.0 and we need to obtain that range of values depending on the severity of the vulnerability.
# 3. CVSS description and CVSS score.

# In[80]:


prompt_templates = {}
prompt_templates['p1']={'role': None,
    'example':"The following snippet is a exploitable code with a score of {}:  {}",
          'question':"What is the score for the following snippet?   {}"}
          
prompt_templates['p2']={'role':'You are a software code vulnerability inspector, CVSS score is a number between 0.0 and 10.0',
        'example':"The following snippet is a vulnerable code with a CVSS score of {}:  {}",
        'question':"What is the CVSS score for the following snippet?  {}"}
        
prompt_templates['p3']={'role':'You are a software code vulnerability inspector and you should to provide a CVSS score depending on the severity. a CVSS score is a number between 0.0 and 10.0',
        'example':"The following vulnerability description has a escore of {}:  {}",
        'question':"What is the CVSS score from the following description?  {}"}



# ### Prompt examples

# The following are just prompt examples with vulnerable code blocks and linked scores

# In[8]:


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


# In[9]:


prompt2 = '''The following snippet has a score of 7.5: int i;
char inLine[64];
cin >> inLine;
i = atoi (inLine);
sleep(i);'''


# In[10]:


prompt3 = '''What is the score for the following snippet? 
int main(int argc, char *argv[])
{
	rc = SQLConnect(Example.ConHandle, argv[0], SQL_NTS,
	(SQLCHAR *) "", SQL_NTS, (SQLCHAR *) "", SQL_NTS);
} '''


# ## Step 2: Experiment configuration

# In[138]:


N_EXAMPLES = 0


# In[13]:


max_trials = 30


# ## Step 3: Load testbed

# On this experiment we are using *Big-vul* datasets. For providing the examples we are reusing training split from that dataset.
# For building the question prompt we use the testing split dataset.

# In[14]:


val_data = pd.read_csv("../data/big-vul/train.csv")


# In[15]:


val_data.head()


# In[16]:


val_data.shape


# In[17]:


val_data['Summary'].notna().sum()


# ### Data filtering by size

# Filtering functions between 100 and 300 length at the function 

# In[106]:


filtered_val_300 = val_data[val_data['func_before'].str.len().between(100,300)]
filtered_val_300 = filtered_val_300[filtered_val_300['Score'].notna()]
filtered_val_300.shape


# Filtering functions less 100

# In[107]:


filtered_val_100 = val_data[val_data['func_before'].str.len()<100]
filtered_val_100 = filtered_val_100[filtered_val_100['Score'].notna()]
filtered_val_100.shape


# ### Load test split from Big-vul

# In[19]:


test_data = pd.read_csv("../data/big-vul/test.csv")


# In[20]:


test_data.shape


# In[21]:


test_data['Summary'].notna().sum()


# In[22]:


filtered_test_300 = test_data[test_data['func_before'].str.len().between(100,300)]
filtered_test_300 = filtered_test_300[filtered_test_300['Score'].notna()]
filtered_test_300.shape


# In[23]:


filtered_test_100 = test_data[test_data['func_before'].str.len()< 100]
filtered_test_100 = filtered_test_100[filtered_test_100['Score'].notna()]
filtered_test_100.shape


# In[24]:


test_data.head()


# In[25]:


test_data.columns.tolist()


# In[26]:


random_row = test_data.sample(n=1)
text = random_row['func_before'].values[0]
random_row['Score'].values[0]


# In[27]:


text


# In[30]:


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


# ### Build messages sandbox

# On this example we are taking random examples from training dataset with code length less than 100 and random question from test dataset with code length less than 100, we also use the prompt template configuration 1

# In[81]:


messages, indexes, gt_score = build_messages(filtered_val_100, filtered_test_100, prompt_templates['p2'])


# **Indexes:** Each datapoint from the testbed has an index, we capture the data point index from the dataset to have trazability. The indexes array reports the indexes from the training dataset (examples) and the index from the test dataset (question)

# In[76]:


indexes


# In[82]:


messages


# In[1]:


# Limit maximum iterations for conversation
def generate_prediction(messages):
    conversation_history = list()

    for message in messages:

        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": ""})
        # Convert conversational history into chat template and tokenize
        inputs = tokenizer.apply_chat_template(conversation_history, return_tensors="pt", return_attention_mask=False).to(device)

        # Generate output
        generated_ids = model.generate(inputs,
            #streamer=streamer,
            max_new_tokens=30,
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


# In[92]:


conversation = generate_prediction(messages)


# In[93]:


conversation


# In[94]:


def generate_trials():
    conversations = list()
    for i in range(max_trials):
        conversation = generate_prediction(messages)
        conversations.append(conversation)
    return conversations
        
    


# ### Conversation example

# In[95]:


conversations= generate_trials()


# In[96]:


conversations


# In[67]:


def save_conversations(conversation_history):
    # Save entire conversation history to text file for debugging or use for loading conversational context
    with open(filename, 'w') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)


# In[124]:


SAMPLES = 300


# ## Step 4: Parameter validation and experiment execution

# In[139]:


N_EXAMPLES


# In[140]:


max_trials


# In[141]:


SAMPLES


# In[146]:
import logging
import time
logging.basicConfig(filename='my_script.log', level=logging.INFO, format='%(asctime)s %(message)s')


results = []
for i in range(SAMPLES):
    logging.info(f"Logging message {i}")
    result = dict()
    messages, indexes, gt_score = build_messages(filtered_val_100, filtered_test_100, prompt_templates['p2'])
    if not gt_score:
        continue #TODO: DRC filter data with gt_score only
    result["indexes"] = indexes
    result["gt_score"] = gt_score
    result["chats"] = generate_trials()
    results.append(result)


# 

# In[147]:


results


# In[148]:


import json
json_data = json.dumps(results, ensure_ascii=False)


# In[149]:


filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_100_30_0_Mistral7b_P2.txt"
save_conversations(results)


# 

# In[ ]:


# Load conversational history from a previous context file
context_filename = "./*.txt"
with open(context_filename, 'r') as f:
     data = json.load(f)
     conversation_history = data



