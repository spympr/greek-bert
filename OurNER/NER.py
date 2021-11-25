### Import useful libraries
import pandas as pd
import numpy as np
import torch,time,datetime,random,statistics,argparse
from torch import cuda,nn
from seqeval.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM,AutoModelForTokenClassification
from torch import cuda

def check_entities(curr_df):
    unique_labels = curr_df.Label.unique()
    print("Number of tags: {}".format(len(unique_labels)))
    print(curr_df.Label.value_counts(),"\n")

def convert_to_sentences(word_df,little_data):
    sentences,sentence_labels,t1,t2 = [], [], [], []

    for index, row in word_df.iterrows():
        
        if index == 1000 and little_data == True:
            break

        if row['Word'] == 'newline' and t1!=[] and t2!=[]:
            sentences.append(t1) 
            sentence_labels.append(t2)
            t1,t2 = [], []
        else:
            t1.append(row['Word'])
            t2.append(row['Label'])
            
    return pd.DataFrame(list(zip(sentences,sentence_labels)), columns=['Sentence', 'Labels'])  

def stats(temp_df):
    max = -1
    count = 0
    li = []
    for i,k in zip(temp_df['Sentence'],temp_df['Labels']):
        li.append(len(i))
        if max < len(i):
            max = len(i)
        count += len(i)
        if(len(i)==0):
            print(i,k)
    return max,count,li

def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        f.close()

    words,labels = [], [] 
    for line in lines:
        var = line.split()

        if len(var)==2:
            words.append(var[0])
            labels.append(var[1])
        elif len(var)==0:
            words.append('newline')
            labels.append('O')

    assert(len(words)==len(labels))
    return pd.DataFrame(list(zip(words,labels)), columns=['Word', 'Label'])  

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def tokenize_and_align_labels(examples,tokenizer,l2id,max_len,label_all_tokens_=False):
    tokenized_inputs = tokenizer(list(examples['Sentence']), 
                                truncation=True, 
                                is_split_into_words=True,
                                add_special_tokens=True,
                                max_length=max_len,
                                padding='max_length',
                                )
    labels = []
    for i, label in enumerate(examples['Labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(l2id[label[word_idx]])
            else:
                label_ids.append(l2id[label[word_idx]] if label_all_tokens_ else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    return {'ids':  torch.as_tensor(tokenized_inputs['input_ids'],dtype=torch.long),
            'mask': torch.as_tensor(tokenized_inputs['attention_mask'], dtype=torch.long),
            'tags': torch.as_tensor(labels, dtype=torch.long)
        } 

class CustomDataset(Dataset):
    def __init__(self, my_dict):
        self.len = len(my_dict["ids"])
        self.data = my_dict

    def __getitem__(self, index):
        return {'ids':  self.data["ids"][index],'mask': self.data["mask"][index],'tags': self.data["tags"][index]} 

    def __len__(self):
        return self.len

class NERBERTModel(nn.Module):

    def __init__(self, bert_model_, num_labels_, dp_prob_):
        super(NERBERTModel, self).__init__()
        self.num_labels = num_labels_
        self.bert_model = bert_model_
        self.dp = nn.Dropout(dp_prob_)
        self.ll = nn.Linear(768, num_labels_)

    def forward(self, ids, mask):
        return self.ll(self.dp(self.bert_model(input_ids=ids,attention_mask=mask)[0]))

def main():

    ### Choose Hyperparameters
    #############################################################################################################
    parser = argparse.ArgumentParser(description='NER')
    parser.add_argument("--which_model", type=int, required=True, help="0 or 1 or 2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-05)
    parser.add_argument("--dp_prob", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--seed_val", type=int, required=True)
    parser.add_argument("--MAX_GRAD_NORM", type=int, default=1)
    parser.add_argument('--little_data', default=False, action='store_true')
    parser.add_argument('--BertForToken', default=False, action='store_true')
    parser.add_argument('--label_all_tokens', default=False, action='store_true')
    args = parser.parse_args()
    #############################################################################################################
    MODELS = ['alexaapo/greek_legal_bert_v2','nlpaueb/bert-base-greek-uncased-v1','alexaapo/greek_legal_bert_v1']
    MODEL = MODELS[args.which_model]
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    dp_prob = args.dp_prob
    max_len = args.max_len
    BertForToken = args.BertForToken
    label_all_tokens = args.label_all_tokens
    MAX_GRAD_NORM = args.MAX_GRAD_NORM
    seed_val = args.seed_val
    little_data = args.little_data
    #############################################################################################################
    print("Model:",MODEL)
    print("Learning Rate:",learning_rate)
    print("Batch Size:",batch_size)
    print("Epochs:",epochs)
    print("Dropout:",dp_prob)
    print("Max Sequence Length:",max_len)
    print("BertForToken:",BertForToken)
    print("label_all_tokens:",label_all_tokens)
    print("MAX_GRAD_NORM:",MAX_GRAD_NORM)
    #############################################################################################################

    ### Load Cuda GPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)
    if device == 'cuda':  print("GPU: {}".format(torch.cuda.get_device_name(0)))

    ### Retrieve Data
    train_df = read_file('../OurNER/train.txt')
    dev_df   = read_file('../OurNER/dev.txt')
    test_df  = read_file('../OurNER/test.txt')

    ### Check Entities
    # print("For Train Data:\n")
    # check_entities(train_df)
    # print("For Dev Data:\n")
    # check_entities(dev_df)
    # print("For Test Data:\n")
    # check_entities(test_df)
    unique_labels = train_df.Label.unique()

    ### Create labels_to_ids
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}
    labels_to_ids

    ### From words to sentences
    train_df = convert_to_sentences(train_df,little_data)
    dev_df = convert_to_sentences(dev_df,little_data)
    test_df = convert_to_sentences(test_df,little_data)

    # print("Total sentences in train:      ", len(train_df))
    # print("Max words in train sentences:  ", stats(train_df)[0])
    # print("Total words in train sentences:", stats(train_df)[1])
    # print("Mean words in train sentences: ", "{0:.2f}".format(statistics.mean(stats(train_df)[2])),"\n")
    # print("Total sentences in dev:        ", len(dev_df))
    # print("Max words in dev sentences:    ", stats(dev_df)[0])
    # print("Total words in dev sentences:  ", stats(dev_df)[1])
    # print("Mean words in dev sentences:   ", "{0:.2f}".format(statistics.mean(stats(dev_df)[2]),"\n"))
    # print("Total sentences in test:       ", len(test_df))
    # print("Max words in test sentences:   ", stats(test_df)[0])
    # print("Total words in test sentences: ", stats(test_df)[1])
    # print("Mean words in test sentences:  ", "{0:.2f}".format(statistics.mean(stats(test_df)[2])))

    # print(sorted(stats(train_df)[2],reverse=True)[0:20])
    # print(sorted(stats(dev_df)[2],reverse=True)[0:20])
    # print(sorted(stats(test_df)[2],reverse=True)[0:20])

    # thresholds = [32,64,128,256]
    # for t in thresholds:
    #     print("For threshold", t)
    #     len_ = len(sorted(stats(train_df)[2]))
    #     sum_ = sum(i < t for i in sorted(stats(train_df)[2]))
    #     print("For train set:","{0:.3f}".format(sum_/len_))
    #     print("For dev   set:","{0:.3f}".format(sum_/len_))
    #     print("For test  set:","{0:.3f}".format(sum_/len_))
    #     print()

    # print(train_df.iloc[200:215])

    ### Load Tokenizer & Model
    if BertForToken:
        model = AutoModelForTokenClassification.from_pretrained(MODEL,num_labels=len(unique_labels)).to(device)
    else:
        model = NERBERTModel(AutoModel.from_pretrained(MODEL),len(unique_labels),dp_prob).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)
        
    train_tokenized_set = tokenize_and_align_labels(train_df,tokenizer,labels_to_ids,max_len,label_all_tokens)
    dev_tokenized_set = tokenize_and_align_labels(dev_df,tokenizer,labels_to_ids,max_len,label_all_tokens)
    test_tokenized_set = tokenize_and_align_labels(test_df,tokenizer,labels_to_ids,max_len,label_all_tokens)

    ### Cross Checking
    # print(type(train_tokenized_set),"\n")

    # print(len(train_tokenized_set["ids"]))
    # print(len(dev_tokenized_set["ids"]))
    # print(len(test_tokenized_set["ids"]),"\n")

    # print(len(train_tokenized_set["ids"][0]),"\n")
        
    # print(train_tokenized_set["ids"][0])
    # print(train_tokenized_set["mask"][0])
    # print(train_tokenized_set["tags"][0])

    ### Create DataLoaders
    train_set = CustomDataset(train_tokenized_set)
    dev_set = CustomDataset(dev_tokenized_set)
    test_set = CustomDataset(test_tokenized_set)

    train_dataloader = DataLoader(train_set,**{'batch_size': batch_size,'shuffle': True,'num_workers': 0})
    dev_dataloader   = DataLoader(dev_set,  **{'batch_size': batch_size,'shuffle': True,'num_workers': 0})
    test_dataloader  = DataLoader(test_set, **{'batch_size': batch_size,'shuffle': True,'num_workers': 0})

    print("Train Batches:",len(train_dataloader), "=",len(train_tokenized_set["ids"]), "/", batch_size)
    print("Dev   Batches:",len(dev_dataloader), " = ",len(dev_tokenized_set["ids"]), "/", batch_size)
    print("Test  Batches:",len(test_dataloader), " = ",len(test_tokenized_set["ids"]), "/", batch_size)

    # ### Inspect train example
    # if little_data: random_example = 0
    # else:   random_example = 212

    # for z,batch in enumerate(train_dataloader):
    #     if z == random_example:
    #         b_input_ids,b_input_mask,b_labels = batch['ids'],batch['mask'],batch['tags']
    #         print(b_input_ids[0])
    #         print(b_input_mask[0])
    #         print(b_labels[0])
    #         break

    # for token, label in zip(tokenizer.convert_ids_to_tokens(train_set[random_example]["ids"]), train_set[random_example]["tags"]):
    #     print('{0:10}  {1}'.format(token, label))

    ### Sanity Check before Training
    # input_ids,attention_mask,labels = train_set[random_example]["ids"].unsqueeze(0).to(device),train_set[random_example]["mask"].unsqueeze(0).to(device),train_set[random_example]["tags"].unsqueeze(0).to(device)
    # logits = model(input_ids,attention_mask)[0]
    # print(input_ids.shape,attention_mask.shape,labels.shape)
    # print(logits.shape,"= (batch_size, sequence_length, num_labels)")

    ### Check Performance
    # Load model and tokenizer
    # tokenizer_greek = AutoTokenizer.from_pretrained(MODEL)
    # lm_model_greek = AutoModelForMaskedLM.from_pretrained(MODEL)

    # def tt():
    #     print("================================================================================================================================================")

    # tt()
    # # # ================ EXAMPLE 1 ================
    # text_1 = 'Ο [MASK] προσανατολισμός της νέας φαρμακευτικής πολιτικής διατρέχει το σύνολο των επί μέρους διατάξεων του νόμου.'
    # # text_1 = 'Ο κοινωνικός προσανατολισμός της νέας φαρμακευτικής πολιτικής διατρέχει το σύνολο των επί μέρους διατάξεων του νόμου.'
    # input_ids = tokenizer_greek.encode(text_1)
    # print("\n",tokenizer_greek.convert_ids_to_tokens(input_ids))
    # outputs = lm_model_greek(torch.tensor([input_ids]))[0]
    # for _,i in enumerate(torch.topk(outputs[0,2],3)[1]):
    #     print("Model's Answer ",_,": ",tokenizer_greek.convert_ids_to_tokens(i.item()),sep='')
    # print("\nCorrect Answer:", "κοινωνικός")

    # tt()
    # # # ================ EXAMPLE 2 ================
    # text_2 = 'Η [MASK] ενός ταμείου που διευκολύνει την κίνηση του πετρελαίου σ’ όλη τη χώρα.'
    # # text_2 = 'H δημιουργία ενός ταμείου που διευκολύνει την κίνηση του πετρελαίου σ’ όλη τη χώρα.'
    # input_ids = tokenizer_greek.encode(text_2)
    # print("\n",tokenizer_greek.convert_ids_to_tokens(input_ids))
    # outputs = lm_model_greek(torch.tensor([input_ids]))[0]
    # for _,i in enumerate(torch.topk(outputs[0,2],3)[1]):
    #     print("Model's Answer ",_,": ",tokenizer_greek.convert_ids_to_tokens(i.item()),sep='')
    # print("\nCorrect Answer:", "δημιουργια")

    # tt()
    # # # ================ EXAMPLE 3 ================
    # text_3 = 'Οι κανόνες [MASK] των δεδομένων προσωπικού χαρακτήρα διέπουν σημαντικές πτυχές του τρόπου αλληλεπίδρασης των επιγραμμικών υπηρεσιών με τους χρήστες, ωστόσο, ισχύουν επίσης και άλλοι κανόνες.'
    # # text_3 = 'Οι κανόνες προστασίας των δεδομένων προσωπικού χαρακτήρα διέπουν σημαντικές πτυχές του τρόπου αλληλεπίδρασης των επιγραμμικών υπηρεσιών με τους χρήστες, ωστόσο, ισχύουν επίσης και άλλοι κανόνες.'
    # input_ids = tokenizer_greek.encode(text_3)
    # print("\n",tokenizer_greek.convert_ids_to_tokens(input_ids))
    # outputs = lm_model_greek(torch.tensor([input_ids]))[0]
    # for _,i in enumerate(torch.topk(outputs[0,3],3)[1]):
    #     print("Model's Answer ",_,": ",tokenizer_greek.convert_ids_to_tokens(i.item()),sep='')
    # print("\nCorrect Answer:", "προστασίας")
    # tt()
   
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ### Train & Evaluate Model
    # We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    best_val_loss = float(np.inf)

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 400 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            b_input_ids,b_input_mask,b_labels = batch['ids'].to(device),batch['mask'].to(device),batch['tags'].to(device)

            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            if BertForToken:
                loss = model(b_input_ids,attention_mask=b_input_mask,labels=b_labels)[0]
            else:
                logits = model(b_input_ids,b_input_mask)
                loss = loss_fn(logits.view(-1,model.num_labels), b_labels.view(-1)) 

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Accumulate the training loss over all of the batches 
            total_train_loss += loss.item()
            
            # Clip the norm of the gradients to 1.0 (prevent the "exploding gradients" problem)
            if MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss/len(train_dataloader)
        print("\n  Average training loss: {0:.2f}".format(avg_train_loss))

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for step, batch in enumerate(dev_dataloader):

            # Unpack this training batch from our dataloader. 
            b_input_ids,b_input_mask,b_labels = batch['ids'].to(device),batch['mask'].to(device),batch['tags'].to(device)

            # Tell pytorch not to bother with constructing the compute graph 
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                if BertForToken:
                    loss = model(b_input_ids,attention_mask=b_input_mask,labels=b_labels)[0]
                else:
                    logits = model(b_input_ids,b_input_mask)
                    loss = loss_fn(logits.view(-1,model.num_labels), b_labels.view(-1))  

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss/len(dev_dataloader)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation took: {:}".format(validation_time))

        # Save model in the point with best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Saving the model with val loss = ","{:.2f}".format(best_val_loss))
            torch.save(model.state_dict(),'../OurNER/'+str(seed_val)+'.pt')
            
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    ### Training Stats
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    ### Test Model
    t0 = time.time()

    # Load model from that point with the best validation loss
    model.load_state_dict(torch.load('../OurNER/'+str(seed_val)+'.pt'))
    model.eval()

    test_preds , test_labels = [], []

    for step, batch in enumerate(test_dataloader):
        # Unpack this training batch from our dataloader. 
        b_input_ids,b_input_mask,b_labels = batch['ids'].to(device),batch['mask'].to(device),batch['tags'].to(device)

        with torch.no_grad():
            if BertForToken:        
                logits = model(b_input_ids,attention_mask=b_input_mask,labels=b_labels)[1]
            else:
                logits = model(b_input_ids,b_input_mask)
                loss = loss_fn(logits.view(-1,model.num_labels), b_labels.view(-1)) 

        # Compute training accuracy
        flattened_targets = b_labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        active_accuracy = b_labels.view(-1) != -100 # shape (batch_size * seq_len)
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        test_labels.extend(labels)
        test_preds.extend(predictions)
        
    final_labels = [ids_to_labels[id.item()] for id in test_labels]
    final_predictions = [ids_to_labels[id.item()] for id in test_preds]

    print("  Test took: {:}".format(format_time(time.time() - t0)))

    ### Classification Report of Test Results
    report = classification_report([final_labels],[final_predictions])
    print(report)
    print(set(final_labels)-set(final_predictions))

    ### Create Report
    with open('../OurNER/report'+str(seed_val)+'.txt', 'w') as f:
        f.write("Model:"+MODEL+"\n")
        f.write("Learning Rate:"+str(learning_rate)+"\n")
        f.write("Batch Size:"+str(batch_size)+"\n")
        f.write("Epochs:"+str(epochs)+"\n")
        f.write("Dropout:"+str(dp_prob)+"\n")
        f.write("BertForToken:"+str(BertForToken)+"\n")
        f.write("label_all_tokens:"+str(label_all_tokens)+"\n")
        f.write("MAX_GRAD_NORM:"+str(MAX_GRAD_NORM)+"\n")
        f.write("Max Sequence Length:"+str(max_len)+"\n\n")
        f.write(report)
        f.write("\n")
        for stat in training_stats:
            f.write("Epoch "+str(stat['epoch'])+"\nTrain Loss: "+str("{:.2f}".format(stat['Training Loss']))+"\nVal Loss:   "+str("{:.2f}".format(stat['Valid. Loss']))+"\n\n")
        f.close()

if __name__ == "__main__":
  main()