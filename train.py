import torch
from bigram_lm import BigramLanguageModel
from lstm_lm import LSTMLanguageModel
from transformer import TransformerLanguageModel
from tqdm import tqdm
torch.manual_seed(1)


'''initialize global vars'''
block_size = 32 #number of characters in a sequence
batch_size = 16 #number of sequences in a mini-batch
embed_size = 64 #size of embedding vectors
num_heads = 8 #number of attention heads
num_layers = 8 #number of transformer layers
dropout = 0.1 #dropout probability
learning_rate = 1e-3 #learning rate for optimizer
epochs = 10000 #number of training iterations
test_step = 500 #number of training iterations between each validation
device = 'cuda' if torch.cuda.is_available() else 'cpu' #use GPU if available
model_name = 'transformer' #choose between 'bigram', 'lstm', and 'transformer'


class Trainer:
    def __init__(self):
        self.best_val_loss = 2e32


    def load_dataset(self, path):
        '''load, prepare, and split dataset into train and test sets'''
        with open(path, 'r') as f:
            text = f.read()

        #identify vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        #create mapping of characters to integers and vice versa
        stoi = dict((c, i) for i, c in enumerate(chars)) #string to int
        itos = dict((i, c) for i, c in enumerate(chars)) #int to string
        self.encode = lambda s: [stoi[c] for c in s] #encode string as list of ints
        self.decode = lambda e: ''.join([itos[i] for i in e]) #decode list of ints to string
        data = torch.tensor(self.encode(text), dtype=torch.long).to(device)
        self.train_data, self.test_data = data[:int(0.9*len(data))], data[int(0.9*len(data)):] #90% train, 10% test


    def get_batch(self, split):
        '''create a small batch of data from the text corpus'''
        data = self.train_data if split == 'train' else self.test_data
        ix = torch.randint(len(data) - block_size, (batch_size,)) #get random starting indices for sequences
        x = torch.stack([data[i:i+block_size] for i in ix]) #convert to 2D tensor
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #shifted by 1
        return x,y
    

    def initialize_model(self, model_name):
        if model_name == 'bigram':
            self.model = BigramLanguageModel(self.vocab_size).to(device)
        elif model_name == 'lstm':
            self.model = LSTMLanguageModel(self.vocab_size).to(device)
        elif model_name == 'transformer':
            self.model = TransformerLanguageModel(self.vocab_size, 
                                                  embed_size, 
                                                  block_size, 
                                                  num_heads, 
                                                  num_layers, 
                                                  dropout).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)


    def train(self):
        tqdm_bar = tqdm(range(epochs))
        for step in tqdm_bar:
            self.optimizer.zero_grad(set_to_none=True)
            x,y = self.get_batch('train')
            logits, loss = self.model(x,y)
            loss.backward()
            self.optimizer.step()
            if step % test_step == 0:
                self.val('train')
                val_loss = self.val('test')
                if val_loss < self.best_val_loss:
                    torch.save(self.model.state_dict(), f'{model_name}_model.pt')


    def val(self, split):
        '''evaluate model on validation set'''
        agg_loss = 0
        with torch.no_grad():
            for _ in range(10):
                x,y = self.get_batch(split)
                _, loss = self.model(x,y)
                agg_loss += loss.cpu().item()
        avg_loss = agg_loss / 10
        print(f'{split}_loss: {avg_loss :.2f}')
        return avg_loss


    def generate_text(self):
        '''generate text from the trained model'''
        tokens = self.model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)
        print(self.decode(tokens[0].tolist()))



if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_dataset('input.txt')
    trainer.initialize_model(model_name)
    trainer.train()
    trainer.generate_text()
