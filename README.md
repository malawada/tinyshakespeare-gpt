# tinyshakespeare-gpt
 Basic language models for learning and generating data from the tiny shakespeare dataset
## Files
main.ipynb - old notebook for initial training and testing with the dataset.

train.py - main training file. Using PyTorch and Python 3, run this file to train a model.

other *.py - model files

*.pt - saved model weights.

input.txt - tinyshakespeare dataset.

## Model Objective
Each model has the goal of predicting the next character given a block of previous characters from randomly selected passages of Shakespeare. Essentially, this means each model is a bigram language model. Once the model is trained, the generate() function can be called to produce new "Shakespeare-like" text by repeatedly predicting the next character and feeding prior predictions back into the model, building a string of text. 

## Data Preparation
I used the tiny shakespeare dataset, consisting of about 1 million characters of Shakespeare. 

A sample from the tiny shakespeare dataset:
```
Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?

Second Citizen:
Worthy Menenius Agrippa; one that hath always loved
the people.

First Citizen:
He's one honest enough: would all the rest were so!
```

This dataset was preprocessed by converting each character to an integer value. This gives us a vocabulary of about 65 unique characters in the dataset. The integer representations of the characters were used to encode the inputs and targets for the models, so that the goal of the model is to predict the next encoded character representation given a sequence of previous character representations. The models predictions can then be decoded back to characters for interpretation.

## Models
### Simple Bigram
Simple model that just uses a torch.nn.Embedding layer to learn character embeddings for each character based on their context and relative frequency with respect to previous characters. The model produces mostly gibberish after 10k training iterations:

```
TEEderyishon,
I ishur: tu:
Modyotouljothen.
NGUNo ndoule casseserere ilknachore?
omo hen bond an-knin.
KAR omod, ibredid, ys S:
LII whisthe nd aje n,
CELLAwe w im t heass dr selline, ple plea'seatheben of js y ORe
HEne:
BERThautisinen, ss C
PAn peresthex, pe ct ONCHeainNambe ten thist merd; frdweeavethaketoupurrongerequs tor othed m wnd ce 'st ane tiren harcor, odou wa 'l diquid se
My str nellap,
Prvir y ierowirind s doraveie-doul loraxis d m,
Whertt?

KIOre cr RCOfe,
Digid hu anes AR:
Do arith
```

### LSTM
The LSTM model implements an Embedding layer along with a 2-layer LSTM followed by a linear projection. By using an LSTM model, the intent is to better capture temporal features that can be leveraged to build the embeddings and predict the next character. The performance after 10k epochs is significantly better than the bigram:

```
JULIET:
Thou, I though no: but it, thy longer and man unto steal and like a sorroom. Then we had-kein.
Know; this bread thyself much his faults
Of those garmen impose as a turn'd sain to please,
And for my say,
Tell needs, but tise to dread
Prother speak, perceive that samble vow obst me down dree,
And, his our queenents to comford is at Parisf it tires have disperks!
O'e diskid second then of pass vir our how
Till spects and tongue rexise, my arrate?

KING EDWARD IV:
Did in thee strikp of reve
```

### Transformer 
The transformer model uses a decoder-only architecture and implements self-attention to better infer contextual features between successive characters and model the relationships between subsets of embedding features. It also includes positional encoding as an additional feature. This is the largest model and has the highest training overheads but its performance is comparable if not better than the LSTM:

```
To be whith frace; priced my old the hear,
My prince go impant's may hooding
Shorted that make the were it me sorrving
Will there, as the king,
And whoed, destrains is valby them hour well of the weap'd pelled
to the saught thesein to them kill he twarlingss,
Hare so blothedy did and to connator
Fill she have before; and beind out not honou does
in the snee, ! thou deadfed he wee no?

ProjTER:
But no, be have to be made now, but fair peace:
Uf conected you wondemen.

RENVOLIO:
I parry, dest men
```

## Limitations
The models  are implemented using character embeddings, while most larger NLP models use subword embeddings as they are more effective on larger corpuses. This would help extend the model beyond this specific application. Transfer learning from pretrained character embeddings from a larger dataset could also improve effectiveness, since the model would have a better initial understanding of English language and would only need to learn the intricacies of Shakespeare's unique writing style.
