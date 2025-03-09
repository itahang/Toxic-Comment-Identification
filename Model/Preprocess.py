from nltk.corpus import stopwords
import string
import spacy
import emoji
from tokenizers import Tokenizer
import re
import jax 
from jax import numpy as jnp
import pandas as pd
from typing import Callable
import nltk
nltk.download('stopwords')

stpwds = stopwords.words('english')

nlp = spacy.load("en_core_web_sm")


punc = string.punctuation
punc = punc.replace('#', '').replace('!', '').replace('?', '')

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}

time_zone_abbreviations = [
        "UTC", "GMT", "EST", "CST", "PST", "MST",
        "EDT", "CDT", "PDT", "MDT", "CET", "EET",
        "WET", "AEST", "ACST", "AWST", "HST",
        "AKST", "IST", "JST", "KST", "NZST"
    ]

patterns = [
    r'\\[nrtbfv\\]',         # \n, \t ..etc
    '<.*?>',                 # Html tags
    r'https?://\S+|www\.\S+',# Links
    r'\ufeff',               # BOM characters
    r'^[^a-zA-Z0-9]+$',      # Non-alphanumeric tokens
    r'ｗｗｗ．\S+',            # Full-width URLs
    r'[\uf700-\uf7ff]',      # Unicode private-use chars
    r'^[－—…]+$',            # Special punctuation-only tokens
    r'[︵︶]'                # CJK parentheses
]

    
class DSet:
    def __init__(self,df:pd.DataFrame,giveMeVector:Callable):
        """
        Constructor for DSet object

        Args:
            df: Dataframes that contains the text and its labels
            giveMeVector: Function that returns Embedding of the text

        Note:
            The Dataframe should have first column [0] be text and remaining [1:] as labels
        Returns:
            None
        """
        self.df =df
        self.giveMeVector=giveMeVector
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        idx = int(idx)
        text,label = self.df.iloc[idx,0],self.df.iloc[idx,1:]
        label = jnp.array(label,dtype=jnp.float32)
        text = self.giveMeVector(text).squeeze()
        return text,label
        

class Preprocessings:
    def __init__(self,dims:int,seqLen:int)->None:
        """
        Constructor for PreProcessing Class
        
        Args:
            dims: Number of dimension vector to use
            seqLen: No of sequence to use
        Returns:
            None
            
        Note:
        dims can only be 50 or 100 
        seqLen should be strictly greater then 10 i.e seqLen>10 not seqLen>=10
        """

        assert dims==50 or dims==100,"dims can only be 50 or 100 "
        assert seqLen>10 ,"Sequence length should be greater then 10"
        self.dims=dims 
        self.seqLen = seqLen
        self.tokenizer = Tokenizer.from_file("SavedModel/tokenizer.json")
        self.vocab = self.tokenizer.get_vocab()
        self.setEmbeddings()
        
    def setEmbeddings(self)->None:
            if (self.dims==50):
                self.embedding =jnp.load("Embeddings/Embedding 50d.npy")
            else:
                self.embedding =jnp.load("Embeddings/Embedding 100d.npy")
        
    def getEmbeddings(self)->jnp.array:
        """
        Returns the Embedding Matrix
        
        Args:
            None
            
        Returns:
            Embedding Matrix (jnp.Array)            
        """
        return self.embedding
    
    def getTokenizer(self):
        """
        Returns the Tokenizer
        
        Args:
            None
        
        Returns:
            Return custom Tokenizer build on top of Hugging Face tokenizer  
        """
        return self.tokenizer
    
    def getVocab(self):
        """
        Returns vocabulary from the tokenizer
        """
        return self.vocab
    
    def preprocess(self,text:str)->str:
        """
        Returns the preprocessed text
        
        Args:
            text:Input string (str)
            
        Returns:
            preprocessed string(str)
        """
        
        for regex in patterns:
            text = re.sub(regex, '', text)
        
        text = text.lower()
        text = ' '.join(word for word in text.split() if word.upper() not in time_zone_abbreviations)
        text = ' '.join(chat_words.get(word.upper(), word) for word in text.split())
        text = ' '.join(word for word in text.split() if word.lower() not in stpwds)
        text = text.translate(str.maketrans(punc, ' ' * len(punc)))
        text = emoji.demojize(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        text = nlp(text)
        temp=[]
        for token in text:
            temp.append(token.lemma_)
        return " ".join(temp)

    def createTokens(self,text)->jnp.array:
        """
        Creates tokens
        Args:
            text:Input string
        Returns:
            tokens:Tokens of the string(jnp.array)
        """
        text= self.preprocess(text)
        ret=[]
        for word in text.split(' '):
            if word in  self.vocab:
                ret.append(self.vocab[word])
            else:
                ret.append(self.vocab['<unk>'])
        return jnp.array(ret)

    def getVectors(self,tokens:jnp.array)->jnp.array:
        assert jnp.all(tokens >= 0), "Negative Tokens are not accepted"
        return self.embedding[tokens]

    def paddedTokens(self,text)->jnp.array:
        """
        Pad the tokens to be of the desired length
        the length is specified by `self.seqLen`initialized at Constructor 
        """
        length = self.seqLen
        txt= self.createTokens(text)
        txt = txt[:length]
        txt = jnp.array(txt).reshape(-1)
        if(len(txt)<length):
            diff = length-len(txt)
            txt = jax.lax.pad(txt,0,[(0, diff, 0)])
        return txt        

    def returnGiveMeVectorFunction(self):
        embedding_matrix = self.embedding  # Capture it inside closure

        def giveMeVectors(text):
            text = self.paddedTokens(text)
            return embedding_matrix[text]  # Use the captured matrix
        
        return giveMeVectors
