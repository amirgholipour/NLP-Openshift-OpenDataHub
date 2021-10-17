import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class BuildFeatures():
    '''
    Turn raw data into features for modeling
    ----------

    Returns
    -------
    self.final_set:
        Features for modeling purpose
    self.labels:
        Output labels of the features
    enc: 
        Ordinal Encoder definition file
    ohe:
        One hot  Encoder definition file
    '''
    def __init__(self, TRAIN_DATA,TEST_DATA,TRAIN_LABELS,TEST_LABELS):
        self.data =  []
        self.final_set = []
        self.labels = []
        self.encoding_flag = False
        self.ohe = []
        self.enc = []
        self.inputs = TRAIN_DATA.values
        self.tokenizer = Tokenizer(num_words=20000)
        self.train_data = TRAIN_DATA.values
        self.test_data = TEST_DATA.values
        self.train_data_seq = []
        self.test_data_seq = []
        self.final_train_data = []
        self.final_test_data = [] 
        self.train_labels = TRAIN_LABELS
        self.test_labels = TEST_LABELS
#         self.final_set,self.labels = self.build_data()
    def DefineTokenizer(self):
        '''
        Define the To
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''

        self.tokenizer.fit_on_texts(self.inputs)#total_complaints

        
#         return self.tokenizer
    ## read the data from the source file
    def TokenizeInputData(self):
        '''
        Reading the csv file
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''

        self.DefineTokenizer()

        self.train_data_seq = self.tokenizer.texts_to_sequences(self.train_data)
        self.test_data_seq = self.tokenizer.texts_to_sequences(self.test_data)
        return self.train_data_seq, self.test_data_seq
    def GetInfo(self):
        '''
        GetRequired Info
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        self.DefineTokenizer()
        total_complaints = np.append(self.train_data,self.test_data)
        MAX_SEQUENCE_LENGTH = max([len(c.split()) for c in total_complaints])
        print('Maximum Sequence length is %s .' % len(MAX_SEQUENCE_LENGTH))
        word_index = self.tokenizer.word_index# dictionary containing words and their index
        print('Found %s unique tokens.' % len(word_index))

        
        return MAX_SEQUENCE_LENGTH,word_index
    
    ## address the missing information
    def PaddingInputSequences(self):
        '''
        Replace the missing value with the zero.
        ----------
        
        Returns
        -------
        Dataframe with replaced missing value.
        '''
        MAX_SEQUENCE_LENGTH,word_index = self.GetInfo()
        self.train_data_seq, self.test_data_seq = self.TokenizeInputData()
        
        
        self.final_train_data = pad_sequences(self.train_data_seq, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
        self.final_test_data = pad_sequences(self.test_data_seq, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
        return self.data

    ## Do the label encoder on output and remove the output column from the feature vector
    def ConvertInputLabelsToCat(self):
        '''
        convert input labels to categorical
        ----------
        
        Returns
        -------
        self.data:
            Separate features data 
        
        self.labels:
            Ground truth or label of feature data

        
        '''
        ## Mapping the output to a numeric range
        self.train_labels = to_categorical(np.asarray(self.train_labels))
        self.test_labels = to_categorical(np.asarray(self.test_labels))
        print('Shape of train data tensor:', self.final_train_data.shape)
        print('Shape of train label tensor:', self.Train_labels.shape)
        print('Shape of test label tensor:', self.test_labels.shape)

        return self.train_labels,self.test_labels

        
    ## function for doing one hot encoding    
    def onehot_encoding(self,feature_list):
        '''
        Apply one hot ecoding on the string data which there order is not important, such as Gender, PaymentMethod and etc.
        ----------
        
        Returns
        -------
        self.final_set:
            encoded data
        
        ohe:
            one hot transformer module
        '''
        

        self.ohe = ce.OneHotEncoder(cols=feature_list)
        # data_ohe = self.data
        self.ohe.fit(self.data)
        # joblib.dump(enc, 'onehotencoder.pkl')  
        self.final_set = self.ohe.transform(data_ohe)

#         final_set.head(5)
        return self.final_set,self.ohe

    ## Doing ordinal encoding for the features which the order of value in the features are important
    def GenerateEmbedingIndexCoefs(self,s3bucketname ="raw-data-saeed",embedding_glove_name ="glove.6B.50d.txt"):
        '''
        ## CNN w/ Pre-trained word embeddings(GloVe)
        We’ll use pre-trained embeddings such as Glove which provides word based vector representation trained on a large corpus.

        It is trained on a dataset of one billion tokens (words) with a vocabulary of 400 thousand words. The glove has embedding vector sizes, including 50, 100, 200 and 300 dimensions.
        ----------
        
        Returns
        -------
        labelled_set:
            encoded data
        
        ohe:
            ordinal transformer module
        '''

        # !wget http://nlp.stanford.edu/data/glove.6B.zip
        f = client.get_object(s3bucketname, embedding_glove_name)
        embeddings_index = {}
        # f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
        # f = open( 'glove.6B.50d.txt')
        for line in f:
            # print(line.decode("utf-8") )
            line = line.decode("utf-8")
            # break
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index
    def CreateEmbeddingMatrix(self,EMBEDDING_DIM=50,word_index):
        '''
        Now lets create the embedding matrix using the word indexer created from tokenizer.

        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
    def build_data(self):
        '''
        Preform feature engineering on the categorical features
        ----------
        
        Returns
        -------
        self.final_set,self.labels, self.enc, self.ohe,self.encoding_flag
        '''
        self.read_data()
        self.data = self.handel_missing_values()
        self.data , self.labels = self.map_output()
        object_data = data.select_dtypes(include=['object'])
        object_data.columns 
        if len(object_data.columns )>=1:
            self.final_set, self.enc, self.ohe = self.encoding()
            encoding_flag = True
        else:
            print('There is no need for encoding')
            self.encoding_flag = False

        self.final_set.head(5)
        # return self.final_set,self.labels, self.enc, self.ohe,self.encoding_flag
    def get_output(self):
        self.build_data()
        return self.final_set,self.labels, self.enc, self.ohe,self.encoding_flag