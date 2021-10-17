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
    def ConvertInputLabelsToCat (self):
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

        return self.data , self.labels

        
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
    def ordinal_encoding(self,feature_list):
        '''
        Apply ordinal ecoding on the string data which there order is  important, such as Dependents, StreamingTV and etc.
        ----------
        
        Returns
        -------
        labelled_set:
            encoded data
        
        ohe:
            ordinal transformer module
        '''

        # for column in names:
        #     labelencoder(column)
        # data_enc = self.data
        self.enc = ce.ordinal.OrdinalEncoder(cols=feature_list)
        
        self.enc.fit(self.data)
        self.final_set = self.enc.transform(data_enc)
        # joblib.dump(enc, 'ordinalencoder.pkl')  
        return self.final_set,self.enc
    def encoding(self):
        '''
        Preform feature engineering on the categorical features
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        ordinal_feature_list = [ 'Partner', 'Dependents', 'PhoneService', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
        
        self.final_set, self.enc = self.ordinal_encoding(ordinal_feature_list)
        one_hot_feature_list = ['gender','MultipleLines', 'InternetService', 'Contract', 'PaymentMethod', 'OnlineSecurity', 'OnlineBackup',
         'DeviceProtection', 'TechSupport']
        self.final_set, self.ohe = self.onehot_encoding(one_hot_feature_list)
        return self.final_set, self.enc, self.ohe
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