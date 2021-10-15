
from tensorflow.keras.layers import Bidirectional, Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.models import Sequential



class BuildModel():
    '''
    Build Lstm model for tensorflow
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model
    
    '''
    def __init__(self, EMWEIGHTS = embedding_matrix, EMBEDDING_DIM= 50,MAX_SEQUENCE_LENGTH= 348, LOSS='categorical_crossentropy',OPTIMIZER='rmsprop',METRICS=['acc'],NUM_CLASSES=11,DROP_OUT_RATE =.4 ):
        self.weights = [EMWEIGHTS]
        self.input_length = MAX_SEQUENCE_LENGTH
        self.embeding_dim = EMBEDDING_DIM
        self.loss = LOSS
        self.optimizer = OPTIMIZER
        self.metrics = METRICS
        self.model = []
        self.num_classes = NUM_CLASSES
        self.drate = DROP_OUT_RATE
        
    def DefineModel(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        #Bidirectional LSTM
        self.model = Sequential()
        self.model.add(Embedding(len(word_index) + 1,
                                    self.embeding_dim,
                                    weights=self.weights,
                                    input_length=self.input_length ,
                                    trainable=True))
        self.model.add(Bidirectional(LSTM(100, dropout = self.drate, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(256, dropout = self.drate)))
        self.model.add(Dense(self.num_classes,activation='sigmoid'))
        # return self.final_set,self.labels, self.enc, self.ohe,self.encoding_flag
    def CompileModel(self):
        '''
        Compile the model
        ----------
        
        Returns
        -------
        
        '''
        self.model.compile(loss=self.loss,
              optimizer=self.optimizer,
              metrics=self.metrics)
#         return self.model
    def BuildModel(self):
        '''
        Build the model
        ----------
        
        Returns
        -------
        
        '''
        self.DefineModel()
        self.CompileModel()
        return self.model



