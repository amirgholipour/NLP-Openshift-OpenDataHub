import tensorflow as tf


    

class Train_Model():
    '''
    Build Lstm model for tensorflow
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model
    
    '''
    
    def __init__(self, MODEL, MLFLOW, TOKENIZER, ENC,TRAIN_DATA, TRAIN_LABELS,TEST_DATA, TEST_LABELS,HOST, EXPERIMENT_NAME, BATCH_SIZE=64,EPOCHS=10):
        self.model_checkpoint_callback = []
        self.enc = ENC
        self.tokenizer = TOKENIZER
        self.mlflow = MLFLOW
        self.model = MODEL
        self.train_data = TRAIN_DATA
        self.train_labels = TRAIN_LABELS
        self.test_data  = TEST_DATA
        self.test_labels = TEST_LABELS
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.host = HOST
        self.experiment_name = EXPERIMENT_NAME
        self.history = []
    def DefineCheckPoint(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        #Bidirectional LSTM
        checkpoint_filepath = 'model.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_acc',
            mode='max',
            save_best_only=True)
        
        
    
    def SavePKL(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        joblib.dump(self.enc, 'labelencoder.pkl')  
        joblib.dump(self.tokenizer, 'tokenizer.pkl')  

        
    
        
    def ModelTraining(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        
        self.mlflow = MLflow(self.mlflow, self.host,self.experiment_name).SetUp_Mlflow()
        with self.mlflow.start_run(tags= {
                "mlflow.source.git.commit" : get_git_revision_hash() ,
                "mlflow.user": get_git_user(),
                "mlflow.source.git.repoURL": get_git_remote(),
                "git_remote": get_git_remote(),
                "mlflow.source.git.branch": get_git_branch(),
                "mlflow.docker.image.name": os.getenv("JUPYTER_IMAGE", "LOCAL"),
                "mlflow.source.type": "NOTEBOOK",
        #         "mlflow.source.name": ipynbname.name()
            }) as run:
                self.history = self.model.fit(self.train_data, self.train_labels,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         validation_data=(self.test_data, self.test_labels),callbacks=[self.model_checkpoint_callback])
                record_details(self.mlflow)
        return self.model,self.history
        # return self.final_set,self.labels, self.enc, self.ohe,self.encoding_flag
    