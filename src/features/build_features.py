import pandas as pd
import numpy as np
import category_encoders as ce

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
    def __init__(self, data_path):
        self.data_path = data_path
        self.data =  []
        self.final_set = []
        self.labels = []
        self.encoding_flag = False
        self.ohe = []
        self.enc = []
#         self.final_set,self.labels = self.build_data()
    ## read the data from the source file
    def read_data(self):
        '''
        Reading the csv file
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        
        self.data = pd.read_csv(self.data_path)
        return self.data
    ## address the missing information
    def handel_missing_values(self):
        '''
        Replace the missing value with the zero.
        ----------
        
        Returns
        -------
        Dataframe with replaced missing value.
        '''
        self.data.replace(0, np.nan, inplace=True)
        return self.data

    ## Do the label encoder on output and remove the output column from the feature vector
    def map_output (self):
        '''
        Mapping the output to a numeric range, separate the features and output, and drop the id and label columns from the features list.
        ----------
        
        Returns
        -------
        self.data:
            Separate features data 
        
        self.labels:
            Ground truth or label of feature data

        
        '''
        ## Mapping the output to a numeric range
        output_column_name = self.data.columns[-1]
        list_uniques = list(self.data[output_column_name].unique())
        self.data[output_column_name] = self.data[output_column_name].map({list_uniques[0]: 1, list_uniques[1]: 0})
        ## Separate the features and output
        self.labels = self.data[output_column_name] 
        ## Drop the id and label columns from the features list
        # print(self.data.columns)
        for item in self.data.columns:
            
            
            if ('id' in item) or ('ID' in item):
                # print(item)
                self.data = self.data.drop([item], axis=1) ##, 'customerID'
        # print(self.data.columns)
        self.data = self.data.drop([output_column_name], axis=1)
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