import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , RobustScaler 
import category_encoders as CategortEncoder
from imblearn.over_sampling import SMOTE 
import pickle


def feature_1(df):
    df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes'] 
    return df

def feature_2(df):
    df['total_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] 
    return df

def feature_3(df):
    df['number_customer_service_calls_binned'] = df['number_customer_service_calls'].apply(lambda x:  0 if x>=4 else 1)
    return df


def feature_engineering(df , feature1 =True , feature2 = True , feature3=True  ):
    result = df.copy()
    
    if feature1 : 
        result = feature_1(result)
    if feature2 : 
        result = feature_2(result)
    if feature3:
        result = feature_3(result)
        
    return result 


def IQR_technique(column,df):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    cut_off = IQR * 1.5
    lower = Q1 - cut_off
    upper =  Q3 + cut_off
    return lower ,upper


def get_outliers(df , col, verbose = True):
    lower ,upper = IQR_technique(col,df)
    qurey= df[(df[col]< lower) |(df[col]> upper) ] 
    if(verbose):
        print ('When chrun = ' , df['churn'].values[0],' the  number of outilers in ', col , ': ' , qurey[col].count())
    return qurey.index.values


def get_outliers_indicies(df , col , verbose = True):
        indicies_to_remove = []
        if col == 'minutes' or col =='charge' or col == 'calls' :
            indicies_to_remove=get_outliers(df , 'total_night_'+col , verbose)
            day_minutes_outliers=get_outliers(df , 'total_day_'+col , verbose)
            indicies_to_remove = np.union1d(day_minutes_outliers , indicies_to_remove)
            eve_minutes_outliers=get_outliers(df , 'total_eve_'+col ,verbose)
            indicies_to_remove = np.union1d(eve_minutes_outliers , indicies_to_remove)
        else:
            indicies_to_remove=get_outliers(df ,col ,verbose)
        if(verbose):  
            print("percentage of outliers: " + str(len(indicies_to_remove) *100 / len(df)   ))
            print('\n')

        return indicies_to_remove
        
        

def remove_outliers(df, cols, group , delete= True , verbose = True):
    result = df.copy()
    result_groupped= result[result['churn'] == group]
    indicies_removed = [] 
    for col in cols: 
        idxs = get_outliers_indicies(result_groupped , col , verbose)
        result_groupped =result_groupped.drop(idxs , axis =0 )
        indicies_removed = np.union1d(indicies_removed , idxs)
        
    if delete :
        result= result.drop(indicies_removed , axis =0 )
        return result , indicies_removed
    else : 
        indicies_removed

        
        
not_normal_distributed_cols = ['number_customer_service_calls' , 'number_vmail_messages']

def standarize_numerical(df, cols):
    result = df.copy() 
    
    for column in cols:
        if result.dtypes[column] != np.object:
            result[column] = StandardScaler().fit_transform(result[[column]]) 
    return result


def normalize_numerical(df, cols):
    result = df.copy() 
    
    normal_distributed_cols = np.setdiff1d(df.columns.values,not_normal_distributed_cols)
    for column in cols:
        if result.dtypes[column] != np.object:
            result[column] = RobustScaler().fit_transform(result[[column]]) 
            
    return result


def features_scaling(df):
    result = df.copy()

    numrical_columns= df.columns[ df.dtypes != np.object].values
    not_normal_distributed_cols = ['number_customer_service_calls' , 'number_vmail_messages']
    normal_distributed_cols = np.setdiff1d(numrical_columns,not_normal_distributed_cols) 
    result= normalize_numerical(result, not_normal_distributed_cols)
    result = standarize_numerical(result, normal_distributed_cols)

        
    return result
        
        
        

def features_selection(df, featurized):
    result = df.copy() 
    
    if(featurized):
        result= result.drop(['total_minutes' ,
                             'total_day_charge' ,
                             'total_night_charge',
                             'total_intl_charge' ,
                             'total_eve_charge' , 
                             'total_day_minutes'  ,
                             'number_customer_service_calls' ] , axis= 1)
    else :
        result = result.drop(['total_night_minutes',
                              'total_intl_minutes' ,
                              'total_eve_minutes' ,
                              'total_day_minutes'  ] , axis= 1)
        
    result = result.drop('area_code' ,axis = 1 )
        
    return result
    
    
def labelEncoder(df, col):
    
    labels = {'yes' : 1 , 'no' :0}
    result = df.copy()
    result[col+'_encoded'] =result[col].map(labels)
    
    return result
    
    

    
    
def BinaryEncoder(df,col):
    result = df.copy()
    
    with open("utils/BinaryEncoder", "rb") as f: 
        BinaryEncoder = pickle.load(f) 
    state_encoded = BinaryEncoder.transform(result[col])
    result = pd.concat([result, state_encoded], axis=1)
    encoded_col_named = state_encoded.columns

    return result , encoded_col_named

def encoder(df):
    result =df.copy()
    
    result = labelEncoder(result , 'churn')
    result = labelEncoder(result , 'international_plan')
    result = labelEncoder(result , 'voice_mail_plan')
    result , encoded_col_names  = BinaryEncoder(result , 'state')
    result.drop(['churn' ,
                 'international_plan' ,
                 'voice_mail_plan' ,
                 'state' ],axis = 1 , inplace = True)
    encoded_col_names = np .append(encoded_col_names, ['churn_encoded' ,
                                                       'international_plan_encoded' ,
                                                       'voice_mail_plan_encoded'], axis=None) 
    
    return result ,encoded_col_names

    
def KNN_oversample(df):
    
    oversample = SMOTE()
    X = df.drop('churn_encoded' ,axis = 1)
    y = df['churn_encoded']
    X_sampled, y_sampled =  oversample.fit_resample(X, y)
    print("Tatget classes counts after oversampling:\n", y_sampled.value_counts())
    
    return X_sampled , y_sampled
    
    
    