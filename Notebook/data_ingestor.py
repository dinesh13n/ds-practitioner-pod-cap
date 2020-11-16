import os
import glob
import pandas as pd
import numpy as np
import re

def read_json_files(DATA_DIR):
    
    #Validation for correct path and file availability
    if not os.path.isdir(DATA_DIR):
        raise Exception('Specified directory does not exist')

    if not len(os.listdir(DATA_DIR)) >0:
        raise Exception('Specified path does not containt any file')

    all_months = {} #Create empty disctionary for file data
    
    #Get all json files along with path.
    file_list = glob.glob(os.path.join(DATA_DIR,'*.json'))
    
    #Below should be correct column for available data
    correct_columns = ['country', 'customer_id', 'day', 'invoice', 'month', 'price', 'stream_id', 'times_viewed', 'year']

    for file in file_list:

        with open (file,"r") as jfile:
            
            df = pd.read_json(file) #Read json file containt
            all_months[os.path.split(file)[-1]] = df # File name would be Key and file containt would be Value on Dict.
        
    #Re-name incorrect column on json data
    for fl, df in all_months.items():
        cols = set(df.columns.tolist())

        if 'StreamID' in cols:
            df.rename(columns = {'StreamID':'stream_id'},inplace=True)
        if 'total_price' in cols:
            df.rename(columns = {'total_price':'price'},inplace=True)
        if 'TimesViewed' in cols:
            df.rename(columns = {'TimesViewed':'times_viewed'},inplace=True)

        cols = df.columns.tolist()
        if sorted(cols) != correct_columns:
            raise Exception("column name could not be matched with correct column")
    
    #Concat all the json data and convert in DataFrame.
    df = pd.concat(list(all_months.values()),sort=True)
    
    # Get the time data and format in YYYY-MM-DD
    years,months,days = df['year'].values,df['month'].values,df['day'].values
    dates = ["{}-{}-{}".format(years[i],str(months[i]).zfill(2),str(days[i]).zfill(2)) for i in range(df.shape[0])]
    
    #Create additional 'invoice_date' column in DataFrame
    df['invoice_date'] = np.array(dates,dtype='datetime64[D]')
    
    #Remove alphabet from the invoice number
    df['invoice'] = df['invoice'].apply(lambda x: re.sub("\D+","",str(x)))
    
    #Sort the data by invoice date and drop the index
    df.sort_values(by='invoice_date',inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    return df

def datatype_map(df):
    df = df.astype({"country": str, "day": int,
                            "month": int, "price": float,
                            "times_viewed": int, "year": int})
    return df