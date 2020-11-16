import requests
from ast import literal_eval
import json


## data needs to be in dict format for JSON
#query = X_new.to_dict()
dict_query = {'country':'United Kingdom','year':'2018','month':'01','day':'05'}
query = json.dumps(dict_query)
print (query)
## test the Flask API
port = 5000
r = requests.post('http://127.0.0.1:{}/predict'.format(port),json=query)

## test the Docker API
#port = 5000
#r = requests.post('http://127.0.0.1:{}/predict'.format(port),json=query)
print('----r--- :',r.content)
response = literal_eval(r.text)
print(response)