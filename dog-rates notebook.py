import pandas as pd
import re
from scipy import stats
import matplotlib.pyplot as plt
dog_data = pd.read_csv("dog_rates_tweets.csv", parse_dates=['created_at'])
#print(dog_data.head())

input_pattern = re.compile(r'\d+\/10')

def filterText(input_string):
    text_pattern = re.compile(r'\d+\/10')
    match = text_pattern.search(input_string)
    if match:
        return match.group()
    else:
        return None

dog_data['data'] = dog_data['text'].apply(filterText)


#Remove outliers - anything larger than 25/10
def validateRating(input_data):
    if input_data == None:
        return None
    data_pattern = re.compile(r'([0-2]{1,}[0-5]?|[0-9]?)\/10')
    match = data_pattern.search(input_data)
    if match:
        return float(match.group(1))
    else:
        return None

dog_data['rating'] = dog_data['data'].apply(validateRating)
data = dog_data[dog_data['rating']>=0]

def to_timestamp(dtObj):
    return dtObj.timestamp()

data['timestamp'] = data['created_at'].apply(to_timestamp)
print(data.head())



fit = stats.linregress(data['timestamp'], data['rating'])


plt.xticks(rotation=25)
plt.plot(data['created_at'], data['rating'], 'b.', alpha=0.5)
plt.plot(data['created_at'], data['timestamp']*fit.slope + fit.intercept, 'r-', linewidth=3)
# plt.show()
plt.savefig('dog-rates-results-Tom.png')
