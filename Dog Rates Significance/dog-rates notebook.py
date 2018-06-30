import pandas as pd
import re
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
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

x = data['timestamp']
y = data['rating'] # y is the observed values
X = x[:, np.newaxis]
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
# let's check the slope and intercept
print(model.coef_[0], model.intercept_)

y_fit = model.predict(X) # predicted values
plt.hist(y - y_fit)
# plt.show()
plt.savefig('dog-rates-hist.png')
plt.close() # close this plt so I can plt another figure


# I will use the stats library to get p-value because it's difficult to do it with LinearRegression
fit = stats.linregress(data['timestamp'], data['rating'])
print(fit.pvalue)
# pvalue is 8.85e-92 which is a lot smaller than 0.05 so we can confidently say that the slope is different from zero

plt.xticks(rotation=25)
plt.plot(data['created_at'], data['rating'], 'b.', alpha=0.5)
plt.plot(data['created_at'], data['timestamp']*fit.slope + fit.intercept, 'r-', linewidth=3)
# plt.show()
plt.savefig('dog-rates-results-Tom.png')
