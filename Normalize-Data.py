import csv
import matplotlib.pyplot as plt
import math


train_csv = csv.reader(open('train.csv','rt'), delimiter=',')
train_data = list()

for data in train_csv:
    train_data.append(data)

length = len(train_data)
sale_prices = list()

for i in range(1, length):
    sale_prices.append(int(train_data[i][80]))

X = sale_prices
Y = list(i for i in range(0, len(sale_prices)))

plt.hist(X, bins=20)
plt.xlabel("sale_prices")
plt.ylabel("frequency")
plt.title("Raw Sale Price Plot")
plt.show()

log_sale_prices = list()
for price in sale_prices:
    log_sale_prices.append(math.log(price))

plt.hist(log_sale_prices, bins=20)
plt.xlabel("log_sale_prices")
plt.ylabel("frequency")
plt.title("Log Sale Price Plot")
plt.show()

