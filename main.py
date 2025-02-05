import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

#Reading Training csv dataset using pandas
try:
    #Taking necessary columns from dataset
    tds = pnd.read_csv("train.csv", usecols=["LotArea", "SalePrice","FullBath","BedroomAbvGr"])
except FileNotFoundError:
    print("File not found.")
    exit()


#Code to print and see the dataset and find max value in it uncomment if you need
#print(tds)
#print(tds[tds['LotArea']==tds['LotArea'].max()])


#For drawing Area vs Sale Price
#plt.scatter(tds.LotArea,tds.SalePrice,color='red')
#plt.xlabel("Area")
#plt.ylabel("Price")
#plt.grid(True)
#plt.show()


#Training The model
x = tds[["LotArea","FullBath","BedroomAbvGr"]]
y = tds["SalePrice"]
reg.fit(x,y)


#Creating a new panda frame and testing a new test data uncomment and use this for single testing
#test = pnd.DataFrame({'LotArea': [11622], 'FullBath': [1], 'BedroomAbvGr': [2]})
#print(reg.predict(test))


#Creating a new panda frame and testing a new test data and saving it, for larger predictions
ts = pnd.read_csv("test.csv", usecols=["LotArea","FullBath","BedroomAbvGr"])
tl = reg.predict(ts)
ts["SalePrice"] = tl
print(ts)
ts.to_csv('test_result.csv',index=False)