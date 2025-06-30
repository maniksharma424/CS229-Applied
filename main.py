# %%
print("Hello, World!")
# %%
import pandas as pd
pd.read_csv("Ecommerce Customers")
# %%
# view initial data rows
df = pd.read_csv("Ecommerce Customers")
df.head()
# %%
# info about cols
df.info()
# %%
df.describe()
# %%
df.columns
# %%
df['Email'].head()
# %%
df['Email'].tail()
# %%
df['Email'].unique()
# %%
df.describe()
# %%
import seaborn as sns
# Distribution graph of a variable and target
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df , alpha=0.5)

# %%
sns.pairplot(df , palette='coolwarm')
# %%
# visualise all data grapgs to find symetry
sns.lmplot(df,x='Length of Membership', y='Yearly Amount Spent' , scatter_kws={"alpha":0.5})
# %%
from sklearn.model_selection import train_test_split
# %%
# Defining Predictive Variables
X = df[['Avg. Session Length' , 'Time on App' , 'Time on Website' , 'Length of Membership']]
#  Defining  Target Variables
Y = df['Yearly Amount Spent']
X,Y

# %%
# make actual splits 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
X_train,Y_train,X_test, Y_test
# %%
# Train the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
# fir the data an run predict method  
lm.fit(X_train,Y_train)
predictions = lm.predict(X_test)
# plot predictions agains text split t verify  
sns.scatterplot(x=predictions, y=Y_test)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
# deduce errors between predection and actual values 
print("mean absolute error " , mean_absolute_error(Y_test,predictions))
print("mean squared error " , mean_squared_error(Y_test,predictions))

# %%
# residuals are difference in predictions and actual values 
residuals = Y_test- predictions
residuals
# residual plot should be normally disrtibuted as there should be no trend bw erros because they are there because of the random noise and not because of something from our model 
sns.histplot(residuals, kde=True)

# %%
import scipy.stats as stats
import matplotlib.pyplot as plt
# qq plot or prob plot when forms a straight line shows that residuals are noramlly distributed how - ? it geenrates same no of inputs as of our residuals and normally distributes them and plot against actuall residual values  if coincode means both are noramlaly disrtibuted 
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

# %%
