## EX NO:3-Feature Encoding and Transformation

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ``` py
import pandas as pd
import numpy as np
from scipy import stats
```
``` py
from google.colab import drive
drive.mount('/content/drive')
```
![image](https://github.com/user-attachments/assets/47828fed-d060-48ef-973a-923617462533)
```
!ls "/content/drive/My Drive/data science" py
```
![image](https://github.com/user-attachments/assets/6f802d78-41a7-404a-b8ac-6c729c55f212)
``` py
file_path="/content/drive/My Drive/data science/Encoding Data.csv"
df=pd.read_csv(file_path)
df
```
![image](https://github.com/user-attachments/assets/5191f239-c677-409f-8f44-b7a04031069a)
# ORDINAL ENCODER
``` PY
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot','Warm','cold']
el=OrdinalEncoder(categories=[pm])
el.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/6b8f133c-bbf5-4833-ac6c-7b6982a18f30)
``` PY
df['bo2']=el.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/b80b6a19-6a79-4cb0-b759-2646e30aab7a)
# LABEL ENCODER
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/1bf7bdf7-7545-49be-b213-bc8211f9a085)
# ONEHOTENCODER
``` py
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2 = pd.concat([df2, enc], axis=1)
df2
```
![image](https://github.com/user-attachments/assets/75bb12fd-3ccd-4363-8c3d-f65feec298ea)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/7b2a829b-aeb3-44c5-b53e-624c4c07cad6)
# BINARY ENCODER
``` py
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/3388ae83-549c-4e73-b73d-32fed9ef43f2)
``` py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/drive/My Drive/data science/data.csv")
df
```
![image](https://github.com/user-attachments/assets/6b2b9037-be6c-4114-8f74-53fa2dca3c95)
``` py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/68a381f4-c015-4030-9a7d-eeddbe05c9df)
# TARGETENCODER
``` py
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/b3f9d818-8c68-44cd-adbc-29ab5c75c870)
# FEATURE TRANSFORMATION
``` py
df=pd.read_csv("/content/drive/My Drive/data science/Data_to_Transform (1).csv")
df
```
![image](https://github.com/user-attachments/assets/425b0918-3e2e-4ee4-bc1f-519eff299419)
``` py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/471b0a32-5bbd-4974-9a16-a4463e59d952)
``` py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4c8cc921-6590-494e-9f94-f5b8162b6ffa)
``` py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/54c4e2d8-5fc4-48fe-b2b4-dccac0ab37ec)
``` py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/dc71a2c5-7bfe-4c42-bf2f-b93d0ed68b44)
``` py
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/a69ca8b4-9737-4443-8ea9-ac2e84142b04)
``` py
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/0f135caa-f21f-4bb6-91d8-222c1aab939a)
``` py
df["highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/0c353d74-fa68-4f7d-8bb6-677eab3910ff)
``` py
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])

df
```

![image](https://github.com/user-attachments/assets/7e67b9b4-b9f9-480d-835e-1df9a4a4f60f)
``` py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
``` py
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/d0936080-76cc-4eb9-900e-d77e3ca037ae)
``` py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/960f5be6-8857-4916-b171-ddf33d57564e)
``` py
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal', n_quantiles=891)

df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"], line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ca72cec9-1eba-49a0-bd8f-160e87b8e749)
``` py
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"], line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a564f00d-20d3-4317-8fb3-b2667d906b4b)
``` py
sm.qqplot(df["Highly Negative Skew_1"], line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/04f889ec-fc7b-4007-aa9a-4793a71cbb09)
``` py
dt=pd.read_csv("/content/drive/My Drive/data science/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal', n_quantiles=891)
dt["Age_1"] = qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'], line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f0f59684-2d5f-4626-99b5-5bbc8d2891a3)
``` py
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/275b99dc-1cb2-4d42-b392-f695d16480c4)


# RESULT:
     Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.



       
