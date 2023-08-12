#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[3]:


get_ipython().system('pip install xgboost')


# In[3]:


# Data manipulation and analysis libraries
import pandas as pd
import numpy as np

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci

# Supress warnings
import warnings 
warnings.filterwarnings("ignore")

# Multi-collinearity check library 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Data preprocessing libraries
from sklearn.preprocessing import StandardScaler

# Feature Decomposition library
from sklearn.decomposition import PCA

# Model Selection libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

# Machine Learning Model Libraries
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost

# Unsupervised ML Libraries
from sklearn.cluster import KMeans


# # Loading the Dataset

# In[4]:


df = pd.read_excel("Capstone Project.xlsx")
df.head()


# Problem Statement : Create a machine learning model with the utmost accuracy, hyper parametrically tune the model and using feature engineering identify the features which are contributing the most in prediction of strength.

# # Exploratory Data Analysis (EDA)

# In[4]:


df.info()


# In[5]:


df.describe()


# Analysis : 
# 1. There are no missing values in the dataset
# 2. There are no outliers in lower whisker region for ash, super plastic since the minimum , Q1, Q2 are exactly the same
# 3. The difference between mean and median is high which is assigned that slag,ash,age might have outliers

# In[6]:


df["ash"].plot(kind = "box")


# In[7]:


df["superplastic"].plot(kind = "box")


# ### Building a custom summary function for EDA report

# In[8]:


def describe(my_df):
    cols = []
    for i in my_df.columns:
        if my_df[i].dtype != object:
            cols.append(i)
    
    result = pd.DataFrame(columns = cols, index = ["Data Type", "Count", "Min", "Q1", "Q2", "Q3", "Max","SD","Mean","Var",
                                                  "Skew", "Kurt", "Range", "IQR", "Skewness Comment", "Kurtosis Comment",
                                                   "Outlier Comment"])
    for i in result.columns :
        result.loc["Data Type",i] = my_df[i].dtype
        result.loc["Count",i] = my_df[i].count()
        result.loc["Min",i] = my_df[i].min()
        result.loc["Q1",i] = my_df[i].quantile(0.25)
        result.loc["Q2",i] = my_df[i].median()
        result.loc["Q3",i] = my_df[i].quantile(0.75)
        result.loc["Max",i] = my_df[i].max()
        result.loc["SD",i] = my_df[i].std()
        result.loc["Mean",i] = my_df[i].mean()
        result.loc["Var",i] = my_df[i].var()
        result.loc["Skew",i] = my_df[i].skew()
        result.loc["Kurt",i] = my_df[i].kurt()
        result.loc["Range",i] = my_df[i].max()-my_df[i].min()
        result.loc["IQR",i] = my_df[i].quantile(0.75)-my_df[i].quantile(0.25)
        
        
        # Adding comments for skewness
        if result.loc["Skew",i] <= -1:
            sk_label = "Highly negatively skewed"
        elif -1 < result.loc["Skew",i] <= -0.5:
            sk_label = "Moderately negatively skewed"
        elif -0.5 < result.loc["Skew",i] < 0:
            sk_label = "Approx normal distribution(-ve)"
        elif 0 <= result.loc["Skew",i] < 0.5:
            sk_label = "Approx normal distribution(+ve)"
        elif 0.5 <= result.loc["Skew",i] < 1:
            sk_label = "Moderately positively skewed"
        elif result.loc["Skew",i] >= 1:
            sk_label = "Highly positively skewed"
        else:
            sk_label = "Error"
        result.loc["Skewness Comment",i] = sk_label
        
        # Adding comments for kurtosis
        if result.loc["Kurt",i] <= -1:
            kt_label = "Highly platykurtic curve"
        elif -1 < result.loc["Kurt",i] <= -0.5:
            kt_label = "Moderately platykurtic curve"
        elif -0.5 < result.loc["Kurt",i] < 0.5:
            kt_label = "Mesokurtic curve"
        elif 0.5 <= result.loc["Kurt",i] < 1:
            kt_label = "Moderately leptokurtic curve"
        elif result.loc["Kurt",i] >= 1:
            kt_label = "Highly leptokurtic curve"
        else:
            kt_label = "Error"
        result.loc["Kurtosis Comment",i] = kt_label
        
        
        # Adding comments for outliers
        lw = result.loc["Q1",i] - 1.5*result.loc["IQR",i]
        uw = result.loc["Q3",i] + 1.5*result.loc["IQR",i]
        if len([x for x in my_df[i] if x < lw or x > uw]) > 0:
            outlier_label = "Have Outliers"
        else:
            outlier_label = "No Outliers"
        result.loc["Outlier Comment",i] = outlier_label
            
    display(result)


# In[9]:


describe(df)


# Analysis :
# 1. There are 6 features which have outliers in them so we need to perform outlier treatment on the data set.
# 2. Age is the only feature which is highly skewed (positively)
# 3. There are 3 features which have high kurtosis they are ash, superplastic, age.
# 

# # Outlier Treatment

# In[10]:


def replace_outlier(my_df,col,method="Quartile",strategy="Median"):
    col_data = my_df[col]
    
    # Using Quartile method to detect the outliers
    if method == "Quartile":
        Q1 = col_data.quantile(0.25)
        Q2 = col_data.quantile(0.50)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        LW = Q1 - 1.5*IQR
        UW = Q3 + 1.5*IQR
    # Using standard deviation to detect the outliers
    elif method == "Standard Deviation":
        Mean = col_data.mean()
        STD = col_data.std()
        LW = Mean - 2*STD
        UW = Mean + 2*STD
        
    # The empirical rule states the for a perfectly normal distribution 68% of data points 
    # lie within the first standard deviation, 95% of data points lie within the second standard deviation
    # and 99.7% of data points lie within the third standard deviation.
    
    else:
        print("Pass the correct method")
        pass
    
    # Detecting all the outliers
    outliers = my_df.loc[(col_data < LW) | (col_data > UW),col]
    outliers_density = round((len(outliers)/len(my_df)) * 100,2)
    if len(outliers) == 0:
        print(f"Feature {col} does not have any outliers")
    else:
        print(f"Feature {col} has outliers")
        print(f"Total number of outliers in {col} are {len(outliers)}")
        print(f"Outlier percentage in {col} is {outliers_density}%")
        display(my_df[(col_data < LW) | (col_data > UW)])
    
    # Treating the ouliers
    if strategy == "Median":
        df.loc[(col_data < LW) | (col_data > UW), col] = Q2
    elif strategy == "Mean":
        df.loc[(col_data < LW) | (col_data > UW), col] = Mean
    else:
        print("Pass a correct strategy")
        pass
    return df


# # ODT Plots (Outlier Detection and Treatment Plots)
# 
# 1. Descriptive plots
# 2. Histogram with outliers
# 3. Histogram without outliers

# In[11]:


def odt_plots(df,col):
    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = (25,8))
    
    # Descriptive Statistics Box Plot
    sns.boxplot(df[col], ax = ax1)
    ax1.set_title(col + " box plot ")
    ax1.set_xlabel("box plot")
    ax1.set_ylabel("Values")
    
    # Plotting histogram with outliers
    sns.distplot(df[col], ax = ax2,fit = sci.norm)
    ax2.axvline(df[col].mean(),color="green")
    ax2.axvline(df[col].median(),color="brown")
    ax2.set_title(col + " histogram with outliers")
    ax2.set_xlabel("Values")
    ax2.set_ylabel("Density")
    
    # Plotting histogram without outliers
    df_out = replace_outlier(df,col)
    
    sns.distplot(df_out[col], ax = ax3,fit = sci.norm)
    ax3.axvline(df_out[col].mean(),color="green")
    ax3.axvline(df_out[col].median(),color="brown")
    ax3.set_title(col + " histogram with outliers")
    ax3.set_xlabel("Values")
    ax3.set_ylabel("Density")
    
    plt.show()


# In[12]:


for col in df.columns:
    odt_plots(df,col)


# # Multivariate Analysis using Regression

# In[13]:


for col in df.columns:
    if col != "strength":
        fig,ax1 = plt.subplots(figsize = (10,5))
        sns.regplot(x = df[col], y = df["strength"], ax = ax1).set_title(
        f"Relationship between {col} and strength")


# Analysis:
# 1. Strength and cement are highly positively correlated
# 2. Strength and slag are slightly positively correlated
# 3. Strength and ash are slightly negatively correlated
# 4. Strength and water are highly negatively correlated
# 5. Strength and superplastic are highly positively correlated
# 6. Strength and coarseagg are moderately negatively correlated
# 7. Strength and fineagg are moderately negatively correlated
# 8. Strength and age are highly positively correlated

# # Multi-collinearity Test
# 
# - Stage 1
#     - Correlation heat map/matrix

# In[5]:


corr = df.corr()
f,ax = plt.subplots(figsize = (8,8))
sns.heatmap(corr,annot = True)


# Analysis :
# 1. Superplastic and Ash has 45% correlation
# 2. Cement and ash has 40% correlation
# 3. Slag and ash has 32% correlation
# 4. Superplastic and water has 66% correlation
# 5. Fineagg and water has 43% correlation
# 
# - Many independent features have collinearity more than 30%, so we can conclude multi-collinearity exists as of stage 1.

# # Multi-collinearity Test
# 
# - Stage 2
# - VIF (Variance Inflation Factor)
# - Formula for VIF = 1/(1 - R2)
# - Steps to calculate VIF :
#     1. Regress every independent variable with each other and find the R2 score
#     2. Calculate the VIF using above formula
#     3. If VIF > 5 for any independent variable, we conclude that multi-collinearity exists.
#     
# 

# In[15]:


def VIF(features):
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(features.values,i) for i in range(features.shape[1])]
    vif["Features"] = features.columns
    return vif 


# In[16]:


VIF(df.drop("strength", axis = 1))


# Analysis :
# 1. Cement, Water, Superplastic, Coarseagg, Fineagg have VIF scores > 5 which means multicollinearity exist in the data.

# - VIF = 1/(1-R2)
# - 1-R2 = 1/VIF
# - R2 = 1 - (1/VIF)

# In[17]:


# R2 score for calculation of VIF
1-(1/86.93)


# # PCA (Principal Component Analysis)
# 
# - Applying PCA to treat multi-collinearity

# In[6]:


def PCA1(x):
    n_com = len(x.columns)
    
    # Standardizing the data (Applying standard scaler)
    x = StandardScaler().fit_transform(x)
    
    # Applying PCA
    for i in range(1,n_com):
        pca = PCA(n_components = i)
        pcom = pca.fit_transform(x)
        evr = np.cumsum(pca.explained_variance_ratio_)
        if evr[i-1] > 0.9:
            n_components = i
            break
    
    print("Explained Variance Ratio after PCA is : ", evr)
    
    # Creating the final dataframe
    col = []
    for j in range(1,n_components+1):
        col.append("PC_"+str(j))
    
    pca_df = pd.DataFrame(pcom, columns = col)
    return pca_df


# In[7]:


transform_df = PCA1(df.drop("strength",1))
transform_df.head()


# In[20]:


transform_df.info()


# In[21]:


transform_df = transform_df.join(df["strength"],how = "left")
transform_df.head()


# # Model Building
# 
# 1. Train-Test-Split
# 2. Cross validation
# 3. Hyperparameter Tuning

# In[22]:


def tts(data,tcol,testsize = 0.3):
    x = data.drop(tcol, axis = 1)
    y = data[tcol]
    return train_test_split(x,y,test_size = testsize, random_state = 89)


# In[23]:


def build_model(model_name,estimator,data,tcol):
    x_train,x_test,y_train,y_test = tts(data,tcol)
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    temp = [model_name, rmse, r2]
    return temp


# In[24]:


build_model("Linear Regression",LinearRegression(),transform_df,"strength")


# In[25]:


def multiple_models(data,tcol):
    col_name = ["Model Name","RMSE","R2 Score"]
    result = pd.DataFrame(columns = col_name)
    result.loc[len(result)] = build_model("Linear Regression",LinearRegression(),transform_df,"strength")
    result.loc[len(result)] = build_model("Lasso Regression",Lasso(),transform_df,"strength")
    result.loc[len(result)] = build_model("Ridge Regression",Ridge(),transform_df,"strength")
    result.loc[len(result)] = build_model("KNN",KNeighborsRegressor(),transform_df,"strength")
    result.loc[len(result)] = build_model("Decision Tree",DecisionTreeRegressor(),transform_df,"strength")
    result.loc[len(result)] = build_model("Random Forest",RandomForestRegressor(),transform_df,"strength")
    result.loc[len(result)] = build_model("Support Vector Machine",SVR(),transform_df,"strength")
    result.loc[len(result)] = build_model("GBoost",GradientBoostingRegressor(),transform_df,"strength")
    result.loc[len(result)] = build_model("XGBoost",XGBRegressor(),transform_df,"strength")
    result.loc[len(result)] = build_model("AdaBoost",AdaBoostRegressor(),transform_df,"strength")
    
    return result


# In[26]:


multiple_models(transform_df,"strength")


# ### Cross Validation

# In[27]:


def k_fold_cv(x,y,fold = 10):
    score_lr = cross_val_score(LinearRegression(),x,y,cv = fold)
    score_la = cross_val_score(Lasso(),x,y,cv = fold)
    score_rd = cross_val_score(Ridge(),x,y,cv = fold)
    score_kn = cross_val_score(KNeighborsRegressor(),x,y,cv = fold)
    score_dt = cross_val_score(DecisionTreeRegressor(),x,y,cv = fold)
    score_rf = cross_val_score(RandomForestRegressor(),x,y,cv = fold)
    score_svm = cross_val_score(SVR(),x,y,cv = fold)
    score_gb = cross_val_score(GradientBoostingRegressor(),x,y,cv = fold)
    score_xgb = cross_val_score(XGBRegressor(),x,y,cv = fold)
    score_adb = cross_val_score(AdaBoostRegressor(),x,y,cv = fold)
    
    model_names = ["LinearRegression","Lasso","Ridge","KNeighborsRegressor","DecisionTreeRegressor",
                   "RandomForestRegressor","SVR","GradientBoostingRegressor","XGBRegressor","AdaBoostRegressor"]
    scores = [score_lr,score_la,score_rd,score_kn,score_dt,score_rf,score_svm,score_gb,score_xgb,score_adb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        name = model_names[i]
        temp = [name,score_std,score_mean]
        result.append(temp)
    result_df = pd.DataFrame(result,columns = ["Model Name","Score SD","Mean Score"])
    return result_df.sort_values(by = "Mean Score", ascending = False)


# In[28]:


k_fold_cv(transform_df.drop("strength",axis = 1),transform_df["strength"])


# In[29]:


def tuning(x,y,fold = 10):
    
    # Creating the parameter grids for tuning
    param_xgb = {"alpha" : [0,1], "gamma" : [0,10,20,30,40,50,60,70,80,90,100], "reg_lambda" : [0,1]}
    param_rf = {"n_estimators" : [50,80,100,120,150], "max_features" : ["auto","log2","sqrt"], 
                "max_depth" : [5,7,10,13,15]}
    
    # Creating model objects
    tune_xgb = GridSearchCV(XGBRegressor(), param_xgb, cv = fold)
    tune_rf = GridSearchCV(RandomForestRegressor(), param_rf, cv = fold)
    
    # Tuning the models
    tune_xgb.fit(x,y)
    tune_rf.fit(x,y)
    
    # Extracting the best hyperparameters
    tune = [tune_xgb,tune_rf]
    names = ["XGBoost","Random Forest"]
    for i in range(len(tune)):
        print("Model : ",names[i])
        print("Best parameters : ",tune[i].best_params_)


# In[30]:


tuning(transform_df.drop("strength",axis = 1),transform_df["strength"])


# In[31]:


def cv_post_hpt(x,y,fold = 10):
    score_lr = cross_val_score(LinearRegression(),x,y,cv = fold)
    score_la = cross_val_score(Lasso(),x,y,cv = fold)
    score_rd = cross_val_score(Ridge(),x,y,cv = fold)
    score_kn = cross_val_score(KNeighborsRegressor(),x,y,cv = fold)
    score_dt = cross_val_score(DecisionTreeRegressor(),x,y,cv = fold)
    score_rf = cross_val_score(RandomForestRegressor(max_depth = 13, max_features = "auto", n_estimators = 80),x,y,cv = fold)
    score_svm = cross_val_score(SVR(),x,y,cv = fold)
    score_gb = cross_val_score(GradientBoostingRegressor(),x,y,cv = fold)
    score_xgb = cross_val_score(XGBRegressor(alpha = 1, gamma = 0, reg_lambda = 1),x,y,cv = fold)
    score_adb = cross_val_score(AdaBoostRegressor(),x,y,cv = fold)
    
    model_names = ["LinearRegression","Lasso","Ridge","KNeighborsRegressor","DecisionTreeRegressor",
                   "RandomForestRegressor","SVR","GradientBoostingRegressor","XGBRegressor","AdaBoostRegressor"]
    scores = [score_lr,score_la,score_rd,score_kn,score_dt,score_rf,score_svm,score_gb,score_xgb,score_adb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        name = model_names[i]
        temp = [name,score_std,score_mean]
        result.append(temp)
    result_df = pd.DataFrame(result,columns = ["Model Name","Score SD","Mean Score"])
    return result_df.sort_values(by = "Mean Score", ascending = False)


# In[32]:


cv_post_hpt(transform_df.drop("strength",axis = 1),transform_df["strength"])


# # Clustering

# In[33]:


cluster_model = KMeans(n_clusters = 2, random_state = 34)
clusters = cluster_model.fit_predict(df.drop("strength",axis = 1))
sns.scatterplot(x = df["cement"],y = df["strength"],hue = clusters)


# In[34]:


def clustering(x,tcol,cluster):
    column = list(set(list(x.columns)) - set(["strength"])) # Avoiding data leakage
    r = int(len(column)/2)
    if len(column) % 2 != 0:
        r += 1 # r = r+1
    f,ax = plt.subplots(r,2,figsize = (15,15))
    a = 0
    for row in range(r):
        for col in range(0,2):
            if a != len(column):
                ax[row][col].scatter(x[tcol],x[column[a]],c=cluster)
                ax[row][col].set_xlabel(tcol)
                ax[row][col].set_ylabel(column[a])
                a += 1


# In[35]:


x = df.drop("strength",axis = 1)
for col in x.columns :
    clustering(x,col,clusters)


# In[36]:


temp_df = df.join(pd.DataFrame(clusters,columns = ["cluster"]),how = "left")
temp_df.head()


# In[37]:


temp_df2 = temp_df.groupby("cluster")["cement"].agg(["mean","median"])
temp_df2.head()


# In[38]:


cluster_df = temp_df.merge(temp_df2, on = "cluster", how = "left")
cluster_df.head()


# In[39]:


multiple_models(cluster_df,"strength")


# In[40]:


x = cluster_df.drop("strength",axis = 1)
y = cluster_df["strength"]
k_fold_cv(x,y)


# In[41]:


cv_post_hpt(x,y)


# # Feature Importance

# In[42]:


xgb = XGBRegressor(alpha = 1, gamma = 0, reg_lambda = 1)
x_train,x_test,y_train,y_test = tts(cluster_df.drop("cluster",axis = 1), "strength")
xgb.fit(x_train,y_train)
xgboost.plot_importance(xgb)


# Analysis:
# 1. Age and Cement are the important features in the XGBoost model.

# In[43]:


temp_df3 = cluster_df[["age","cement","strength","water","coarseagg","fineagg"]]
cv_post_hpt(temp_df3.drop("strength",axis = 1), temp_df3["strength"])


# # Learning Curve Analysis

# In[45]:


def generate_learning_curve(model_name,estimator,x,y):
    train_size,train_score,test_score = learning_curve(estimator = estimator, X = x, y = y,cv = 10)
    train_score_mean = np.mean(train_score, axis = 1)
    test_score_mean = np.mean(test_score, axis = 1)
    plt.plot(train_size,train_score_mean,c = "blue")
    plt.plot(train_size,test_score_mean, c = "red")
    plt.title("Learning Curve for "+model_name)
    plt.xlabel("Samples")
    plt.ylabel("Accuracies")
    plt.legend(("Training accuracy","Testing accurcay"))


# In[46]:


generate_learning_curve("Linear Regression",LinearRegression(),x,y)


# In[47]:


models = [LinearRegression(),Lasso(),Ridge(),KNeighborsRegressor(),DecisionTreeRegressor(),
         SVR(),AdaBoostRegressor(),GradientBoostingRegressor(),RandomForestRegressor(),XGBRegressor()]
for a,model in enumerate(models):
    fg = plt.figure(figsize = (6,3))
    ax = fig.add_subplot(5,2,a+1)
    generate_learning_curve(type(models[a]).__name__, model, x, y)


# # Model Predictions

# In[2]:


test_df = pd.read_excel("Test Capstone 1.xlsx")
test_df.head(6)


# In[3]:


x_train.head()


# In[59]:


test_df["cluster"] = cluster_model.predict(test_df)
test_df.head(6)


# In[60]:


test_df = test_df.merge(temp_df2, on = "cluster", how = "left")
test_df.head(6)


# In[61]:


test_df["predictions"] = xgb.predict(test_df.drop("cluster", axis = 1))
test_df.head(6)


# In[ ]:




