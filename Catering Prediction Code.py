#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

import os

os.chdir('MSAS/Independent Project/sifted_data')

#import service data
hostlog= pd.read_csv('lunch_metrics.csv')

#import menu list

menus = pd.read_csv('menu_list.csv')


#merge menu to data round 1 & create standardized menus

log_stan = pd.merge(hostlog, menus , how = 'left', left_on = 'Menu', right_on = 'menu')

menuless = pd.merge(hostlog, menus , how = 'outer', left_on = 'Menu', right_on = 'menu')

menulesslist = menuless['Menu']

menulesslist = menulesslist.drop_duplicates()

menulesslist.to_csv('menulesslist.csv')

#Manually clean data, reupload new mappings

menu_map = pd.read_csv('menu_map.csv')

#merge new mappings to hostlogs

log_stan2 = pd.merge(log_stan, menu_map, how = 'inner', left_on = 'Menu', right_on = 'Menu')

log_stan2 = log_stan2.dropna(subset = ['mapping_name'])

log_stan2.groupby('mapping_name').count()

#remove drop off services
log_stan3 = log_stan2[log_stan2['Onsite']=='Onsite']

log_stan3 = log_stan3.dropna(axis=1, how = 'all')

element_table = pd.read_csv('Cleaned_Menu Data2.csv')

protein_table = element_table[element_table['New Element Type']=='Protein']

#protein_list = protein_table[['V4 Menu','Cuisine','Protein','Dish Title','Ingredients','Standard Method']]

protein_list2 = protein_table[['ingredients']]

#full corpus
protein_list2 = element_table[['ingredients']]


#protein ingredients for word2vec

df2 = protein_list2.apply(lambda x: ','.join(x.astype(str)), axis=1)

proteinclean = pd.DataFrame({'clean': df2})

proteinlist2 = [row.split(',') for row in proteinclean['clean']]

#Word2Vec process on ingredient list
import gensim
from gensim.test.utils import common_texts, get_tmpfile 
from gensim.models import Word2Vec


protein_model2 = Word2Vec(proteinlist2, min_count=1,size= 50,workers=3, window =3, sg = 1)

#produces list of protein dishes from Word2Vec model
proteinoutput2 = protein_model2.wv.index2word

#graph
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#pulls embedded vector values and creates tuple for each elemenent in corpus
corpus = sorted(protein_model2.wv.vocab.keys()) 
emb_tuple = tuple([protein_model2[v] for v in corpus])
X = np.vstack(emb_tuple)

#creates 2 t-SNE dimenesion coponenets from embedded vector values
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])


#zips ingredients with X & Y dimensional components
recipevec = pd.DataFrame({'X':X_tsne[:, 0],'Y': X_tsne[:, 1],'ingredient':corpus})

import plotly.express as px

from plotly.offline import plot

#plots ingredients using t-SNE components on 2 dimensional scatterplot
fig = px.scatter(recipevec, x="X", y="Y", hover_data=['ingredient'])
plot(fig)


#write recipe vec to dictionary, pair dictionary back into recipe for dish
ingreddict = recipevec.set_index('ingredient').T.to_dict()

ingredlists = pd.DataFrame.from_records(proteinlist2)

ingredlists = pd.DataFrame.from_records(proteinlist2)

ingredlists2 = pd.DataFrame(np.array(proteinlist2),columns=['ingredients'])


    
#join t-sne components back to each menu based on ingredients
protein_tablev = protein_table.reset_index()

protein_tablev = pd.merge(protein_tablev,ingredlists2,how='left',left_index=True, right_index=True)


#aggregate each protein element as a sum of t-sne ingredient component values
lst_col = 'ingredients_y'

test = pd.DataFrame({
     col:np.repeat(protein_tablev[col].values, protein_tablev[lst_col].str.len())
     for col in protein_tablev.columns.difference([lst_col])
     }).assign(**{lst_col:np.concatenate(protein_tablev[lst_col].values)})[protein_tablev.columns.tolist()]

test = pd.merge(test,recipevec,how='left',left_on='ingredients_y',right_on='ingredient')    
    
testgroup = test.groupby(['V4 Menu','Cuisine']).agg({'X':'sum','Y':'sum'})

testpivot = pd.pivot_table(test,index=['V4 Menu','Cuisine'],values=['X','Y'],aggfunc=np.sum).reset_index()

#plot menus
fig2 = px.scatter(testpivot, x="X", y="Y", hover_data=['V4 Menu','Cuisine'],color ='Cuisine')
plot(fig2)


#join service data with the newly created menu components

linregtest= pd.merge(log_stan3,testpivot, how ='left',left_on='mapping_name', right_on='V4 Menu')

#fill in all secondary protein missing values with zero
linregtest['pro2_pan_cnt']=linregtest['pro2_pan_cnt'].fillna(0)
linregtest['pro2_back'] =linregtest['pro2_back'].fillna(0)

#create target variable, all protein pans brought minus left overs
linregtest['panconsumption'] =(linregtest['pro1_pan_cnt']-linregtest['pro1_back']+linregtest['pro2_pan_cnt']-linregtest['pro2_back'])


#graph target distributions
fig3 = px.histogram(linregtest,x='panconsumption')

plot(fig3)

fig3 = px.histogram(linregtest,x='pro1_back')

plot(fig3)


#first round of feature selection in the data

lincolumns = ['Service_Date','menu','client','servetime_sched','ontime_status','setup_type','headcount','attendance',
              'meal_start','meal_end','anim_pro_bool','pro1_type','pro1_pan_cnt','pro1_pce_cnt','pro1_crt_cnt','pro1_avail_bool',
              'pro1_end','pro1_pct_end','pro1_pce_left','pro1_back','pro1_pce_back','pro2_bool'
              ,'pro2_type','pro2_bool','pro2_pan_cnt','pro2_pce_cnt','pro2_crt_cnt','pro2_avail_bool','pro2_end'
              ,'pro2_pct_end','pro2_pce_left','pro2_pce_back','veg_bool','veg_type','veg_pan_cnt','veg_pan_cnt','veg_pce_cnt'
              ,'veg_pan_cnt','veg_pce_cnt','starch_bool','starch_type','starch_pan_cnt','starch_crt_cnt','side_bool','side_type',
              'side_pan_cnt','side_pce_cnt','side_crt_cnt','side2_bool','side2_type','side2_pan_cnt','side2_crt_cnt','salad_bool'
              ,'salad_type','salad_pan_cnt','salad_crt_cnt','grbr_bool','city','Dish Title','V4 Menu','X','Y','panconsumption']

lintestcol = linregtest[lincolumns]


lintest1 = linregtest[['headcount','pro1_pan_cnt','panconsumption','X','Y']]

lintest1.describe()

#variable selection screened for data leakage

lincolumns2 = ['Service_Date','headcount','attendance',
              'pro1_pan_cnt'
              ,'pro2_pan_cnt'
              ,'veg_pan_cnt'
              ,'starch_pan_cnt',
              'side_pan_cnt','side2_pan_cnt'
              ,'salad_pan_cnt','X','Y','panconsumption','city']

#data cleanining
lintest1 = linregtest[lincolumns2].copy()

#reformat service date for dummy coding procedure

lintest1['Service_Date'] = pd.to_datetime(lintest1['Service_Date'],format="%m/%d/%Y")

#extract days of week from dates
lintest1['weekday'] = lintest1['Service_Date'].dt.day_name()

#dummy code day of week & city variable
lintest1 = pd.get_dummies(lintest1,columns=['city','weekday'])

#impute missing pan counts with 0
lintest1.isnull().sum(axis = 0)

lintest1[['pro1_pan_cnt','pro2_pan_cnt','veg_pan_cnt','starch_pan_cnt','side_pan_cnt','side2_pan_cnt','salad_pan_cnt']]=lintest1[['pro1_pan_cnt','pro2_pan_cnt','veg_pan_cnt','starch_pan_cnt','side_pan_cnt','side2_pan_cnt','salad_pan_cnt']].fillna(0)

lintest1.isnull().sum(axis = 0)

#remove potential data entry errors or impossible values
lintest1 = lintest1[lintest1['panconsumption']>0]

lintest1 = lintest1[lintest1['panconsumption']<20]

lintest1= lintest1.dropna()

#full model features = lintest1[['headcount','pro2_pan_cnt','veg_pan_cnt','starch_pan_cnt','side_pan_cnt','side2_pan_cnt','salad_pan_cnt','X','Y','city_ATL','city_ATX','city_BNA','city_DEN','city_PHX','city_SEA','weekday_Monday','weekday_Tuesday','weekday_Wednesday','weekday_Thursday','weekday_Friday']]

#features for full model
#features = lintest1[['headcount','pro2_pan_cnt','veg_pan_cnt','starch_pan_cnt','side_pan_cnt','side2_pan_cnt','salad_pan_cnt','city_ATL','city_ATX','city_BNA','city_DEN','city_PHX','city_SEA','weekday_Monday','weekday_Tuesday','weekday_Wednesday','weekday_Thursday','weekday_Friday']]

#current error rate for consumption performance benchmarking
lintest1error = lintest1

linpos = lintest1error[lintest1error['panconsumption']>=0]

lintest1error['pan_back'] = lintest1error['pro1_pan_cnt']+lintest1error['pro2_pan_cnt'] - lintest1error['panconsumption']

lintest1error['pan_back_sq'] = lintest1error['pan_back']**2


#screened features
features = lintest1[['headcount','pro2_pan_cnt','starch_pan_cnt','side_pan_cnt','salad_pan_cnt','city_ATL','city_ATX','city_BNA','city_DEN','city_PHX','weekday_Monday','weekday_Tuesday','weekday_Wednesday','weekday_Thursday']]

#target
pancount = lintest1[['panconsumption']]

#Linear Regression for Pan Count
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#split into training and testing datasets
features_train, features_test, pancount_train, pancount_test = train_test_split(features, pancount, test_size=0.3, random_state=0)

import statsmodels.api as sm
from scipy import stats

#get outputs from 
X2 = sm.add_constant(features_train)
est = sm.OLS(pancount_train, X2)
est2 = est.fit()
print(est2.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.Series([variance_inflation_factor(X2.values, i) 
               for i in range(X2.shape[1])], 
              index=X2.columns)


regressor = LinearRegression()  
regressor.fit(features_train, pancount_train)
 #training the algorithm
 
 #get outputs from multiple linear regression
X2 = sm.add_constant(features_train)
est = sm.OLS(pancount_train, X2)
est2 = est.fit()
print(est2.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.Series([variance_inflation_factor(X2.values, i) 
               for i in range(X2.shape[1])], 
              index=X2.columns)
 
 #To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

#predict pan counts using linear regression
pancount_pred = regressor.predict(features_test)

#plot sample of first 25 predictions of actual vs predicted values
pancount_test1 = pancount_test['panconsumption'].values.reshape(-1,1)

consumresults = pd.DataFrame({'Actual': pancount_test1.flatten(), 
                              'Predicted': pancount_pred.flatten()})

consumresults1 = consumresults.head(25)
consumresults1.plot(kind='bar',figsize=(16,10))
plt.title('Linear Regression')
plt.xlabel('Service Records')
plt.ylabel('Pan Count')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#residuals for linear model
linear_residuals = consumresults['Actual']- consumresults['Predicted'] 

linear_residuals = pd.DataFrame(linear_residuals, columns = ['Residuals'])

#plot residuals
fig5 = px.histogram(linear_residuals,x='Residuals')

plot(fig5)

#count of negative and positive residuals
linpos = linear_residuals[linear_residuals['Residuals']>=0]

linneg = linear_residuals[linear_residuals['Residuals']<0]

print(linpos['Residuals'].mean())
print(linneg['Residuals'].mean())


#print model outputs
print('Mean Absolute Error:', metrics.mean_absolute_error(pancount_test, pancount_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(pancount_test, pancount_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(pancount_test, pancount_pred)))
print('R-squared:',metrics.r2_score(pancount_test, pancount_pred,multioutput = 'raw_values'))



#Regression Tree for Pan Count
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(features_train, pancount_train)

regtree_model = regressor.fit(features_train, pancount_train)

#predict pan counts using regression tree
pancount_pred = regressor.predict(features_test)



#regression tree diagnostics
print('Mean Absolute Error:', metrics.mean_absolute_error(pancount_test, pancount_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(pancount_test, pancount_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(pancount_test, pancount_pred)))
print('R-squared:',metrics.r2_score(pancount_test, pancount_pred))

#plot sample of first 25 predictions vs actuals
pancount_test1 = pancount_test['panconsumption'].values.reshape(-1,1)

consumresults = pd.DataFrame({'Actual': pancount_test1.flatten(), 'Predicted': pancount_pred.flatten()})


consumresults1 = consumresults.head(25)
consumresults1.plot(kind='bar',figsize=(16,10))
plt.title('Decision Tree Regression')
plt.xlabel('Service Records')
plt.ylabel('Pan Count')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#Regression Tree Residual Analysis

regtree_residuals = consumresults['Actual']- consumresults['Predicted'] 

regtree_residuals = pd.DataFrame(regtree_residuals, columns = ['Residuals'])

fig5 = px.histogram(regtree_residuals, x = 'Residuals')

plot(fig5)

treepos = regtree_residuals[regtree_residuals['Residuals']>=0]

treeneg = regtree_residuals[regtree_residuals['Residuals']<0]

print(treepos['Residuals'].mean())
print(treeneg['Residuals'].mean())

#regression tree feature importance
importance_regtree = regtree_model.feature_importances_
importregtree_df = pd.DataFrame(importance_regtree, index=features_train.columns, 
                      columns=["Importance"])

#visual output of regression tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(regtree_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())



#data prep for run-out models
#selected features
lincolumns2 = ['Service_Date','headcount','attendance',
              'pro1_pan_cnt'
              ,'pro2_pan_cnt'
              ,'veg_pan_cnt'
              ,'starch_pan_cnt',
              'side_pan_cnt','side2_pan_cnt'
              ,'salad_pan_cnt','pro1_avail_bool','pro2_avail_bool','X','Y','panconsumption','city']



lintest1 = linregtest[lincolumns2].copy()

#drops all observations without a protein served
lintest1= lintest1.dropna(subset = ['pro1_avail_bool'])

#recodes protein and alt protein run outs into one binary variable
#that represents run out of all protein/ not just main or alt
lintest1[['pro2_avail_bool']]=lintest1['pro2_avail_bool'].fillna('Yes,brought enough food')

lintest1.loc[lintest1['pro1_avail_bool'].str.contains('Yes'),'pro1_out'] = 1
lintest1.loc[lintest1['pro1_avail_bool'].str.contains('ran out'),'pro1_out'] = 0

lintest1.loc[lintest1['pro2_avail_bool'].str.contains('Yes'),'pro2_out'] = 1
lintest1.loc[lintest1['pro2_avail_bool'].str.contains('ran out'),'pro2_out'] = 0

lintest1.loc[lintest1['pro1_out']+lintest1['pro2_out']==2,'run_out'] = 1
lintest1.loc[lintest1['pro1_out']+lintest1['pro2_out']<2,'run_out'] = 0

#retrieves day of week from service date variable
lintest1['Service_Date'] = pd.to_datetime(lintest1['Service_Date'],format="%m/%d/%Y")


lintest1['weekday'] = lintest1['Service_Date'].dt.day_name()

#dummy coding procedure for city and weekday variables
lintest1 = pd.get_dummies(lintest1,columns=['city','weekday'])

#replaces missing values of selected interval variables with 0
lintest1.isnull().sum(axis = 0)

lintest1[['pro1_pan_cnt','pro2_pan_cnt','veg_pan_cnt','starch_pan_cnt','side_pan_cnt','side2_pan_cnt','salad_pan_cnt']]=lintest1[['pro1_pan_cnt','pro2_pan_cnt','veg_pan_cnt','starch_pan_cnt','side_pan_cnt','side2_pan_cnt','salad_pan_cnt']].fillna(0)

lintest1.isnull().sum(axis = 0)

#screens out values too large or too small for pan consumption
lintest1 = lintest1[lintest1['panconsumption']>0]

lintest1 = lintest1[lintest1['panconsumption']<12]

lintest1= lintest1.dropna()



#selected features

features = lintest1[['headcount','pro1_pan_cnt','pro2_pan_cnt','veg_pan_cnt','starch_pan_cnt','side_pan_cnt','side2_pan_cnt','salad_pan_cnt','X','Y','city_ATL','city_ATX','city_BNA','city_DEN','city_PHX','city_SEA','weekday_Monday','weekday_Tuesday','weekday_Wednesday','weekday_Thursday','weekday_Friday']]

runouts = lintest1[['run_out']]

from sklearn.model_selection import train_test_split
from sklearn import tree

#dataset is split into training and testing sets
features_train, features_test, runouts_train, runouts_test = train_test_split(features, runouts, test_size=0.3, random_state=0)

#decision tree is fit to training data
regressor = tree.DecisionTreeClassifier()
model=regressor.fit(features_train, runouts_train)

runouts_pred = regressor.predict(features_test)

#feature importance list is created
importance = model.feature_importances_
importance_df = pd.DataFrame(importance, index=features_train.columns, 
                      columns=["Importance"])

from sklearn import plot_confusion_matrix

#confusion matrix is plotted for assessment
cm = metrics.confusion_matrix(runouts_test, runouts_pred)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g', cmap='Blues'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
ax.set_title('Decision Tree Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Run Out', 'No Run Out']); ax.yaxis.set_ticklabels(['Run Out', 'No Run Out']);

#output for model assessment is produced
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

accuracy_score(runouts_test, runouts_pred)
precision_recall_fscore_support(runouts_test, runouts_pred)
print(classification_report(runouts_test, runouts_pred, target_names=['Run-Out','No Run-Out']))


#visual of decision tree is produced and saved to png file
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())



#logistic regression model for predicting runout
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logisticRegr = LogisticRegression()
logisticRegr.fit(features_train, runouts_train)

runouts_pred = logisticRegr.predict(features_test)

#output for model assessment is produced
accuracy_score(runouts_test, runouts_pred)
precision_recall_fscore_support(runouts_test, runouts_pred)
print(classification_report(runouts_test, runouts_pred, target_names=['Run-Out','No Run-Out']))


#confusion matrix output
cm = metrics.confusion_matrix(runouts_test, runouts_pred)
print(cm)


#Confusion Matrix plot
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g', cmap='Blues'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
ax.set_title('Logistic Regression Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Run Out', 'No Run Out']); ax.yaxis.set_ticklabels(['Run Out', 'No Run Out']);





#additional model tested, random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(features_train, runouts_train)
print(clf.feature_importances_)
runouts_pred = clf.predict(features_test)
cm = metrics.confusion_matrix(runouts_test, runouts_pred)
print(cm)

accuracy_score(runouts_test, runouts_pred)





