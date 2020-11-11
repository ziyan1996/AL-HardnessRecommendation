#generate recommendations for a pool of unknown compounds

from dscribe.descriptors import SOAP
from ase.io import read
from ase import Atoms
from ase.spacegroup import get_space_group
import torch as th
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
import pandas as pd
import numpy as np
import pymatgen as mg
import os
import argparse

parser = argparse.ArgumentParser(description='Giving recommendations based on predicted hardness and thermal stability (293.15 K to 1073.15 K)')
parser.add_argument('--pool_path', required=True, help="Path to the directory of cif files")
#parser.add_argument('--composition', required=True, help="List of chemical formulas stored in spread sheet")
#parser.add_argument('--temp', required=True, help="At which temperature (K)")
#parser.add_argument('--load', required=True, help="At which applied temperature (N)")
args = parser.parse_args()

#1. train the model

path = '/Users/ziyanzhang/Downloads/dscribe/examples/hv_temp_cif/'
dir = os.listdir(path)
structure_files = []
for file in dir:
    structure_files.append(file)

os.chdir(path)
atoms = [None]*591
for i in range(591):
    atoms[i] = read(str(i) + ".cif")
print('got structures from cif files')

df = pd.read_excel('/Users/ziyanzhang/Downloads/dscribe/examples/hv_temp_cif_labels.xlsx')
y = df['hardness'] #lower case y is the target hardness
temp = df['temp']
load = df['load']
print('got targets from spread sheet')

species = set()
for i in range(len(atoms)):
    species.update(atoms[i].get_chemical_symbols())

soap = SOAP(species=species, periodic=True, rcut=5, nmax=1, lmax=1, average="outer")
print('generating SOAP descriptors...')
soap.get_number_of_features()

feature_vectors = soap.create(atoms, n_jobs=1)
feature_tensor = th.tensor(feature_vectors)
print('DONE, SOAP descriptors ready to use')
feature_pd = pd.DataFrame(feature_vectors) #feature pd is the soap descriptors

#generate compositional descriptors
df = pd.read_excel('/Users/ziyanzhang/Downloads/dscribe/examples/hv_temp_cif_labels.xlsx')
class Vectorize_Formula:

    def __init__(self):
        elem_dict = pd.read_excel('/Users/ziyanzhang/Desktop/subgroup/elementsnew.xlsx') # CHECK NAME OF FILE
        self.element_df = pd.DataFrame(elem_dict)
        self.element_df.set_index('Symbol',inplace=True)
        self.column_names = []
        for string in ['avg','diff','max','min']:
            for column_name in list(self.element_df.columns.values):
                self.column_names.append(string+'_'+column_name)

    def get_features(self, formula):
        try:
            fractional_composition = mg.Composition(formula).fractional_composition.as_dict()
            element_composition = mg.Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            std_feature = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += self.element_df.loc[key].values * fractional_composition[key]
                    diff_feature = self.element_df.loc[list(fractional_composition.keys())].max()-self.element_df.loc[list(fractional_composition.keys())].min()
                except Exception as e:
                    print('The element:', key, 'from formula', formula,'is not currently supported in our database')
                    return np.array([np.nan]*len(self.element_df.iloc[0])*4)
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            std_feature=self.element_df.loc[list(fractional_composition.keys())].std(ddof=0)

            features = pd.DataFrame(np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature)]))
            features = np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature)])
            return features.transpose()
        except:
            print('There was an error with the Formula: '+ formula + ', this is a general exception with an unkown error')
            return [np.nan]*len(self.element_df.iloc[0])*4
gf=Vectorize_Formula()

# empty list for storage of features
features=[]

# add values to list using for loop
for formula in df['composition']:
    features.append(gf.get_features(formula))

# feature vectors and targets as X and y
X = pd.DataFrame(features, columns = gf.column_names)
pd.set_option('display.max_columns', None)
# allows for the export of data to excel file
header=gf.column_names
header.insert(0,"Composition")

composition=df['composition']
#composition=pd.read_excel('pred_hv_comp.xlsx',sheet_name='Sheet1', usecols="A")
composition=pd.DataFrame(composition)

predicted=np.column_stack((composition,X))
predicted=pd.DataFrame(predicted) #predicted(dataframe) is the compositional features

array = predicted.values
X = array[:,1:141]
Xpd = pd.DataFrame(X)
Xpd['temp'] = temp
Xpd['load'] = load
scaler = preprocessing.StandardScaler().fit(Xpd)
Xpd = scaler.transform(Xpd)
Xpd = pd.DataFrame(Xpd)
combine_feature_pd = pd.concat([Xpd, feature_pd], axis=1, sort=False) #this is soap+compositional+temp+load

array = combine_feature_pd.values
X = array[:,1:4049]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99, test_size=0.01,random_state=100, shuffle=True)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree =0.7, learning_rate = 0.18,
                max_depth = 4, alpha = 5, n_estimators = 500, subsample=0.6)
xgb_model=xg_reg.fit(X_train, y_train)
preds=xgb_model.predict(X_test)
r2=r2_score(preds, y_test)
mae=mean_absolute_error(preds, y_test)
#print('R2 score on test set: ' + str(r2))
#print('MAE on test set: ' + str(mae))
print('Model ready to use')

# 2. now looking at the MP pool, read cif in the directory recursively,
# give prediciton of hardness at RT 293.15K (with uncertainty) and 1073.15K(with uncertainty),
# give percentage loss as a measure of thermal stability

print('Getting structures from pool directory...')

pool_dir = os.listdir(args.pool_path)
pool_structure_files = []
for file in pool_dir:
    pool_structure_files.append(file)

os.chdir(pool_path)
pool_atoms = [None]*13172
pool_compositions = [None]*13172
pool_spacegroups = [None]*13172
for i in range(13172):
    pool_atoms[i] = read(str(i) + ".cif")
    pool_compositions[i] = read(str(i) + ".cif").get_chemical_formula()
    pool_spacegroups[i] = get_space_group(read(str(i) + ".cif")).no

print('Making predictions...')

pool_feature_vectors = soap.create(pool_atoms, n_jobs=1)

pool_feature_pd = pd.DataFrame(pool_feature_vectors) #feature pd is the soap descriptors

#generate compositional descriptors
#df = pd.read_excel('/Users/ziyanzhang/Downloads/dscribe/examples/hv_temp_cif_labels.xlsx')
pool_df=pd.DataFrame(pool_compositions, columns='composition')

gf=Vectorize_Formula()

# empty list for storage of features
pool_features=[]

# add values to list using for loop
for formula in pool_df['composition']:
    pool_features.append(gf.get_features(formula))

# feature vectors and targets as X and y
pool_X = pd.DataFrame(pool_features, columns = gf.column_names)
pool_pd.set_option('display.max_columns', None)
# allows for the export of data to excel file
#header=gf.column_names
#header.insert(0,"Composition")

#composition=df['composition']
#composition=pd.read_excel('pred_hv_comp.xlsx',sheet_name='Sheet1', usecols="A")
#composition=pd.DataFrame(composition)

#predicted=np.column_stack((composition,X))
#predicted=pd.DataFrame(predicted) #predicted(dataframe) is the compositional features

pool_X['temp']=293.15
pool_X['load']=0.49

#scaler = preprocessing.StandardScaler().fit(predcomp_fea)
RT_pool_X=scaler.transform(pool_X)
RT_pool_X = pd.DataFrame(RT_pool_X)

RT_combine_feature_pred = pd.concat([RT_pool_X, pool_feature_pd], axis=1, sort=False)

RT_pred_array = RT_combine_feature_pred.values
RT_X_predict = RT_pred_array[:,1:4049]

# Query by Committee
SVM = SVR(C=100, gamma=0.01)
rf = RandomForestRegressor(max_depth=4, n_estimators=500)
gb = GradientBoostingRegressor(max_depth=4, learning_rate=0.1, n_estimators=500)
xgb = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree =0.7, learning_rate = 0.18, max_depth = 4, alpha = 5, n_estimators = 500, subsample=0.6)

committee=[rf, gb, xgb, SVM]

#choose which strategy to use
selection=input('Currently supported active learning strategy: \n1. Query by committee \n2. Entropy-based query by bagging \n3. Adaptive max disagreement \nWhich one you want to use (enter number 1, 2 or 3):')
if selection == 1:
    rt_predict_y = np.zeros((len(RT_X_predict), 4))
    i=0
    for estimator in committee:
        estimator.fit(X_train, y_train)
        rt_predict_y[:,i] = estimator.predict(RT_X_predict)
        i = i+1
    rt_pred_y = pd.DataFrame(rt_predict_y)
    rt_std = rt_pred_y.std(axis=1)
    rt_mean = rt_pred_y.mean(axis=1)
    rt_pred_y['std']=rt_std
    rt_pred_y['ypred']=rt_mean
    rt_pred_y['composition']=pool_compositions
    rt_pred_y['space_group']=pool_spacegroups
    rt_sorted_pred=rt_pred_y.sort_values(by='std', ascending=False)
    rt_tops = rt_sorted_pred.head(30)
    print_rt_tops = rt_tops[['composition', 'space_group', 'ypred', 'std']].copy()
    print('Query by committee \nRoom temperature (293.15K) recommendations: ')
    print(print_rt_tops)

    #generate high temperature predictions
    pool_X['temp']=1073.15
    HT_pool_X=scaler.transform(pool_X)
    HT_pool_X = pd.DataFrame(HT_pool_X)

    HT_combine_feature_pred = pd.concat([HT_pool_X, pool_feature_pd], axis=1, sort=False)

    HT_pred_array = HT_combine_feature_pred.values
    HT_X_predict = HT_pred_array[:,1:4049]

    ht_predict_y = np.zeros((len(HT_X_predict), 4))
    i=0
    for estimator in committee:
        estimator.fit(X_train, y_train)
        ht_predict_y[:,i] = estimator.predict(HT_X_predict)
        i = i+1
    ht_pred_y = pd.DataFrame(ht_predict_y)
    ht_std = ht_pred_y.std(axis=1)
    ht_mean = ht_pred_y.mean(axis=1)
    ht_pred_y['std']=ht_std
    ht_pred_y['ypred']=ht_mean
    ht_pred_y['composition']=pool_compositions
    ht_pred_y['space_group']=pool_spacegroups
    ht_sorted_pred=ht_pred_y.sort_values(by='std', ascending=False)
    ht_tops = ht_sorted_pred.head(30)
    print_ht_tops = ht_tops[['composition', 'space_group', 'ypred', 'std']].copy()
    print('Query by committee \nHigh temperature (1073.15K) recommendations: ')
    print(print_ht_tops)

elif selection == 2:
    n_repeat = 10
    estimator=BaggingRegressor(base_estimator=xgb, max_samples=300)
    rt_predict_y = np.zeros((len(RT_X_predict), n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train, y_train)
        rt_predict_y[:,i] = estimator.predict(RT_X_predict)
    rt_pred_y = pd.DataFrame(rt_predict_y)
    rt_std = rt_pred_y.std(axis=1)
    rt_mean = rt_pred_y.mean(axis=1)
    rt_pred_y['std']=rt_std
    rt_pred_y['ypred']=rt_mean
    rt_pred_y['composition']=pool_compositions
    rt_pred_y['space_group']=pool_spacegroups
    sorted_pred=rt_pred_y.sort_values(by='std', ascending=False)
    rt_tops = sorted_pred.head(30)
    print_rt_tops = rt_tops[['composition', 'space_group', 'ypred', 'std']].copy()
    print('Entropy-based query by bagging \nRoom temperature (293.15K) recommendations: ')
    print(print_rt_tops)

    #generate high temperature predictions
    pool_X['temp']=1073.15
    HT_pool_X=scaler.transform(pool_X)
    HT_pool_X = pd.DataFrame(HT_pool_X)

    HT_combine_feature_pred = pd.concat([HT_pool_X, pool_feature_pd], axis=1, sort=False)

    HT_pred_array = HT_combine_feature_pred.values
    HT_X_predict = HT_pred_array[:,1:4049]

    ht_predict_y = np.zeros((len(HT_X_predict), n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train, y_train)
        rt_predict_y[:,i] = estimator.predict(HT_X_predict)
    ht_pred_y = pd.DataFrame(ht_predict_y)
    ht_std = ht_pred_y.std(axis=1)
    ht_mean = ht_pred_y.mean(axis=1)
    ht_pred_y['std']=ht_std
    ht_pred_y['ypred']=ht_mean
    ht_pred_y['composition']=pool_compositions
    ht_pred_y['space_group']=pool_spacegroups
    ht_sorted_pred=ht_pred_y.sort_values(by='std', ascending=False)
    ht_tops = ht_sorted_pred.head(30)
    print_ht_tops = ht_tops[['composition', 'space_group', 'ypred', 'std']].copy()
    print('Entropy-based query by bagging \nHigh temperature (293.15K) recommendations: ')
    print(print_ht_tops)


else:
    n_repeat = 10
    estimator=BaggingRegressor(base_estimator=xgb, bootstrap_features=True, max_features=100)
    rt_predict_y = np.zeros((len(RT_X_predict), n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train, y_train)
        rt_predict_y[:,i] = estimator.predict(RT_X_predict)
    rt_pred_y = pd.DataFrame(rt_predict_y)
    rt_std = rt_pred_y.std(axis=1)
    rt_mean = rt_pred_y.mean(axis=1)
    rt_pred_y['std']=rt_std
    rt_pred_y['ypred']=rt_mean
    rt_pred_y['composition']=pool_compositions
    rt_pred_y['space_group']=pool_spacegroups
    sorted_pred=rt_pred_y.sort_values(by='std', ascending=False)
    rt_tops = sorted_pred.head(30)
    print_rt_tops = rt_tops[['composition', 'space_group', 'ypred', 'std']].copy()
    print('Adaptive max disagreement \nRoom temperature (293.15K) recommendations: ')
    print(print_rt_tops)

    #generate high temperature predictions
    pool_X['temp']=1073.15
    HT_pool_X=scaler.transform(pool_X)
    HT_pool_X = pd.DataFrame(HT_pool_X)

    HT_combine_feature_pred = pd.concat([HT_pool_X, pool_feature_pd], axis=1, sort=False)

    HT_pred_array = HT_combine_feature_pred.values
    HT_X_predict = HT_pred_array[:,1:4049]

    ht_predict_y = np.zeros((len(HT_X_predict), n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train, y_train)
        rt_predict_y[:,i] = estimator.predict(HT_X_predict)
    ht_pred_y = pd.DataFrame(ht_predict_y)
    ht_std = ht_pred_y.std(axis=1)
    ht_mean = ht_pred_y.mean(axis=1)
    ht_pred_y['std']=ht_std
    ht_pred_y['ypred']=ht_mean
    ht_pred_y['composition']=pool_compositions
    ht_pred_y['space_group']=pool_spacegroups
    ht_sorted_pred=ht_pred_y.sort_values(by='std', ascending=False)
    ht_tops = ht_sorted_pred.head(30)
    print_ht_tops = ht_tops[['composition', 'space_group', 'ypred', 'std']].copy()
    print('Adaptive max disagreement \nHigh temperature (1073.15K) recommendations: ')
    print(print_ht_tops)
