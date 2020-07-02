#####################################################
## FOr this work, no hold outs are created. We simply want to directly compare to 
## what was done for CALF.
## We know already, holdouts will not be well predicted for any of these models.
## 

import time as tm
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed

#import scipy
from scipy.stats import reciprocal, uniform
######################################################
#
def generateHeaderNaming(columns):
    newnames = list()
    for i in columns:
        if (i=='ctrl/case' or i=='group'):
            newnames.append('PHENO')
        else:
            newnames.append(str(i))
    return (newnames)

def updateHeaderData(df):
    columns = df.columns
    newcolumns = generateHeaderNaming(columns)
    df.columns=newcolumns
    return(df)

def generateIndexNaming(indices):
    newnames = list()
    for i in indices:
        newnames.append('subject_'+str(i))
    return (newnames)

def updateIndexData(df):
    indices = df.index
    newindices = generateIndexNaming(indices)
    df.index=newindices
    return(df)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

#def plot_roc_curve(fpr, tpr, label=None):
#    # plot the curve using default line style and color
#    plt.plot(fpr, tpr, linewidth=2, label=label)
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.axis([0, 1, 0, 1])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')

def top_five(dfin,header):
    """ Zero out all terms except the top five where top
    values are always
    checked as their absolute value
    Expects only a SINGLE column. Indexing is retained
    """
    #dfout = dfin.reindex(dfin.LogReg.abs().sort_values().index)
    dfout = dfin.reindex(dfin.loc[:,header].abs().sort_values(ascending=False).index)
    dfout.iloc[5:,]=0.0 # Or perhaps use an NA?
    return (dfout.reindex(dfin.index))

def plot_roc_curve(fpr, tpr, title=None):
    # plot the curve using default line style and color
    plt.plot(fpr, tpr, linewidth=2, color='green')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def plot_pvals(pvals, title='notitle'):
    n, bins, patches = plt.hist(pvals,20, normed=1, facecolor='blue', alpha=0.3)

def generate_roc_plot(model, x_train, y_train, title='test', method='decision_function'):
    y_scores = cross_val_predict(model,x_train,y_train,cv=3,method=method)
    if method=='predict_proba':
        y_scores = y_scores[:,1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)
    plt.figure(figsize=(14, 7))
    plt.clf()
    plot_roc_curve(false_positive_rate, true_positive_rate,title=title)
    #plt.show()    
    r_a_score = roc_auc_score(y_train, y_scores)
    print('AUC score is '+str(r_a_score))
    return r_a_score

#########################################################
# Take a look at the data 

# According to Clark these data have already been normalized. 
infilename = 'BLOOD_72X135.csv'

inData_read = pd.read_csv(infilename)
startData = updateIndexData(inData_read)
startData = updateHeaderData(inData_read)

# Split the PHENO (Y) and FEATURE (X) data sets 
Y_raw = startData['NCvC']
X_unscaled = startData.drop(['NCvC'],axis=1)

from sklearn.preprocessing import LabelEncoder
CC = LabelEncoder()
Y_factor = CC.fit_transform(Y_raw)

################################################################################
# The features have been pre standardized and z-scored no need to do it again

from sklearn.model_selection import train_test_split
x_train = X_unscaled
y_train = Y_factor

###############################################################################
# feature correlations

corr = x_train.corr()
ax = sns.heatmap(
    corr,
    vmin=0, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    horizontalalignment='right')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right')
plt.tight_layout()
plt.savefig('NCvCBlood_all_correlation.pdf')

###############################################################################
# Are the feature normal? 
#x_train.hist()
#plt.tight_layout()
#plt.show()
#plt.savefig('NCvCBlood_histogram.pdf')

###############################################################################
##
## Start simply processing different ML approaches in-order
##

# 1: Stochastic Gradient Descent (SGD): Elastic net

param_grid = [{"l1_ratio": [0.,.1,.2,.3,.4,.5,.6,.7,.8,.9, 1.0]}]
grd_search_cv = GridSearchCV(SGDClassifier(verbose=True, max_iter=1000, tol=0.0001,
    penalty='elasticnet', learning_rate='optimal', validation_fraction=0.2, random_state=42), param_grid, verbose=False, cv=10)
grd_search_cv.fit(x_train,y_train)
print('params')
print(grd_search_cv.best_params_)
print(grd_search_cv.best_score_)
grd_search_cv.best_estimator_.fit(x_train, y_train)
y_pred_train = grd_search_cv.best_estimator_.predict(x_train)
accuracy_score(y_train, y_pred_train)

print('SGD best params {}'.format(grd_search_cv.best_params_))
l1ratio = grd_search_cv.best_params_['l1_ratio']
sgd = SGDClassifier(verbose=True, l1_ratio=l1ratio, validation_fraction=0.2,loss='hinge',random_state=42, max_iter=1000, tol=0.0001)
sgd.fit(x_train, y_train)
#y_pred = sgd.predict(X_test)
df_coef_sgd = pd.DataFrame(sgd.coef_.T)
df_coef_sgd.index = x_train.columns
df_coef_sgd.columns=['SGD']
df_coef_sgd = top_five(df_coef_sgd,df_coef_sgd.columns[0])
sgd_coef = sgd.coef_
acc_sgd = round(np.mean(cross_val_score(sgd, x_train, y_train, cv=10, scoring="accuracy"))*100, 2)

# Check permutation status
SGD = SGDClassifier(verbose=True, max_iter=1000, tol=0.0001,
    penalty='elasticnet', learning_rate='optimal', validation_fraction=0.2, loss='hinge',l1_ratio=l1ratio,random_state=42)
perm_cv = permutation_test_score(SGD,x_train, y_train, cv=10, n_permutations=3, n_jobs=8, verbose=0)
score = perm_cv[0]
scorelist = perm_cv[1]
pval = perm_cv[2]
print(score) # If using CV, this is the MEAN of the Kfold scores
print(pval)
dict_coef_sgd_pval={'SGD':pval}

# Plot the AUCROC curve and get integral
sgd_auc = generate_roc_plot(sgd, x_train, y_train,title='ROC-AUC: SGD')
plt.savefig('NCvCBlood_all_SGD_ROC.pdf')
print('Done with SGD')

#################################################################################################
# 2: Random forest

param_distributions = {"n_estimators": range(10, 30)}
rnd_search_cv = RandomizedSearchCV(RandomForestClassifier(random_state=42),param_distributions,n_iter=5000, verbose=0, cv=10)
rnd_search_cv.fit(x_train, y_train)
print(rnd_search_cv.best_params_)
print('RFN best params {}'.format(rnd_search_cv.best_params_))

inn_estimators=rnd_search_cv.best_params_['n_estimators']
rfn = RandomForestClassifier(n_estimators=inn_estimators,random_state=42).fit(x_train, y_train)
df_coef_rfn = pd.DataFrame(rfn.feature_importances_.T)
df_coef_rfn.index = x_train.columns
df_coef_rfn.columns=['RNFT']
df_coef_rfn = top_five(df_coef_rfn,df_coef_rfn.columns[0])

RND=RandomForestClassifier(n_estimators=inn_estimators)
perm_cv = permutation_test_score(RND,x_train, y_train, cv=10, n_permutations=3, n_jobs=8, verbose=0)
score = perm_cv[0]
pval = perm_cv[2]
perm_scores = perm_cv[1]
print(score) # If using CV, this is the MEAN of the Kfold scores
print(pval)
dict_coef_rfn_pval={'RNFT':pval}
acc_rfn = round(np.mean(cross_val_score(RND, x_train, y_train, cv=10, scoring="accuracy"))*100, 2)
rfn_auc = generate_roc_plot(rfn, x_train, y_train, title='ROC-AUC: Random Forest', method='predict_proba')
plt.savefig('NCvCBlood_all_RFN_ROC.pdf')
print('Done with RFN')

######################################################################################################
# 3: SVM Linear : Should be about the same as the SGD

param_distributions = {"gamma": reciprocal(0.0001, 0.1), "C": uniform(1, 10)}
rnd_search_linear_cv = RandomizedSearchCV(SVC(kernel='linear'), param_distributions, random_state=42,n_iter=5000, verbose=0, cv=10)
rnd_search_linear_cv.fit(x_train,y_train)
print(rnd_search_linear_cv.best_params_)
print('SVM best params {}'.format(rnd_search_cv.best_params_))
print(rnd_search_linear_cv.best_score_)
rnd_search_linear_cv.best_estimator_.fit(x_train, y_train)
y_pred_train = rnd_search_linear_cv.best_estimator_.predict(x_train)
accuracy_score(y_train, y_pred_train)

# Get the coefficients
bestGamma = rnd_search_linear_cv.best_params_['gamma']
bestC = rnd_search_linear_cv.best_params_['C']
svc_linear_clf = SVC(verbose=True, gamma=bestGamma, kernel='linear',C=bestC,random_state=42, max_iter=1000, tol=0.0001)
svc_linear_clf.fit(x_train, y_train)
df_coef_svc_linear = pd.DataFrame(svc_linear_clf.coef_.T)
df_coef_svc_linear.index = x_train.columns
df_coef_svc_linear.columns=['SVC']
df_coef_svc_linear = top_five(df_coef_svc_linear,df_coef_svc_linear.columns[0])
acc_svm = round(np.mean(cross_val_score(svc_linear_clf, x_train, y_train, verbose=0, cv=10, scoring="accuracy"))*100, 2)

# Check permutation status
SVC = SVC(verbose=True, gamma=bestGamma, kernel='linear',C=bestC,max_iter=1000)
perm_cv = permutation_test_score(SVC,x_train, y_train, random_state=42, cv=10, n_permutations=3, n_jobs=8, verbose=0)
score = perm_cv[0]
pval = perm_cv[2]
perm_scores = perm_cv[1]
print(score) # If using CV, this is the MEAN of the Kfold scores
print(pval)
dict_coef_svc_linear_pval={'SVC':pval}

svc_linear_auc = generate_roc_plot(svc_linear_clf, x_train, y_train, title='ROC-AUC: SVM')
plt.savefig('NCvCBlood_all_SVC_LINEAR_ROC.pdf')
print('Done with SVM')

###################################################################################################
# 4: Logistic Regression: elastic net. grid search alpha

param_grid = [{"l1_ratio": [.1,.2,.3,.4,.5,.6,.7,.8,.9, 1.0],
               "C": [0.1,0.2,0.3,0.4,0.5,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]}]
LRG = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=1000,
                   n_jobs=None, penalty='elasticnet',
                   random_state=42, solver='saga', tol=0.001, verbose=0,
                   warm_start=False)

grd_search_cv = GridSearchCV(LRG,param_grid, verbose=True, cv=10)
grd_search_cv.fit(x_train,y_train)
print('LRG best params {}'.format(grd_search_cv.best_params_))
print(grd_search_cv.best_score_)

l1ratio=grd_search_cv.best_params_['l1_ratio']
newC = grd_search_cv.best_params_['C']
# Fit coefs using the best combination of l1/l2
reg = LogisticRegression(C=newC, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=1000, l1_ratio=l1ratio,
                   multi_class='auto', n_jobs=None, penalty='elasticnet',
                   random_state=42, solver='saga', tol=0.001, verbose=0,
                   warm_start=False).fit(x_train,y_train)
df_coef_log = pd.DataFrame(reg.coef_.T)
df_coef_log.index = x_train.columns
df_coef_log.columns=['LogReg']
df_coef_log = top_five(df_coef_log,df_coef_log.columns[0])

# Permutation testing to check significance
LRG = LogisticRegression(C=newC, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=1000, l1_ratio=l1ratio,
                   multi_class='auto', n_jobs=None, penalty='elasticnet',
                   random_state=42, solver='saga', tol=0.001, verbose=0,
                   warm_start=False)

perm_cv = permutation_test_score(LRG,x_train, y_train, cv=10, n_permutations=3, n_jobs=8, verbose=0)
score = perm_cv[0]
pval = perm_cv[2]
perm_scores = perm_cv[1]
print(score) # If using CV, this is the MEAN of the Kfold scores
print(pval)
acc_log = round(np.mean(cross_val_score(LRG, x_train, y_train, cv=10, scoring="accuracy"))*100, 2)

# Add pval to the df.
dict_coef_log_pval={'LogReg':pval}
log_auc = generate_roc_plot(reg, x_train, y_train,title='ROC-AUC: Logistic Regression Elestic net')
plt.savefig('NCvCBlood_all_LRG_ROC.pdf')
print('Done with LGR')

###############################################################################
# 5: LogisticRegression-Lasso only

reg_l1 = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=None, solver='saga', tol=0.001, verbose=0,
                   warm_start=False).fit(x_train,y_train)

df_coef_l1 = pd.DataFrame(reg_l1.coef_.T)
df_coef_l1.index = x_train.columns
df_coef_l1.columns=['LogRegL1']
df_coef_l1 = top_five(df_coef_l1,df_coef_l1.columns[0])

#reg_l1.coef_
y_pred_train = reg_l1.predict(x_train)
accuracy_score(y_train, y_pred_train)

acc_logl1 = round(np.mean(cross_val_score(reg_l1, x_train, y_train, cv=10, scoring="accuracy"))*100, 2)

# Permutation testing to check significance
LRG_l1 = reg_l1
perm_cv = permutation_test_score(LRG_l1,x_train, y_train, cv=10, n_permutations=3, n_jobs=8, verbose=0)
score = perm_cv[0]
pval = perm_cv[2]
perm_scores = perm_cv[1]
print(score) # If using CV, this is the MEAN of the Kfold scores
print(pval)

dict_coef_l1_pval={'LogRegL1':pval}
log_l1_auc = generate_roc_plot(reg_l1, x_train, y_train,title='ROC-AUC: Logistic Regression Lasso')
plt.savefig('NCvCBlood_all_LRG_l1_ROC.pdf')

print('Done with LGR-l1')

########################################################################################
# 6: K Nearest Neighbor:
 
param_grid = [{"n_neighbors": [5,10,15,20,25,30,35,40], "leaf_size": [10,20,30,40,50,60,70]}]
grd_search_cv = GridSearchCV(KNeighborsClassifier(),param_grid, verbose=True, cv=10)
grd_search_cv.fit(x_train,y_train)
grd_search_cv.best_params_
grd_search_cv.best_score_
print('KNN best params {}'.format(grd_search_cv.best_params_))
print(grd_search_cv.best_score_)

grd_search_cv.best_estimator_.fit(x_train, y_train)
y_pred_train = grd_search_cv.best_estimator_.predict(x_train)
accuracy_score(y_train, y_pred_train)

bestneighbors = grd_search_cv.best_params_['n_neighbors']
inleaf_size = grd_search_cv.best_params_['leaf_size']

knn_clf = KNeighborsClassifier(n_neighbors=bestneighbors)
#df_coef_knn = pd.DataFrame(knn_clf.fit(x_train,y_train).coef_.T)

acc_knn = round(np.mean(cross_val_score(knn_clf, x_train, y_train, cv=10, scoring="accuracy"))*100, 2)

KNN = KNeighborsClassifier(n_neighbors=bestneighbors)
perm_cv = permutation_test_score(KNN,x_train, y_train, random_state=42, cv=10, n_permutations=3, n_jobs=8, verbose=0)
score = perm_cv[0]
pval = perm_cv[2]
perm_scores = perm_cv[1]
print(score) # If using CV, this is the MEAN of the Kfold scores print(pval)
dict_coef_knn_pval={'KNN':pval}

knn_auc = generate_roc_plot(knn_clf, x_train, y_train,method='predict_proba',title='ROC-AUC: KNN')
plt.savefig('NCvCBlood_all_KNN_ROC.pdf')
print('Done with KNN')

################################################################################
# 7: Gaussian Naive Bayes:

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
acc_gaussian = round(np.mean(cross_val_score(gaussian, x_train, y_train, cv=10, scoring="accuracy"))*100, 2)

# No coefficients

# Check permutation status
GAS = GaussianNB()
perm_cv = permutation_test_score(GAS,x_train, y_train, random_state=42, cv=10, n_permutations=3, n_jobs=8, verbose=0)
score = perm_cv[0]
pval = perm_cv[2]
perm_scores = perm_cv[1]
print(score) # If using CV, this is the MEAN of the Kfold scores
print(pval)
dict_coef_gaussian_pval={'GAUSS':pval}

gaussian_auc = generate_roc_plot(gaussian, x_train, y_train, method='predict_proba',title='ROC-AUC: GaussianNB')
plt.savefig('NCvCBlood_all_GAUSSIAN_ROC.pdf')

print('Done with GB')

###############################################################################
# 8: No grid searching: Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
#Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(np.mean(cross_val_score(decision_tree, x_train, y_train, cv=10, scoring="accuracy"))*100, 2)

df_coef_dt = pd.DataFrame(decision_tree.feature_importances_.T)
df_coef_dt.index = x_train.columns
df_coef_dt.columns=['DTREE']
df_coef_dt = top_five(df_coef_dt,df_coef_dt.columns[0])
#
DT = DecisionTreeClassifier()
perm_cv = permutation_test_score(DT,x_train, y_train, cv=10, n_permutations=3, n_jobs=8, verbose=0)
score = perm_cv[0]
pval = perm_cv[2]
perm_scores = perm_cv[1]
print(score) # If using CV, this is the MEAN of the Kfold scores
print(pval)
dict_coef_dt_pval={'DTREE':pval}
dt_auc = generate_roc_plot(decision_tree, x_train, y_train,method='predict_proba', title='ROC-AUC: Decision Tree')
plt.savefig('NCvCBlood_all_DT_ROC.pdf')
print('Done with DT')

print('Output ML model results')

results = pd.DataFrame({
    'Model': ['SGD', 'Random Forest', 'Support Vector Machines', 'Logistic Regression',
              'LogisticRegression-L1', 'KNN', 'Gaussian', 'Decision Tree'],
    'Score': [acc_sgd, acc_rfn, acc_svm, acc_log,
              acc_logl1, acc_knn, acc_gaussian, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))

results_auc = pd.DataFrame({
    'Model': ['SGD', 'Random Forest', 'Support Vector Machines', 'Logistic Regression',
              'LogisticRegression-L1', 'KNN', 'Gaussian', 'Decision Tree'],
    'AUC': [sgd_auc, rfn_auc, svc_linear_auc, log_auc,
              log_l1_auc, knn_auc, gaussian_auc, dt_auc]})

result_auc_df = results_auc.sort_values(by='AUC', ascending=False)
result_auc_df = result_auc_df.set_index('AUC')
print(result_auc_df.head(8))

###################################################################################
# Combine avail coefs and the pvals and store to disk

df_final_coef = df_coef_sgd.copy()
df_final_coef = pd.merge(df_final_coef,df_coef_rfn,left_index=True,right_index=True)
df_final_coef = pd.merge(df_final_coef,df_coef_svc_linear,left_index=True,right_index=True)
df_final_coef = pd.merge(df_final_coef,df_coef_log,left_index=True,right_index=True)
df_final_coef = pd.merge(df_final_coef,df_coef_l1,left_index=True,right_index=True)
df_final_coef = pd.merge(df_final_coef,df_coef_dt,left_index=True,right_index=True)
df_final_coef.to_csv('NCvCBlood_alltraining_coefs.tsv',sep=' ')

# Permutation Pvalues

final_pvalues = {**dict_coef_sgd_pval, **dict_coef_rfn_pval}
final_pvalues = {**final_pvalues, **dict_coef_svc_linear_pval}
final_pvalues = {**final_pvalues, **dict_coef_log_pval}
final_pvalues = {**final_pvalues, **dict_coef_l1_pval}
final_pvalues = {**final_pvalues, **dict_coef_knn_pval}
final_pvalues = {**final_pvalues, **dict_coef_gaussian_pval}
final_pvalues = {**final_pvalues, **dict_coef_dt_pval}
df_pval = pd.DataFrame.from_dict(final_pvalues,orient='index')
df_pval.columns=['PERMPVAL']
print(df_pval)
df_pval.to_csv('NCvCBloodalltraining_pvals.tsv',sep=' ')

###################################################################################
# Hard voting Consensus across all methods
# This will rerun the above models and so is inefficient.

print('Start voting clasifier')

from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
estimators=[ ('sgd', sgd), ('randomforest',rfn), ('svc', svc_linear_clf), ('logreg',LRG), ('logreg_l1',LRG_l1), ('knn', knn_clf),
              ('gauss', gaussian),
              ('dt', decision_tree) ], voting = 'hard')
voting_clf.fit(x_train, y_train)
print('{}'.format(voting_clf.estimators_))
listScores = list()
for clf in (sgd,rfn, svc_linear_clf, LRG, LRG_l1, knn_clf, gaussian, decision_tree, voting_clf):
    listScores.append( (clf, round(np.mean(cross_val_score(clf, x_train, y_train, cv=10, scoring="accuracy"))*100,2)))
#
print('Output model results and the consensus value')

for data in listScores:
    print(data[0].__class__.__name__,data[1])
