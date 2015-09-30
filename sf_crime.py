import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from dateutil.parser import parse
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import BaggingClassifier

data = pd.read_csv("/Users/janghoo/Documents/kaggle project/criminal classification/data/train.csv")
data2 = pd.read_csv("/Users/janghoo/Documents/kaggle project/criminal classification/data/test.csv")

def calculate(x):
    return parse(x).year,parse(x).month,parse(x).day,parse(x).hour

def get_v(x):
    d = [0]*len(x)
    l = list()
    for i in range(0,len(x)):
        d[i] += 1
        l.append(d)
        d = [0]*len(x)
    return l

data['year'],data['month'],data['day'],data['hour'] = zip(*data.Dates.map(calculate))
data2['year'],data2['month'],data2['day'],data2['hour'] = zip(*data2.Dates.map(calculate))

dw = list(set(data.DayOfWeek))
dist = list(set(data.PdDistrict))

dw_v = get_v(dw)
dw_df = pd.DataFrame({'DayOfWeek' : dw,'dow' : dw_v})
dist_v = get_v(dist)
dist_df = pd.DataFrame({'PdDistrict' : dist,'dis' : dist_v})
data_1 = pd.merge(data,dw_df,on = 'DayOfWeek',how = 'left')
data_2 = pd.merge(data_1,dist_df,on = 'PdDistrict',how = 'left')
data2_1 = pd.merge(data2,dw_df,on = 'DayOfWeek',how = 'left')
data2_2 = pd.merge(data2_1,dist_df,on = 'PdDistrict',how = 'left')

y_train = list(data_2.Category)
x1_train = list(zip(data_2.X,data_2.Y,data_2.year,data_2.month,data_2.day,data_2.hour))
x2_train = [0]*len(x1_train)
for i in range(0,len(x1_train)):
    x2_train[i]= list(x1_train[i])+data_2['dow'][i]+data_2['dis'][i]

clf1 = RandomForestClassifier(n_estimators=50)
clf2 = AdaBoostClassifier(n_estimators=50)
clf3 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
clf4 = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.5, kernel='poly',degree = 2, max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False )  #svm
clf5 = linear_model.LogisticRegression()#logit
clf4 = BaggingClassifier(clf4,max_samples=0.7, max_features=1.0,n_estimators=20)
clf5 = BaggingClassifier(clf5,max_samples=0.7, max_features=1.0,n_estimators=20)


clf1 = clf1.fit(x2_train, y_train)
clf2 = clf2.fit(x2_train, y_train)
clf3 = clf3.fit(x2_train, y_train)
clf4 = clf4.fit(x2_train, y_train)
clf5 = clf5.fit(x2_train, y_train)

x1_test = list(zip(data2_2.X,data2_2.Y,data2_2.year,data2_2.month,data2_2.day,data2_2.hour))
x2_test = [0]*len(x1_test)
for i in range(0,len(x1_test)):
    x2_test[i]= list(x1_test[i])+data2_2['dow'][i]+data2_2['dis'][i]


r1 = clf1.predict_proba(x2_test)
r2 = clf2.predict_proba(x2_test)
r3 = clf3.predict_proba(x2_test)
r4 = clf4.predict_proba(x2_test)
r5 = clf5.predict_proba(x2_test)



