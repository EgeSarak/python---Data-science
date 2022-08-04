
#: Importing libraries
import sys
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#! ENV = 'prep'
#! ENV = 'dev'

ENV = 'prep'

#: Declare constants
path = 'data/'
target = 'Attrition'
#: Loading data
general = pd.read_csv(path + 'general_data.csv')

if ENV == 'dev':
    general = general.sample(frac = 0.10)


#: Eliminate
for c in general:
    if len(general[c].unique()) == 1:
        print(f"Deleting because there is only one {c}")
        del general[c]

del general['MaritalStatus'] # KVKK nedeniyle
#: Mapping
general[target] = general[target].map({'Yes': 1, 'No':0})
general['Gender'] = general['Gender'].map({'Female': 1, 'Male':0})
general['BusinessTravel'] = general['BusinessTravel'].map(
    {'Travel_Rarely': 0.10, 'Travel_Frequently': 0.50, 'Non-Travel': 0.0}
)
general = pd.get_dummies(data = general, columns = ['Department'])
general['EducationField2'] = general['EducationField']
#: Create both dummies and group by mean (target)
vals = dict( general.groupby( by = ['EducationField2'] )[target].mean() )
general['EducationField2'] = general['EducationField2'].map(vals)

general = pd.get_dummies(data = general, columns = ['EducationField'])

vals = dict( general.groupby( by = ['JobRole'] )[target].mean() )
general['JobRole'] = general['JobRole'].map(vals)

#: Fill empty values
#! general['TotalWorkingYears_isnull'] = general['TotalWorkingYears'].isnull()

general['TotalWorkingYears'] = general['TotalWorkingYears'].fillna( general['TotalWorkingYears'].mean() )
general['NumCompaniesWorked'] = general['NumCompaniesWorked'].fillna( general['NumCompaniesWorked'].mean() )

#! Feature transformation
general['MonthlyIncome'] = np.log(general['MonthlyIncome'])

general['DistanceFromHome'] = general['DistanceFromHome'] > 19

#: Pre-analyze
#! for c in general:
#!    print(c, general[c].corr(general[target]))

def findBestRange( df, target, column ):
    maxValue = 0
    maxItem = None
    for i in range(int(df[column].min()), int(df[column].max())):
        df['TEMP'] = df[column] > i
        v = df['TEMP'].corr(df[target])
        if abs(v) > abs(maxValue):
            maxValue = v
            maxItem = i
    return (maxValue, maxItem)

#!for c in general:
#!print( "!", c, general[c].corr(general[target]), findBestRange( general, target, c ) )

#: Merge 
employee_survey = pd.read_csv( path + 'employee_survey_data.csv')
general = pd.merge(general, employee_survey, how='inner', on = 'EmployeeID')
#: Replace NA
general['EnvironmentSatisfaction'] = general['EnvironmentSatisfaction'].replace( 'NA', general['EnvironmentSatisfaction'].mean() )
general['JobSatisfaction'] = general['JobSatisfaction'].replace( 'NA', general['JobSatisfaction'].mean() )
general['WorkLifeBalance'] = general['WorkLifeBalance'].replace( 'NA', general['WorkLifeBalance'].mean() )
#: Merge
general['JobInvolvement'] = general['JobInvolvement'].replace( 'NA', general['JobInvolvement'].mean() )
general['PerformanceRating'] = general['PerformanceRating'].replace( 'NA', general['PerformanceRating'].mean() )

#: Read the in_time csv file
in_time = pd.read_csv(path + 'in_time.csv')
for c in in_time:
    if c != 'EmployeeID':
        in_time[c] = pd.to_datetime( in_time[c] )
        in_time[c] = in_time[c].dt.time

def workingDays( lst: list ):
    wd = 0
    for l in lst:
        if not pd.isnull(l): 
            wd += 1
    return wd

def lateDays( lst: list ):
    ld = 0

    #tenthirty = datetime.time(10, 30, 00)

    for l in lst:
        if not pd.isnull(l): 
            #! if l > tenthirty
            if l.hour > 10 or (l.hour == 10 and  l.minute > 30 ):
                ld += 1
    return ld

#: Read the in_time csv file
out_time = pd.read_csv(path + 'out_time.csv')
for c in out_time:
    if c != 'EmployeeID':
        out_time[c] = pd.to_datetime( out_time[c] )
        out_time[c] = out_time[c].dt.time

WORKING_DAYS = {}
LATE_DAYS = {}
WORKING_HOURS = {}
WORKING_HOURS_TREND = {}



def workingHours( lst1, lst2 ):
    wh = []
    for i in range(len(lst1)):
        lin = lst1[i]
        lout = lst2[i]
        if not pd.isnull(lin): 
            duration = datetime.combine(date.min, lout) - datetime.combine(date.min, lin)
            duration = duration.seconds / 3600
            wh.append( duration )
    return np.mean(wh)



def linreg(X, Y):
	"""
	return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
	"""
	N = len(X)
	Sx = Sy = Sxx = Syy = Sxy = 0.0
	for x, y in zip(X, Y):
		Sx = Sx + x
		Sy = Sy + y
		Sxx = Sxx + x*x
		Syy = Syy + y*y
		Sxy = Sxy + x*y
	det = Sxx * N - Sx * Sx
	return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det


def workingHoursTrend( lst1, lst2 ):
    wh = []
    for i in range(len(lst1)):
        lin = lst1[i]
        lout = lst2[i]
        if not pd.isnull(lin): 
            duration = datetime.combine(date.min, lout) - datetime.combine(date.min, lin)
            duration = duration.seconds / 3600
            wh.append( duration )

    m, c = linreg(range(len(wh)), wh)
    return m


for i in range(len(in_time)):
    values = list(in_time.iloc[i].values)
    employee_id = values.pop(0)
    wd = workingDays( values )
    WORKING_DAYS[ employee_id ] = wd

    ld = lateDays( values )
    LATE_DAYS[ employee_id ] = ld


    values2 = list(out_time.iloc[i].values)
    values2.pop(0)

    wh = workingHours( values, values2 )
    WORKING_HOURS[ employee_id ] = wh

    wht = workingHoursTrend( values, values2 )
    WORKING_HOURS_TREND[ employee_id ] = wht
    
#: Combine
general['WorkingDays'] = general['EmployeeID'].map( WORKING_DAYS )
general['LateDays'] = general['EmployeeID'].map( LATE_DAYS )
general['WorkingHours'] = general['EmployeeID'].map( WORKING_HOURS )
general['WorkingHoursTrend'] = general['EmployeeID'].map( WORKING_HOURS_TREND )


general['ConfortZone'] = general['YearsAtCompany'] > 2 # True / False
general['ConfortZone'] = general['ConfortZone'].astype(int) # True => 1 , False = 0


#: Shuffle
general = general.sample(frac = 1.0)

#: Fill na
general = general.fillna(0)

#: Create a new feature from lda 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
xx = general.copy()
yy = xx[target]
del xx[target]

lda.fit( xx, yy )
general['LDA_RESULT'] = lda.predict( xx )



#: Save to file
general.to_csv("output.csv", index = False)


#* Use environments below
#* DEV
#* PREP





#: Split
limit = int(len(general) * 0.70)
train = general[:limit]
test = general[limit:]

print("BEFORE, BALANCING",  train.shape)
print("BEFORE, BALANCING", train[target].value_counts())

#! (0) DO NOT USE ACCURACY IN INBALANCED DATASET
#! (1) TRICKS IN INBALANCED DATASET: REBALANCING!!!
#! (2) TRICKS IN INBALANCED DATASET: USE F1_SCORE INSTEAD OF ACCURACY
#! (3) TRICKS IN INBALANCED DATASET: USE COST MATRIX
#!     a- Business department
#!     b- Research on internet
#!     c- destekli SALLA :)
#! (4) FOR ALL CLASSIFICATION PROBLEMS, WE MAY SPLIT DATASET INTO SUB DATASETS FOR ACHIEVING HIGHER ACCURACY






trainP = train[ train[target] == 1 ]
trainN = train[ train[target] == 0 ]
trainN = trainN.sample(frac = 0.5)

train = pd.concat( [trainP, trainN] )
print("AFTER, BALANCING",  train.shape)
print("AFTER, BALANCING", train[target].value_counts())

train_y = train[target]
del train[target]

test_y = test[target]  # SADECE TARGET KOLONU, 1 kolonluk veri
del test[target] # KALAN TUM KOLONLAR, N kolonluk veri

# PARAMETER OPTIMIZATION (changing parameters of algorithm)
clf = RandomForestClassifier()
clf.fit( train, train_y )
print( clf.score( test, test_y ) )






"""
# 5000, 
c0 = general[ general['ConfortZone'] == 0 ] # 2000
c1 = general[ general['ConfortZone'] == 1 ] # 3000

del c0['ConfortZone']
del c1['ConfortZone']


limit = int(len(c0) * 0.70)
c0_train = c0[:limit]
c0_test = c0[limit:]

# .......

"""

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

clf1 = RandomForestClassifier()
clf2 = MLPClassifier()
clf3 = LinearSVC()


sonuc1 = clf1.predict( test )
sonuc2 = clf2.predict( test )
sonuc3 = clf3.predict( test )

en_genel_sonuc = sonuc1 + sonuc2 + sonuc3 
vallahi_en_son_sonuc = en_genel_sonuc > 1 # 2, 3

# 0 1 2 3


# 0 => bu kisi churn etmeyecek, churn ediyorsa, o zaman 3 algoritmanin 3 u de yanlis yapti, ki dusuk olasilik
# 1 => bu kisi churn etmeyecek, churn ediyorsa, o zaman 3 algoritmanin 2 si de yanlis yapti, ki dusuk olasilik
# 2 => bu kisi churn edecek, churn etmiyorsa, o zaman 3 algoritmanin 1 i de yanlis yapti, ki dusuk olasilik
# 3 => bu kisi churn edecek, churn etmiyorsa, o zaman 3 algoritmanin 3 si de yanlis yapti, ki dusuk olasilik


# =====================================================================================


#! VOTING !

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X, y)
print(eclf1.predict(X))



# ========================

sonuc1 = clf1.predict( test )
sonuc2 = clf2.predict( test )
sonuc3 = clf3.predict( test )

en_genel_sonuc = sonuc1 + sonuc2 + sonuc3
vallahi_en_son_sonuc = en_genel_sonuc > 0

