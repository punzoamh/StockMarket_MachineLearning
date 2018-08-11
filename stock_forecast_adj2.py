import datetime
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
import pylab as pl
from pandas.io.data import DataReader
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn import datasets
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

#text_file = open("scores/accuracy_changed.txt", "w")
#text_file.write('Dates from 300 Days\n 50 Days Prediction Accuracy\n')
def main():
    global text_file
    i = 0
    """
    while(i < 50):
        #i =+ 1

        days = np.array(['100','200','300','400','500','600','700','800','900','1000','1100',
        '1200','1300','1400','1500','1600','1700','1800','1900','2000','2100','2200',
        '2300','2400','2500','2600','2700','2800','2900','3000','3100',
        '3200','3300','3400','3500','3600','3700','3800','3900','4000','4100',
        '4200','4300','4400','4500','4600','4800','4900'])
        yr = np.array([2014,2014,2014,2013,2013,2012,2012,2012,2011,2011,2010,2010,2010,2009,
        2009,2009,2008,2008,2007,2007,2007,2006,2006,2006,2005,2005,2004,
        2004,2004,2003,2003,2002,2002,2002,2001,2001,2001,2000,2000,1999,1999,
        1999,1998,1998,1997,1997,1997,1996,1996])
        mo = np.array([10,6,1,9,4,11,7,2,10,5,12,8,3,11,6,1,9,4,12,7,2,10,5,1,8,
        3,11,6,2,9,4,12,7,3,10,5,1,8,4,11,6,2,9,5,12,7,3,10,6])
        dy = np.array([29,11,22,4,17,28,11,22,5,18,29,11,24,4,17,28,10,23,5,18,28,11,24,4,17,
        30,10,23,4,17,30,11,24,6,17,30,10,23,5,17,30,10,23,6,17,30,12,23,5])

        yr2 = np.array([2014,2014,2014,2014,2014,2014,2014,2014,2014,2014,2014,
        2014,2014,2014,2014,2014,2014,2014,2014,2014,2014,2013,2013,2013,2013,
        2013,2013,2013,2013,2013,2013,2013,2013,2013,2013,2013,2013,2013,2013])
        mo2 = np.array([10,10,10,9,9,8,8,7,7,6,6,5,5,4,4,4,3,3,2,2,1,1,12,12,
        11,11,10,10,10,9,9,8,8,7,7,6,6,5,5,5,4])
        dy2 = np.array([29,15,1,17,3,20,6,23,9,25,11,28,14,30,16,2,19,5,19,5,
        22,8,25,11,27,13,30,16,2,18,4,21,7,24,10,26,12,29,15,1,7])
        days2 = np.array(['100','110','120','130','140','150','160','170','180',
        '190','200','210','220','230','240','250','260','270','280','290',
        '300','310','320','330','340','350','360','370','380','390','400',
        '410','420','430','440','450','460','470','480','490','500'])

        title = 'scores/accuracy_' + str(days[i]) + '.txt'
        text_file = open(title, "w")
        text_file.write('Dates from ' + str(days[i]) + ' Days\n 50 Days Prediction Accuracy\n')
        training_date = datetime.datetime(yr[i],mo[i],dy[i])
        makePrediction(training_date,days[i])
        i += 1
        print i
    """
    title = 'scores/accuracy_' + str(200) + 'SP500.txt'
    text_file = open(title, "w")
    text_file.write('Dates from ' + str(200) + ' Days\n 50 Days Prediction Accuracy\n')
    training_date = datetime.datetime(2014,6,11)
    makePrediction(training_date,200)

def create_lagged_series(symbol, start_date, end_date, lags=5):
    """This creates a pandas DataFrame that stores the percentage returns of the
    adjusted closing value of a stock obtained from Yahoo Finance, along with
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""

    # Obtain stock information from Yahoo Finance
    ts = DataReader(symbol, "yahoo", start_date-datetime.timedelta(days=365), end_date)
    #ts["Close"].to_csv("apple.csv", header=True)

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]
    tslag2 = pd.DataFrame(index=ts.index)
    tslag2["Close"]  = ts["Close"]
    # Create the shifted lag series of prior trading period close values
    for i in xrange(0,lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    #tsret["Close"] = tslag["Close"]
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in xrange(0,lags):
        tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

    tsret2 = pd.DataFrame(index=tslag2.index)
    for i in xrange(0, lags):
        tsret2["Lag%s" % str(i+1)] = ts["Close"].shift(i+1)
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    """
    THE PLOT
    """
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(ts["Close"], c='k', linewidth=2.0,label = "Close Price")
    plt.plot(tsret2["Lag1"], c='r', linewidth=2.0, linestyle='--')
    plt.legend(loc='upper left')
    plt.ylim(20,100)
    #plt.yticks([85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140])
    plt.xlabel("Dates")
    plt.ylabel("Closing Price")
    #title
    plt.title("Logistic Regression NASDAQ Lag 1 VS. Lag 2")
    plt.subplot(2,1,2)
    plt.plot(ts["Close"], c='k', linewidth=2.0, label = "Close Price")
    plt.plot(tsret2["Lag2"], c='c', linewidth=2.0, linestyle='--')

    #subplot(2,1,3)
    #plt.plot(ts["Close"])
    #plt.plot(tsret2["Lag3"], c='k')
    plt.ylim(20,100)
    #plt.yticks([85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140])
    plt.xlabel("Dates")
    plt.ylabel("Closing Price")
    plt.legend(loc='upper left')
    #print tsret2["Lag3"]
    #print "+++++++++++++++++++++++++++++++++++++++++++++"
    #print ts["Close"]
    #print "+++++++++++++++++++++++++++++++++++++++++++++"
    return tsret

def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""

    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)
    score = model.score(X_train,y_train, sample_weight = None)
    print score
    text_file.write(name + ' : ')
    text_file.write(str(score) + '\n')
    plt.plot(pred[name])
    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print "%s: %.3f" % (name, hit_rate)
    #print pred["Actual"]
    #print X_train
    #print y_train
    #print X_test

def makePrediction(training_date,days):
    #training_date = datetime.datetime(2014,1,22)
    #title = 'scores/accuracy_' + str(days) + '.txt'
    #print title
    #text_file = open(title, "w")


    snpret = create_lagged_series("^GSPC", training_date, datetime.datetime(2015,5,26), lags=5)
    #print snpret
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["Lag1","Lag2","Lag3","Lag4","Lag5"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2015,3,17)
    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    #print pred
    pred["Actual"] = y_test
    # Create and fit the three models
    print "Hit Rates:"
    models = [("Linear", linear_model.LinearRegression()),("LR",LogisticRegression()),
    ("KNN",neighbors.KNeighborsClassifier(n_neighbors = 3)),
    ("SVM",SVC(C=10)),("RF",RandomForestClassifier(n_estimators=4))]
    for m in models:
        fit_model(m[0],m[1],X_train, y_train, X_test, pred)

    logistic = LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose = True)
    classifier = Pipeline(steps=[('rbm',rbm), ('logistic', logistic)])
    rbm.learning_rate = .06
    rbm.n_iter = 15
    rbm.n_components = 100
    logistic.C = 6000
    classifier.fit(X_train, y_train)
    logistic_classifier = LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train,y_train)
    score = classifier.score(X_train,y_train)
    print score
    text_file.write('Neural Network : ' + str(score) + '\n')


    # 100 Days
    text_file.write('100 Days Prediction Accuracies\n')
    snpret = create_lagged_series("NDAQ", training_date, datetime.datetime(2015,8,6), lags=5)
    #print snpret
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["Lag1","Lag2","Lag3","Lag4","Lag5"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2015,3,17)
    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    #print pred
    pred["Actual"] = y_test
    # Create and fit the three models
    print "Hit Rates:"
    models = [("Linear", linear_model.LinearRegression()),("LR",LogisticRegression()),
    ("KNN",neighbors.KNeighborsClassifier(n_neighbors = 3)),
    ("SVM",SVC(C=10)),("RF",RandomForestClassifier(n_estimators=4))]
    for m in models:
        fit_model(m[0],m[1],X_train, y_train, X_test, pred)

    logistic = LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose = True)
    classifier = Pipeline(steps=[('rbm',rbm), ('logistic', logistic)])
    rbm.learning_rate = .06
    rbm.n_iter = 15
    rbm.n_components = 100
    logistic.C = 6000
    classifier.fit(X_train, y_train)
    logistic_classifier = LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train,y_train)
    score = classifier.score(X_train,y_train)
    print score
    text_file.write('Neural Network : ' + str(score) + '\n')

    # 200 Days
    text_file.write('200 Days Prediction Accuracies\n')
    snpret = create_lagged_series("NDAQ", training_date, datetime.datetime(2015,12,31), lags=5)
    #print snpret
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["Lag1","Lag2","Lag3","Lag4","Lag5"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2015,3,17)
    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    #print pred
    pred["Actual"] = y_test
    # Create and fit the three models
    print "Hit Rates:"
    models = [("Linear", linear_model.LinearRegression()),("LR",LogisticRegression()),
    ("KNN",neighbors.KNeighborsClassifier(n_neighbors = 3)),
    ("SVM",SVC(C=10)),("RF",RandomForestClassifier(n_estimators=4))]
    for m in models:
        fit_model(m[0],m[1],X_train, y_train, X_test, pred)

    logistic = LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose = True)
    classifier = Pipeline(steps=[('rbm',rbm), ('logistic', logistic)])
    rbm.learning_rate = .06
    rbm.n_iter = 15
    rbm.n_components = 100
    logistic.C = 6000
    classifier.fit(X_train, y_train)
    logistic_classifier = LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train,y_train)
    score = classifier.score(X_train,y_train)
    print score
    text_file.write('Neural Network : ' + str(score))
    text_file.close()

main()


"""
if __name__ == "__main__":
    # Create a lagged series of the S&P500 US stock market index
    #^IXIC
    training_date = datetime.datetime(2014,1,22)

    snpret = create_lagged_series("NDAQ", training_date, datetime.datetime(2015,5,26), lags=5)
    #print snpret
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["Lag1","Lag2","Lag3","Lag4","Lag5"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2015,3,17)
    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    #print pred
    pred["Actual"] = y_test
    # Create and fit the three models
    print "Hit Rates:"
    models = [("Linear", linear_model.LinearRegression()),("LR",LogisticRegression()),
    ("KNN",neighbors.KNeighborsClassifier(n_neighbors = 2)),
    ("SVM",SVC(C=1)),("RF",RandomForestClassifier(n_estimators=1))]
    for m in models:
        fit_model(m[0],m[1],X_train, y_train, X_test, pred)

    logistic = LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose = True)
    classifier = Pipeline(steps=[('rbm',rbm), ('logistic', logistic)])
    rbm.learning_rate = .06
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 6000
    classifier.fit(X_train, y_train)
    logistic_classifier = LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train,y_train)
    score = classifier.score(X_train,y_train)
    print score
    text_file.write('Neural Network : ' + str(score) + '\n')


    # 100 Days
    text_file.write('100 Days Prediction Accuracies\n')
    snpret = create_lagged_series("NDAQ", training_date, datetime.datetime(2015,8,6), lags=5)
    #print snpret
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["Lag1","Lag2","Lag3","Lag4","Lag5"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2015,3,17)
    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    #print pred
    pred["Actual"] = y_test
    # Create and fit the three models
    print "Hit Rates:"
    models = [("Linear", linear_model.LinearRegression()),("LR",LogisticRegression()),
    ("KNN",neighbors.KNeighborsClassifier(n_neighbors = 2)),
    ("SVM",SVC(C=1)),("RF",RandomForestClassifier(n_estimators=1))]
    for m in models:
        fit_model(m[0],m[1],X_train, y_train, X_test, pred)

    logistic = LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose = True)
    classifier = Pipeline(steps=[('rbm',rbm), ('logistic', logistic)])
    rbm.learning_rate = .06
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 6000
    classifier.fit(X_train, y_train)
    logistic_classifier = LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train,y_train)
    score = classifier.score(X_train,y_train)
    print score
    text_file.write('Neural Network : ' + str(score) + '\n')

    # 200 Days
    text_file.write('200 Days Prediction Accuracies\n')
    snpret = create_lagged_series("NDAQ", training_date, datetime.datetime(2015,12,31), lags=5)
    #print snpret
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["Lag1","Lag2","Lag3","Lag4","Lag5"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2015,3,17)
    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    #print pred
    pred["Actual"] = y_test
    # Create and fit the three models
    print "Hit Rates:"
    models = [("Linear", linear_model.LinearRegression()),("LR",LogisticRegression()),
    ("KNN",neighbors.KNeighborsClassifier(n_neighbors = 2)),
    ("SVM",SVC(C=1)),("RF",RandomForestClassifier(n_estimators=1))]
    for m in models:
        fit_model(m[0],m[1],X_train, y_train, X_test, pred)

    logistic = LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose = True)
    classifier = Pipeline(steps=[('rbm',rbm), ('logistic', logistic)])
    rbm.learning_rate = .06
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 6000
    classifier.fit(X_train, y_train)
    logistic_classifier = LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train,y_train)
    score = classifier.score(X_train,y_train)
    print score
    text_file.write('Neural Network : ' + str(score))
    text_file.close()
    #fit_model("LR", LogisticRegression(), X_train, y_train, X_test, pred)
"""
