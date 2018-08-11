import matplotlib.pyplot as plt
import numpy as np
# plotly.plotly as py
import csv
import ast
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

"""
Right now the program plots lin1-ann1 which are averages of the 3 month data Sets
It will only graph one set of data at a time.
Say you want 50 day data at 3 month set then the value for linear would need to be lin1
LR would need to be log1, etc.

"""


def main():
    day = 100

    while(day < 2600):

        data = []
        with open('bar_' + str(day) +'.csv','rb') as f:
            reader = csv.reader(f)
            for data in reader:
                plot1(float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),str(day))
        day += 100

    manyLines()
    oneLine()

def manyLines():
    plt.figure(figsize=(15, 8), dpi=100)
    item = []
    with open('lin_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    #plt.figure(figsize=(13, 8), dpi=100)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Linear Regression over 2500 Days for NASDAQ Data','x')

    item = []
    with open('log_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Logistic Regression over 2500 Days for NASDAQ Data','s')

    item = []
    with open('knn_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of K-Nearest Neighbor over 2500 Days for NASDAQ Data','D')

    item = []
    with open('svm_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Support Vector Machine over 2500 Days for NASDAQ Data','*')

    item = []
    with open('rf_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Random Forest over 2500 Days for NASDAQ Data','<')

    item = []
    with open('ann_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Algorithms to NASDAQ data over 2500 Days','|')
    lin_leg = plt.Line2D([], [], color='blue', marker='x',
                          markersize=5, label='Linear Regression')
    log_leg = plt.Line2D([], [], color='black', marker='s',
                          markersize=5, label='Logistic Regression')
    knn_leg = plt.Line2D([], [], color='blue', marker='D',
                          markersize=5, label='KNN')
    svm_leg = plt.Line2D([], [], color='black', marker='*',
                          markersize=5, label='SVM')
    rf_leg = plt.Line2D([], [], color='blue', marker='<',
                          markersize=5, label='Random Forest')
    ann_leg = plt.Line2D([], [], color='black', marker='|',
                          markersize=5, label='Neural Net')

    plt.legend(handles=[lin_leg,log_leg,knn_leg,svm_leg,rf_leg,ann_leg],loc = 4)
    plt.savefig("all_sensitivities.png")

    plt.show()

def oneLine():
    plt.figure(figsize=(15, 8), dpi=100)
    item = []
    with open('lin_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Linear Regression over 2500 Days for NASDAQ Data','x')
    plt.savefig("lin_sensitivities.png")
    plt.show()
    item = []
    plt.figure(figsize=(15, 8), dpi=100)
    with open('log_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Logistic Regression over 2500 Days for NASDAQ Data','s')
    plt.savefig("log_sensitivities.png")
    plt.show()
    item = []
    plt.figure(figsize=(15, 8), dpi=100)
    with open('knn_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of K Nearest Neighbor over 2500 Days for NASDAQ Data','D')
    plt.savefig("knn_sensitivities.png")
    plt.show()
    plt.figure(figsize=(15, 8), dpi=100)
    item = []
    with open('svm_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Support Vector Machine over 2500 Days for NASDAQ Data','*')
    plt.savefig("svm_sensitivities.png")
    plt.show()
    plt.figure(figsize=(15, 8), dpi=100)
    item = []
    with open('rf_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Random Forest over 2500 Days for NASDAQ Data','<')
    plt.savefig("rf_sensitivities.png")
    plt.show()
    plt.figure(figsize=(15, 8), dpi=100)
    item = []
    with open('ann_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],'Sensitivity of Artificial Neural Networks over 2500 Days for NASDAQ Data','|')
    plt.savefig("nnet_sensitivities.png")
    plt.show()

def linePlot(one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive,title,shape):
    objects = ('100','200','300','400','500','600','700','800','900','1000','1100',
    '1200','1300','1400','1500','1600','1700','1800','1900','2000','2100','2200',
    '2300','2400','2500')
    #plt.figure(figsize=(13, 8), dpi=100)
    x_pos = np.arange(len(objects))

    plt.plot([one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
    seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive])
    plt.plot([one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
    seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive],shape, color = 'blue')
    plt.xticks(x_pos, objects)
    #plt.yticks(y_pos, objectsY)
    plt.ylim([0.0,1.0])
    #plt.xlim([100,2500])
    plt.xlabel('Training Days')
    plt.ylabel('Precent Acurate')
    """
    lin_leg = plt.Line2D([], [], color='blue', marker='x',
                          markersize=5, label='Linear Regression')
    log_leg = plt.Line2D([], [], color='black', marker='s',
                          markersize=5, label='Logistic Regression')
    knn_leg = plt.Line2D([], [], color='blue', marker='D',
                          markersize=5, label='KNN')
    svm_leg = plt.Line2D([], [], color='black', marker='*',
                          markersize=5, label='SVM')
    rf_leg = plt.Line2D([], [], color='blue', marker='<',
                          markersize=5, label='Random Forest')
    ann_leg = plt.Line2D([], [], color='black', marker='|',
                          markersize=5, label='Neural Net')
    """
    #plt.legend(handles=[lin_leg,log_leg,knn_leg,svm_leg,rf_leg,ann_leg],loc = 9)
    plt.grid(True)
    #plt.legend()
    plt.title(title)
    #plt.show()

def plot1(lin,log,knn,svm,rf,ann,title):
    objects = ('Lin\n'+str("{0:.2f}".format(lin)),'Log\n'+str("{0:.2f}".format(log)),
    'KNN\n'+str("{0:.2f}".format(knn)),'SVM\n'+str("{0:.2f}".format(svm)),
    'Random Forest\n'+str("{0:.2f}".format(rf)), 'ANN\n'+str("{0:.2f}".format(ann)))
    performance = [lin, log, knn, svm, rf, ann]
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, performance, align='center', alpha=1.0)
    plt.xticks(y_pos, objects)
    plt.ylim([0.0,1.0])
    plt.ylabel('Percent Acurate')
    plt.title('Accuracies of Algorithms Predicting NASDAQ Prices with ' + title + ' Days Data')
    plt.savefig('Accu_' + title +'.png')
    plt.show()



"""
n_groups sets up the groupings, since we are comparing just one data set then there is 1 group
if we wanted to have to data sets side by side we could make this = 2
however each more rects would need to be added and
the bar_width would need to be incremented for each algorithm added.
So for example if n_groups = 2 and the 6 month data is added then
we would need rects6 = plt.bar(index + (bar_width*7), bar_lin2, bar_width...)
then rects7 = plt.bar(index + (bar_width*8), bar_log2, bar_width)...
and so on
I think opacity is the darkness of the color, we can play with this
"""
# data to plot
def plot2(lin,log,knn,svm,rf,ann):
    lin1 = lin
    log1 = log
    knn1 = knn
    svm1 = svm
    rf1 = rf
    ann1 = ann

    n_groups = 1
    bar_lin = (lin1)
    bar_log = (log1)
    bar_knn = (knn1)
    bar_svm = (svm1)
    bar_rf = (rf1)
    bar_ann = (ann1)

    #    create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    plt.xlabel('Data Sets')
    plt.ylabel('Percent Acurate')
    plt.title('Accuracies of Algorithms Predicting Stock Prices')
    plt.xticks(index + bar_width)
    plt.legend()

    plt.tight_layout()


    #plt.show()

    rects1 = plt.bar(index, bar_lin, bar_width,
                    alpha=opacity,
                    color='b',
                    label='lin')

    rects2 = plt.bar(index + bar_width, bar_log, bar_width,
                    alpha=opacity,
                    color='g',
                    label='log')

    rects3 = plt.bar(index + (bar_width*2), bar_knn, bar_width,
                 alpha=opacity,
                 color='r',
                 label='knn')
    rects4 = plt.bar(index + (bar_width*3), bar_svm, bar_width,
                 alpha=opacity,
                 color='#cc00cc',
                 label='svm')

    rects5 = plt.bar(index + (bar_width*4), bar_rf, bar_width,
                 alpha=opacity,
                 color='c',
                 label='rf')
    rects5 = plt.bar(index + (bar_width*5), bar_ann, bar_width,
                 alpha=opacity,
                 color='m',
                 label='ann')

    plt.xlabel('Data Sets')
    plt.ylabel('Percent Accurate')
    plt.title('Accuracies of Algorithms Predicting Stock Prices')
    blue_line = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='Blue stars')
    plt.legend(handles=[blue_line])


    plt.tight_layout()


    plt.show()


main()
