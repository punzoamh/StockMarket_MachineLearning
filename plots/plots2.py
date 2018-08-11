import matplotlib.pyplot as plt
import numpy as np
# plotly.plotly as py
import csv
import ast
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Right now the program plots lin1-ann1 which are averages of the 3 month data Sets
It will only graph one set of data at a time.
Say you want 50 day data at 3 month set then the value for linear would need to be lin1
LR would need to be log1, etc.

"""


def main():
    day = 100
    """
    while(day < 2600):

        data = []
        with open('bar_' + str(day) +'.csv','rb') as f:
            reader = csv.reader(f)
            for data in reader:
                plot1(float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),str(day))
        day += 100

        """
    
    with open('bar_' + str(200) +'sp.csv','rb') as f:
        reader = csv.reader(f)
        for data in reader:
            plot1(float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),str(200))


    with open('bar_' + str(200) +'_dj.csv','rb') as f:
        reader = csv.reader(f)
        for data in reader:
            plot1(float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),str(200))

    #oneLine3300()
    #manyLines()
    #oneLine()
    #manyLines3300()
    #manyLines500()

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
    log_leg = plt.Line2D([], [], color='green', marker='s',
                          markersize=5, label='Logistic Regression')
    knn_leg = plt.Line2D([], [], color='red', marker='D',
                          markersize=5, label='KNN')
    svm_leg = plt.Line2D([], [], color='cyan', marker='*',
                          markersize=5, label='SVM')
    rf_leg = plt.Line2D([], [], color='purple', marker='<',
                          markersize=5, label='Random Forest')
    ann_leg = plt.Line2D([], [], color='yellow', marker='|',
                          markersize=5, label='Neural Net')

    plt.legend(handles=[lin_leg,log_leg,knn_leg,svm_leg,rf_leg,ann_leg],loc = 4)
    plt.savefig("all_sensitivities.png")

    plt.show()

def manyLines500():
    plt.figure(figsize=(15, 8), dpi=100)
    item = []
    with open('lin_csv_500.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    #plt.figure(figsize=(13, 8), dpi=100)

    linePlot500(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],item[33],item[34],item[35],item[36],
    'Sensitivity of Linear Regression over 2500 Days for NASDAQ Data','x')

    item = []
    with open('log_csv_500.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot500(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],item[33],item[34],item[35],item[36],
    'Sensitivity of Logistic Regression over 2500 Days for NASDAQ Data','s')

    item = []
    with open('knn_csv_500.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot500(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],item[33],item[34],item[35],item[36],
    'Sensitivity of K-Nearest Neighbor over 2500 Days for NASDAQ Data','D')

    item = []
    with open('svm_csv_500.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot500(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],item[33],item[34],item[35],item[36],
    'Sensitivity of Support Vector Machine over 2500 Days for NASDAQ Data','*')

    item = []
    with open('rf_csv_500.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot500(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],item[33],item[34],item[35],item[36],
    'Sensitivity of Random Forest over 2500 Days for NASDAQ Data','<')

    item = []
    with open('ann_csv_500.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot500(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],item[33],item[34],item[35],item[36],
    'Sensitivity of Algorithms to NASDAQ data over 500 Days','|')
    lin_leg = plt.Line2D([], [], color='blue', marker='x',
                          markersize=5, label='Linear Regression')
    log_leg = plt.Line2D([], [], color='green', marker='s',
                          markersize=5, label='Logistic Regression')
    knn_leg = plt.Line2D([], [], color='red', marker='D',
                          markersize=5, label='KNN')
    svm_leg = plt.Line2D([], [], color='cyan', marker='*',
                          markersize=5, label='SVM')
    rf_leg = plt.Line2D([], [], color='purple', marker='<',
                          markersize=5, label='Random Forest')
    ann_leg = plt.Line2D([], [], color='yellow', marker='|',
                          markersize=5, label='Neural Net')

    plt.legend(handles=[lin_leg,log_leg,knn_leg,svm_leg,rf_leg,ann_leg],loc = 4)
    plt.savefig("all_sensitivities_500.png")

    plt.show()

def oneLine3300():
    plt.figure(figsize=(25, 8), dpi=100)
    item = []
    with open('lin_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    #plt.figure(figsize=(13, 8), dpi=100)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Dataset Size Sensitivity of Linear Regression over 3300 Days for NASDAQ Data','x')
    #plt.tight_layout()
    plt.xlim(0,32)
    plt.savefig("lin_sensitivities_3300.png")
    #plt.tight_layout()
    plt.show()
    plt.figure(figsize=(25, 8), dpi=100)
    item = []
    with open('log_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Dataset Size Sensitivity of Logistic Regression over 3300 Days for NASDAQ Data','s')
    plt.xlim(0,32)
    plt.savefig("log_sensitivities_3300.png")

    plt.show()
    plt.figure(figsize=(25, 8), dpi=100)
    item = []
    with open('knn_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Dataset Size Sensitivity of K-Nearest Neighbor over 3300 Days for NASDAQ Data','D')
    plt.xlim(0,32)
    plt.savefig("knn_sensitivities_3300.png")

    plt.show()
    plt.figure(figsize=(25, 8), dpi=100)
    item = []
    with open('svm_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Dataset Size Sensitivity of Support Vector Machine over 3300 Days for NASDAQ Data','*')
    plt.xlim(0,32)
    plt.savefig("svm_sensitivities_3300.png")

    plt.show()
    plt.figure(figsize=(25, 8), dpi=100)
    item = []
    with open('rf_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Dataset Size Sensitivity of Random Forest over 3300 Days for NASDAQ Data','<')
    plt.xlim(0,32)
    plt.savefig("rf_sensitivities_3300.png")

    plt.show()
    plt.figure(figsize=(25, 8), dpi=100)
    item = []
    with open('ann_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Dataset Size Sensitivity of Neural Network to NASDAQ data over 3300 Days','|')
    plt.xlim(0,32)
    plt.savefig("ann_sensitivities_3300.png")

    plt.show()


def manyLines3300():
    plt.figure(figsize=(25, 8), dpi=100)
    item = []
    with open('lin_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    #plt.figure(figsize=(13, 8), dpi=100)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Sensitivity of Linear Regression over 2500 Days for NASDAQ Data','x')

    item = []
    with open('log_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Sensitivity of Logistic Regression over 2500 Days for NASDAQ Data','s')

    item = []
    with open('knn_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Sensitivity of K-Nearest Neighbor over 2500 Days for NASDAQ Data','D')

    item = []
    with open('svm_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Sensitivity of Support Vector Machine over 2500 Days for NASDAQ Data','*')

    item = []
    with open('rf_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Sensitivity of Random Forest over 2500 Days for NASDAQ Data','<')

    item = []
    with open('ann_csv.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            print row
            item.append(row)

    linePlot3300(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],
    item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],
    item[22],item[23],item[24],item[25],item[26],item[27],item[28],item[29],item[30],
    item[31],item[32],
    'Dataset Size Sensitivity of Algorithms to NASDAQ data over 3300 Days','|')
    lin_leg = plt.Line2D([], [], color='blue', marker='x',
                          markersize=5, label='Linear Regression')
    log_leg = plt.Line2D([], [], color='green', marker='s',
                          markersize=5, label='Logistic Regression')
    knn_leg = plt.Line2D([], [], color='red', marker='D',
                          markersize=5, label='KNN')
    svm_leg = plt.Line2D([], [], color='cyan', marker='*',
                          markersize=5, label='SVM')
    rf_leg = plt.Line2D([], [], color='purple', marker='<',
                          markersize=5, label='Random Forest')
    ann_leg = plt.Line2D([], [], color='yellow', marker='|',
                          markersize=5, label='Neural Net')

    plt.legend(handles=[lin_leg,log_leg,knn_leg,svm_leg,rf_leg,ann_leg],loc = 4)
    plt.savefig("all_sensitivities_3300.png")

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

def linePlot500(one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive,twentysix,
twentyseven,twentyeight,twentynine,thirty,thirtyone,thirtytwo,thirtythree,thirtyfour,
thirtyfive,thirtysix,thirtyseven,title,shape):
    objects = ('110','120','130','140','150','160','170','180','190','200',
    '210','220','230','240','250','260','270','280','290','300','310','320',
    '330','340','350','360','370','380','390','400','410','420','430','440',
    '450','460','470','480')
    #plt.figure(figsize=(13, 8), dpi=100)
    x_pos = np.arange(len(objects))

    plt.plot([one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
    seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive,twentysix,
    twentyseven,twentyeight,twentynine,thirty,thirtyone,thirtytwo,thirtythree,thirtyfour,
    thirtyfive,thirtysix,thirtyseven])
    plt.plot([one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
    seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive,twentysix,
    twentyseven,twentyeight,twentynine,thirty,thirtyone,thirtytwo,thirtythree,thirtyfour,
    thirtyfive,thirtysix,thirtyseven],shape, color = 'blue')
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


def linePlot3300(one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive,twentysix,
twentyseven,twentyeight,twentynine,thirty,thirtyone,thirtytwo,thirtythree,title,shape):
    objects = ('1','2','3','4','5','6','7','8','9','10','11',
    '12','13','14','15','16','17','18','19','20','21','22',
    '23','24','25','26','27','28','29','30','31','32','33')
    #plt.figure(figsize=(13, 8), dpi=100)
    x_pos = np.arange(len(objects))

    plt.plot([one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
    seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive,twentysix,
    twentyseven,twentyeight,twentynine,thirty,thirtyone,thirtytwo,thirtythree])
    plt.plot([one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,
    seventeen,eighteen,nineteen,twenty,twentyone,twentytwo,twentythree,twentyfour,twentyfive,twentysix,
    twentyseven,twentyeight,twentynine,thirty,thirtyone,thirtytwo,thirtythree],shape, color = 'blue')
    plt.xticks(x_pos, objects)
    #plt.yticks(y_pos, objectsY)
    plt.ylim([0.0,1.0])
    #plt.xlim([100,2500])
    plt.xlabel('Training Days (Hundreds of Days)')
    plt.ylabel('Precent Acurate')
    #plt.legend(handles=[lin_leg,log_leg,knn_leg,svm_leg,rf_leg,ann_leg],loc = 9)
    plt.grid(True)
    #plt.legend()
    plt.title(title)
    #plt.show()

def plot1(lin,log,knn,svm,rf,ann,title):
    objects = ('Lin','Log','KNN','SVM','Random Forest','ANN')
    performance = [lin, log, knn, svm, rf, ann]
    y_pos = np.arange(len(objects))
    plt.figure(figsize=(15, 8), dpi=100)
    plt.bar(y_pos, performance, align='center', alpha=1.0)
    plt.xticks(y_pos, objects)
    plt.ylim([0.0,1.0])
    plt.ylabel('Percent Acurate')
    plt.title('Accuracies of Algorithms Predicting DJIA Prices with ' + title + ' Days Data')
    plt.annotate(str("{0:.2f}".format(lin)), xy=(.8, lin+.01), xytext=(-0.15, lin+.01))
    plt.annotate(str("{0:.2f}".format(log)), xy=(.8, log+.01), xytext=(.8, log+.01))
    plt.annotate(str("{0:.2f}".format(knn)), xy=(.8, knn+.01), xytext=(1.8, knn+.01))
    plt.annotate(str("{0:.2f}".format(svm)), xy=(.8, svm+.01), xytext=(2.8, svm+.01))
    plt.annotate(str("{0:.2f}".format(rf)), xy=(.8, rf+.01), xytext=(3.8, rf+.01))
    plt.annotate(str("{0:.2f}".format(ann)), xy=(.8, ann+.01), xytext=(4.8, ann+.01))
    #plt.savefig('Accu_' + title +'.png')
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
def plot3(lin,log,knn,svm,rf,ann,title):

    frequencies = [lin,log,knn,svm,rf,ann]   # bring some raw data

    freq_series = pd.Series.from_array(frequencies)   # in my original code I create a series and run on that, so for consistency I create a series from the list.

    x_labels = ['Linear Regression','Logistic Regression','K-Nearest Neighbor','Support Vector Machine',
    'Random Forrest','Neural Networks']

    # now to plot the figure...
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_title("Amount Frequency")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(x_labels)

    rects = ax.patches

    # Now make some labels
    labels = ["label%d" % i for i in xrange(len(rects))]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

    #plt.savefig("image.png")

main()
