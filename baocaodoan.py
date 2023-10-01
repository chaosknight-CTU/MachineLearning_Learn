import csv
import numpy as np
from sklearn.metrics import mean_absolute_error

# # Doc du lieu
def loadData(path):
    f = open(path, "r") #Mo file
    data = csv.reader(f) #Doc file dang csv
    data = np.array(list(data)) #Chuyen doi tap du lieu thanh dang ma tran
    data = np.delete(data, 0, 0) #Xoa dong dau tien
    np.random.shuffle(data) #Tron ngau nhien tap du lieu
    f.close() #Dong file
    trainSet = data[:int(len(data)*2/3.0)] #Tap training chiem 2/3 tap du lieu
    testSet = data[int(len(data)*2/3.0):] #Tap test chiem 1/3 tap du lieu
    return trainSet, testSet

# #tinh khoang cach
def calcDistance(pointA, pointB, numOfFeature=10): #Vi thuoc tinh la roi rac
    tmp = 0
    for i in range(numOfFeature): #Duyet qua cac thuoc tinh cua diem
        if pointA[i] == pointB[i]: #Neu 2 diem co thuoc tinh giong nhau
            tmp += 1 #Tang bien dem len 1
    return float((numOfFeature-tmp)/numOfFeature) #Tinh khoang cach dua vao so thuoc tinh giong nhau

# Tim k diem du lieu gan nhat
def kNearestNeighbor(trainSet, point, k):
    distances = [] #list luu khoang cach
    for item in trainSet:
        distances.append({ #append 1 dictionary
            "label1": int(item[-3]), # nhan thu nhat
            "label2": int(item[-2]), # nhan thu hai
            "label3": int(item[-1]), # nhan thu ba
            "value": calcDistance(item, point) # khoang cach 
        })
    distances.sort(key=lambda x: x["value"]) #Sap xep cac khoang cach theo thu tu tang dan
    labels1 = [item["label1"] for item in distances] #Luu lai cac nhan thu nhat
    labels2 = [item["label2"] for item in distances] #Luu lai cac nhan thu hai
    labels3 = [item["label3"] for item in distances] #Luu lai cac nhan thu ba
    labels1 = labels1[:k] # lay k du lieu dau tien
    labels2 = labels2[:k]
    labels3 = labels3[:k]
    labels = [sum(labels1)/k, sum(labels2)/k, sum(labels3)/k] # vi nhan lien tuc nen nhan du doan se lay trung binh
    return labels

if __name__ == "__main__":
    trainSet, testSet = loadData("flare1.csv")
    label1_expected = list() #Dung de luu lai cac gia tri nhan thuc su
    label2_expected = list()
    label3_expected = list()
    label1_predicted = list() #Dung de luu lai cac gia tri nhan du doan
    label2_predicted = list()
    label3_predicted = list()
    hien_thi_phan_tu = 10
    for item in range(hien_thi_phan_tu):
        label1_expected.append(int(testSet[item][-3]))
        label2_expected.append(int(testSet[item][-2]))
        label3_expected.append(int(testSet[item][-1]))
        knn = kNearestNeighbor(trainSet, testSet[item], 5) #KNN voi 5 diem gan nhat
        print("{}) label: {}, {}, {} -> predicted: {}, {}, {}".format(item+1, testSet[item][-3], testSet[item][-2], testSet[item][-1], knn[0], knn[1], knn[2]))
        label1_predicted.append(knn[0])
        label2_predicted.append(knn[1])
        label3_predicted.append(knn[2])
    errors1 = float(mean_absolute_error(label1_expected, label1_predicted)) #Chi so MAE
    errors2 = float(mean_absolute_error(label2_expected, label2_predicted))
    errors3 = float(mean_absolute_error(label3_expected, label3_predicted))
    print("MAE cua nhan 1: "+str(errors1))
    print("MAE cua nhan 2: "+str(errors2))
    print("MAE cua nhan 3: "+str(errors3))