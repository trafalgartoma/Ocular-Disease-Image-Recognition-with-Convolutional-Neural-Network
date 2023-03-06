import csv
import pandas as pd


# class names
Normal = 0
Diabetes = 0
Glaucoma = 0
Cataract = 0
AMD = 0
Hypertension = 0
Myopia = 0
Others = 0

data_csv = "/Users/giuse/Desktop/file_labels_final.csv"  # file csv con le annotazioni

# lettura del CSV
data = pd.read_csv(data_csv, sep=';', encoding='utf-8')  # lettura del csv

print(data.head().to_string())
# print(data.columns)



with open(data_csv) as csvDataFile:
    csv_reader = csv.reader(csvDataFile, delimiter=',')
    next(csv_reader)  # skip prima riga
    ts = 0
    vs = 0
    threshold = 1000
    for row in csv_reader:
        column_id = row[0]
        labels = [0, 0, 0, 0, 0, 0, 0, 0]
        labels[0] = row[1]  # normal
        labels[1] = row[2]  # diabetes
        labels[2] = row[3]  # glaucoma
        labels[3] = row[4]  # cataract
        labels[4] = row[5]  # amd
        labels[5] = row[6]  # hypertension
        labels[6] = row[7]  # myopia
        labels[7] = row[8]  # others
        if(labels ==  ['1', '0', '0', '0', '0', '0', '0', '0']):
            Normal = Normal + 1
        elif (labels == ['0', '1', '0', '0', '0', '0', '0', '0']):
            Diabetes = Diabetes + 1
        elif (labels == ['0', '0', '1', '0', '0', '0', '0', '0']):
            Glaucoma = Glaucoma +1
        elif (labels == ['0', '0', '0', '1', '0', '0', '0', '0']):
            Cataract = Cataract +1
        elif (labels == ['0', '0', '0', '0', '1', '0', '0', '0']):
            AMD = AMD + 1
        elif (labels == ['0', '0', '0', '0', '0', '1', '0', '0']):
           Hypertension = Hypertension + 1
        elif (labels == ['0', '0', '0', '0', '0', '0', '1', '0']):
           Myopia = Myopia + 1
        elif (labels == ['0', '0', '0', '0', '0', '0', '0', '1']):
            Others = Others + 1
        elif(True):
            print("no")




print("N : ", Normal)
print("D : ", Diabetes)
print("G : ", Glaucoma)
print("C : ",Cataract)
print("A: ",AMD)
print("H: ",Hypertension)
print("M: ",Myopia)
print("O: ",Others)
