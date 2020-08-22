# Importing Libraries
import os
# Getting Arff file
files = [arff for arff in os.listdir('.') if arff.endswith(".arff")]
# Function for arff to csv convergence
def toCsv(content):
    data = False
    header = ""
    newContent = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute") + 1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent
# path of csv_file
csv_path = r'C:\Ruchita\MSc_Data_Science\Module-8-7085-Advanced_MAchine_Learning\CW-7085\Part_1\Autism\autism_input.csv'
# Reading and writing file
# for file in files:
with open(
        r'C:\Ruchita\MSc_Data_Science\Module-8-7085-Advanced_MAchine_Learning\CW-7085\Part_1\Autism\Autism-Adult-Data.arff',
        "r") as inFile:
    content = inFile.readlines()
    name, ext = os.path.splitext(inFile.name)
    new = toCsv(content)
    with open(csv_path + ".csv", "w") as outFile:
        outFile.writelines(new)
