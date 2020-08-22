#Importing Libraries
from csv import reader
from csv import writer
import pandas as pd
from numpy import nan
#Function for Gender from string to Integer
def GenderToInt(gender):
    genderlist = ['f','m']
    return(genderlist.index(gender)+1)
#Function for ethnicity from
def EthnicityToInt(ethnicity):
    ethnicitylist = ['White-European','Latino','Black','Asian','Middle Eastern','Pasifika','South Asian','Hispanic','Turkish','Others','Nan']
    return(ethnicitylist.index(ethnicity)+1)
# Jaundice  from string to Integer
def JaundiceToInt(jaundice):
    jaundicelist = ['no','yes']
    return(jaundicelist.index(jaundice)+1)
#Autism string to Integer
def AutismToInt(autism):
    autismlist = ['no','yes']
    return(autismlist.index(autism)+1)
#Function for country_of_recidence string to Integer
def CountryToInt(country_of_res):
    countrylist = ['United States','Brazil','Spain','Egypt','New Zealand','Bahamas','Burundi','Austria',
                   'Argentina','Jordan','Ireland','United Arab Emirates','Afghanistan','Lebanon','United Kingdom',
                   'South Africa','Italy','Pakistan','Bangladesh','Chile','France','China','Australia',
                   'Canada','Saudi Arabia','Netherlands','Romania','Sweden','Tonga','Oman','India','Philippines',
                   'Sri Lanka','Sierra Leone','Ethiopia','Viet Nam','Iran','Costa Rica','Germany','Mexico',
                   'Russia','Armenia','Iceland','Nicaragua','Hong Kong','Japan','Ukraine','Kazakhstan',
                   'AmericanSamoa','Uruguay','Serbia','Portugal','Malaysia','Ecuador','Niger','Belgium',
                   'Bolivia','Aruba','Finland','Turkey','Nepal','Indonesia','Angola','Azerbaijan','Iraq',
                   'Czech Republic','Cyprus','Others']
    return(countrylist.index(country_of_res)+1)
#funtion for App use string to Integer
def UsedAppToInt(used_app_before):
    usedApplist = ['no','yes']
    return(usedApplist.index(used_app_before)+1)
#Function for class string to Integer
def ClassToInt(ASD):
    classList = ['NO','YES']
    return(classList.index(ASD))
#opening input file and making a new output file after pre processing
with open(r'C:\Ruchita\MSc_Data_Science\Module-8-7085-Advanced_MAchine_Learning\CW-7085\Part_1\Autism\autism_input.csv', 'r') as read_obj,\
    open(r'C:\Ruchita\MSc_Data_Science\Module-8-7085-Advanced_MAchine_Learning\CW-7085\Part_1\Autism\autism_output.csv','w',newline='\n') as write_obj:
    csv_reader = reader(read_obj)
    csv_writer = writer(write_obj)
    #Applying function and removing age_range and relation row
    for row in csv_reader:
        if row[11] == 'gender':
            pass
        else:
            row[11] = GenderToInt(row[11])
            row[12] = EthnicityToInt(row[12])
            row[13] = JaundiceToInt(row[13])
            row[14] = AutismToInt(row[14])
            row[15] = CountryToInt(row[15])
            row[16] = UsedAppToInt(row[16])
            row[20] = ClassToInt(row[20])
            del row[18]
            del row[18]
            #Writting output file
            csv_writer.writerow(row)







