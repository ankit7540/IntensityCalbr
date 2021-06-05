# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:17:50 2020

@author: Ankit Raj
"""

import requests
import pandas as pd
import mysql.connector

from .extract_header import table_header

#**********************************************************

def get_data_India(url):
    html = requests.get(url).content
    df_list = pd.read_html(html)

    #--------------------------------------
    df = df_list[-1]

    print('\n\tGetting data...')

    header=table_header(url)

    print('-----------------------------------')
    for i in range(len(header)):
        print('\t\t Header : ',i,' ', header[i])
    print('-----------------------------------')
    hd=header[4]
    #extract foeign cases------
    fr_str=header[2]
    if "foreign" in fr_str:
        num=[int(s) for s in fr_str.split() if s.isdigit()]
        FC = int(num[0])
        print('Foreign cases : ',num[0], type(FC))
    else:
        FC=0  # check FC==0 later upon failure
    #--------------------------

    #--------------------------------------
    selected=['Name of State / UT',	header[2],'Cured/Discharged/Migrated',	hd ]
    new_f = df[selected]
    new_f = new_f.rename(columns={'Name of State / UT': 'State', header[2]: 'IN'})
    new_f = new_f.rename(columns={ 'Cured/Discharged/Migrated':'RC' })
    new_f = new_f.rename(columns={ hd:'Death' })

    last=new_f.tail(1)
    #print((last), '\n', last.iat[0,0])
    val = last.iat[0,0]
    #print(type(val))
    #------------------------------
    if "Total number of confirmed cases" in val:
        print('\t Last row found ok.')
    else:
        print('\t *Removing last row.')
        new_f=new_f[:-1]
    #------------------------------
    #------------------------------

    #remove extra rows
    new_f = new_f[~new_f.State.str.contains("distribution")]
    new_f = new_f[~new_f.State.str.contains("shifted")]
    new_f = new_f[~new_f.State.str.contains("ICMR")]
    new_f = new_f[~new_f.State.str.contains("assigned")]
    new_f = new_f[~new_f.State.str.contains("Including foreign Nationals")]
    new_f = new_f[~new_f.State.str.contains("cases due to comorbidities")]

    #------------------------------


    new_f.to_csv('temp.csv')
    print('\t')

    #--------------------------------------
    f=pd.read_csv("temp.csv",sep=r'\s*,\s*',encoding='ascii',engine='python', index_col ='State')

    f2 = f.transpose()
    f2 = f2.rename(columns={'Total number of confirmed cases in India': 'Total'})
    del f2['Total']

    #--------------------------------------
    f2=f2.transpose()

    # give the output table ---------------
    f2.to_csv("output_simplified_IN.csv", index=True)
    #Change datatype to integer here
    print(f2, f2.dtypes)
    f2['IN']=f2['IN'].astype(int)
    f2['RC']=f2['RC'].astype(int)
    f2['Death']=f2['Death'].astype(int)

    print(f2.dtypes)

    IN_sum=f2['IN'].sum()
    RC_sum=f2['RC'].sum()
    D_sum=f2['Death'].sum()
    print('\t',IN_sum,  RC_sum, D_sum)
    return(IN_sum, FC,  RC_sum, D_sum)
    #=======================================================================

#**********************************************************
#--------------------------------------------------
def update_india_data(in_INN, in_FR, in_RC, in_D, in_migrate ):

    # in_IN = infected Indian nationals
    # in_FR = infected foreign nationals
    # in_RC = recovered
    # in_D = dead

    confirmed=in_INN
    confirmed=int(confirmed)
    in_FRN=int(in_FR)
    in_RC=int(in_RC)
    in_D=int(in_D)
    in_migrate=int(in_migrate)
    # --------------------------------
    mydb = mysql.connector.connect(
      host="194.59.164.64",
      user="u917412676_site_access",
      passwd="gRb6CLavyQkXZau2",
      database="u917412676_ncov_data"
    )

    values=(confirmed, in_D, in_RC, in_FRN  )

    mycursor = mydb.cursor()
    sql = "UPDATE `data_india` SET `infected`=%s,`dead`=%s,`recovered`=%s, `foreignp`=%s    ";

    mycursor.execute(sql,values)
    mydb.commit()
    print('\t MYSQL executed.')
    print("\t",mycursor.rowcount, " record(s) affected.")

#--------------------------------------------------
#--------------------------------------------------

url_mohfw='https://www.mohfw.gov.in/'
data=get_data_India(url_mohfw)
print('\t\t', data[0], data[1], data[2], data[3] )
#print(data.type)

#migrated=1
#update_india_data( data[0],data[1],data[2], migrated )

#--------------------------------------------------
#--------------------------------------------------
