# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:59:14 2020

@author: ankit
"""
import os
import subprocess
import shutil
import requests
import pandas as pd
import datetime
#import time
from dateutil import tz
from urllib.request import urlopen

from .extract_header import table_header

#**********************************************************

t_head = os.path.join(os.path.dirname(__file__), './content/table_p1.txt')
t_foot = os.path.join(os.path.dirname(__file__), './content/table_p2.txt')
upload_script = os.path.join(os.path.dirname(__file__), './upload_ftp.sh')
#**********************************************************
def get_mod_string(url):
    f=urlopen(url)
    i=f.info()
    lm=i.items()[2]
    value=lm[1]
    #value = value.replace('GMT', '')
    value= (value.rstrip())
    utc = datetime.datetime.strptime(value, '%a, %d %b %Y %H:%M:%S %Z')
    month=(datetime.datetime.now(datetime.timezone.utc).strftime("%b"))


    # convert to Asia/Kolkata
    to_zone = tz.gettz('Asia/Kolkata')
    from_zone = tz.gettz('UTC')
    utc = utc.replace(tzinfo=from_zone)
    ist = utc.astimezone(to_zone)
    #----------------------------
    dtv=ist.strftime("%d ")
    mt=month
    yr=ist.strftime(" %Y, ")
    tm=ist.strftime("%H:%m %p")
    tstr=dtv+mt+yr+tm
    #----------------------------
    return tstr

#**********************************************************
#**********************************************************

def get_formatted_India_table(url):
    html = requests.get(url).content
    df_list = pd.read_html(html)

    #print(df_list)
    df = df_list[-1]
    #print(df)

    print('\n\tGetting data...')

    header=table_header(url)
    print('-----------------------------------')
    for i in range(len(header)):
        print('\t\t Header : ',i,' ', header[i])
    print('-----------------------------------')
    hd=header[4] # deaths
    #extract foreign cases------
    fr_str=header[2]
    if "foreign" in fr_str:
        num=[int(s) for s in fr_str.split() if s.isdigit()]
        print('Foreign cases : ',num[0])
    #--------------------------


    selected=['Name of State / UT',     header[2],'Cured/Discharged/Migrated',  hd]
    new_f = df[selected]
    new_f = new_f.rename(columns={'Name of State / UT': 'State',   header[2]: 'Confirmed <br>cases'})
    new_f = new_f.rename(columns={ 'Cured/Discharged/Migrated':'Discharged' })
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

    f=pd.read_csv("temp.csv",sep=r'\s*,\s*',encoding='ascii',engine='python', index_col ='State')

    f2 = f.transpose()
    #f2['Jammu and Kashmir'] = f2['Jammu and Kashmir']+f2['Ladakh']
    f2 = f2.rename(columns={'Total number of confirmed cases in India': 'Total'})
    output = f2.transpose()
    #new_f.reset_index()

    #output = f.drop(['Ladakh'])
    #output = output.rename(  index={'Jammu and Kashmir': 'Jammu and Kashmir, Ladakh'})
    del output['Unnamed: 0']


    output.to_csv("output.csv", index=True)
    html = output.to_html()

    ##############################################
    #print(html)
    start = '<tbody>'
    end = '</tbody>'
    s = html
    outs=s[s.find(start)+len(start):s.rfind(end)]
    #write trs to file
    text_file = open("table_tr.html", "w")
    text_file.write(outs)
    text_file.close()
    ##############################################

    header=t_head
    footer=t_foot
    table_full='table_main2.html'

    with open(table_full,'wb') as wfd:
        for f in [header,'table_tr.html',footer]:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)

    print('\ttable generated...')
    ##############################################
    ct=get_mod_string(url)

    p1='<!-- declare value --> <?php $updated="'
    p2=' " ?> <!-- ============================= --> </html> '

    end=p1+ct+p2

    with open(table_full, "a") as myfile:
        myfile.write(end)

    print('\tFinished.\n')
    print('\tTrying upload...')
    list_files = subprocess.run(["ls", "-l"])
    subprocess.call(upload_script)
    print ("\tDone.")

#**********************************************************
#**********************************************************
