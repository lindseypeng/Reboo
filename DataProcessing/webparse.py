from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

def download_web(url):#amazon uses lxml    
    r=requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")##do unit text
    return soup
    
def find_description(soup):
    soup2=soup.find_all(id="descriptionContainer")## some book have no description:error comes from here
    soupstr=str(soup2)
    text=soupstr.split("<a data-text-id")[0]
    text=text.split("freeTextContainer")[1]
    return text
#remove all tags and clean abit for goodreads
def remove_tags_for_description(text):
    tagswhere=re.compile('<.*?>')
    clean=re.sub(tagswhere,'',text)
    backwhere=re.compile('\n')
    cleaner=re.sub(backwhere,'',clean)
    final=cleaner.split(">")[1]
    return final

def find_genre(soup):
    soupstr=str(soup)
    text=soupstr.split('googletag.pubads().setTargeting("shelf", ["')[1]
    text2=text.split('googletag.pubads().setTargeting("gtargeting"')[0]
    textgenre=text2.split(']);\n ')[0]
    return textgenre
    #
def list_genre(textgenre):
    tagswhere=re.compile('"')
    clean=re.sub(tagswhere,'',textgenre)
    listofgenre=clean.split(",")
    return listofgenre
#
def make_frame(text,bookno,bookname,listofgenre):
    book=pd.DataFrame( {'Book#':bookno, "Book_Title":bookname,'Description':text,"Genres":[listofgenre]},index=[str([i])])
    return book   

import numpy as np
booklist=pd.read_csv("/home/lindsey/insightproject/url_Difference.csv")
urls=booklist['URL']
booknos=booklist['Book_#']
booknames=booklist['Book_Name']
texttest=pd.DataFrame()
for i in np.arange(len(urls)-1):
    url=urls[i]
    bookno=booknos[i]
    bookname=booknames[i]
    soup=download_web(url)
    text=find_description(soup)
    final=remove_tags_for_description(text)
    textgenre=find_genre(soup)
    listofgenre=list_genre(textgenre)
    book=make_frame(final,bookno,bookname,listofgenre)
    texttest=texttest.append(book,ignore_index=True) 
    
#texttest.to_csv("/home/lindsey/Desktop/bookdescriptionfromgoodreadswithgenres.csv")
