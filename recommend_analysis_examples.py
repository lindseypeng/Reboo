#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:00:03 2019

@author: lindsey
"""
import pandas as pd
from bert_emb_modul import mastercode as ms
description="In this landmark book, Scott Page redefines the way we understand ourselves in relation to one another. \
The Difference is about how we think in groups--and how our collective wisdom exceeds the sum of its parts. \
Why can teams of people find better solutions than brilliant individuals working alone? And why are the best group \
decisions and predictions those that draw upon the very qualities that make each of us unique? The answers lie \
in diversity--not what we look like outside, but what we look like within, our distinct tools and abilities."
method=0
obj=ms(description,method)

##get recommendation based o n closest neighbors##
result,avgs,current=obj.master()
n_neighbors=5
suggestion=obj.output_neighbors(n_neighbors)
recommendation=obj.recommend_closest_neighbor(suggestion,result)
Closest_recomm=obj.clean_rec_close_neighbor(recommendation)


##get recommendation based on cluster method gaussian##
numberofclusters=20
result_labels,current=obj.gmm_cluster(n_components=numberofclusters)
similarbooks=obj.cluster_neighbor(result_labels,current)
booktitles,recommend=obj.recommend_cluster_nei(similarbooks)



def cleanstrings(list1):
    list2=str(list1).replace('[','').replace(']','')
    list3=list2.replace('"\'','').replace('\'"','')
    return list3


##print out recommendation for closest neighbor iterative
def print_close_recommend(Closest_recomm):
    booktitles=[]
    Sens=[]
    for j,row in Closest_recomm.iterrows():
        booktitle=str(row['booktitle'])
        booktitle_clean=cleanstrings(booktitle)
        booktitles.append(booktitle_clean)
        sentences=str(row['book_sentences'])
        sentences_clean=cleanstrings(sentences)
        Sens.append(sentences_clean)
    return booktitles, Sens


def print_cluster_recommend(recommend):
    for j,row in recommend.iterrows():
        booktitle=str(row['booktitles'])
        sentences=str(row['book_sentences'])
        print('Title:{} \n {}'.format(cleanstrings(booktitle),cleanstrings(sentences)))
        
#for j, row in clean.iterrows():
#    booktitle=str(row['booktitle'])
#    sentences=str(row['book_sentences'])
#    print('title{}:{}'.format(booktitle,sentences))
