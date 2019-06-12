#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:13:37 2019

@author: lindsey
"""
#############NEED MASSIVE CHANGE##
from flask import Flask, render_template, request
from bert_emb_modul import mastercode as ms

# Create the application object
app = Flask(__name__)

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


@app.route('/',methods=["GET","POST"])
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output', methods=["POST"])
def tag_output():
#       
       # Pull input
       some_input =request.form.get('user_input')     
       # Case if empty
       if some_input == '':
           return render_template("index.html",
                                  my_input = some_input,
                                  my_form_result="Empty")
       else:
           description=some_input##methods with tfidf
           method=0
           obj=ms(description,method)
           result,avgs,current=obj.master()
           obj.plot()
           n_neighbors=5
           suggestion=obj.output_neighbors(n_neighbors)
           recommendation=obj.recommend_closest_neighbor(suggestion,result)
           Closest_recomm=obj.clean_rec_close_neighbor(recommendation)
           booktitles, Sens=print_close_recommend(Closest_recomm)
           a=zip(booktitles, Sens)
           some_output=a
           some_output2=3
           some_image="myfigure.png"
           return render_template("index.html",
                              my_input=some_input,
                              my_output=some_output,
                              my_number=some_output2,
                              my_img_name=some_image,
                              my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/
