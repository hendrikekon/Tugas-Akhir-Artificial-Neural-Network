#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib
import cv2
import csv
import os
import glob
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras import Input
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sys import exit


#-------------------------------------------------------------------------------------------------------------------#
def menu_utama():
    print('\n\t|---------------------------------------------------------------|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t Klasifikasi Bunga ANN\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t Main Menu :\t\t\t\t\t\t|')
    print('\t|---------------------------------------------------------------|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|  [1]Ekstraksi Fitur Data Train \t\t\t\t|')
    print('\t|  [2]Ekstraksi Fitur Data Test \t\t\t\t|')
    print('\t|  [3]Lihat Dimensi Data dan Normalisasi Data \t\t\t|')
    print('\t|  [4]Buat Model ANN dan Training Program \t\t\t|')
    print('\t|  [5]Lihat Grafik Hasil Training \t\t\t\t|')
    print('\t|  [6]Lihat Hasil Prediksi Data Test dan Akurasi \t\t|')
    print('\t|  [7]Pengujian Klasifikasi \t\t\t\t\t|')
    print('\t|  [8]Exit \t\t\t\t\t\t\t|')
    
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|---------------------------------------------------------------|')
    menu = input('\t   Pilih Menu: ')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|---------------------------------------------------------------|')
    
    if menu =='1':
        data_train()
    elif menu == '2':
        data_test()
    elif menu == '3':
        normalisasi()
    elif menu == '4':
        training()
    elif menu == '5':
        grafik()
    elif menu == '6':
        prediksi_test()
    elif menu == '7':
        GUI()
    elif menu == '8':
        exit()
    else :
        print ("Angka Yang Diinputkan Salah \n")
        
    print ('----------------------------------------------------------------------\n')
    
    return menu_utama()

#-------------------------------------------------------------------------------------------------------------------#
def data_train():
    global x_train, y_train
    
    os.remove('train.csv')
    #JUDUL CSV
    Y={'R':1,'G':2,'B':3,
       'H':4,'S':5,'V':6,
       'Hasil':7}


    with open('train.csv', 'a', newline=None) as file:
        fieldnames = ['R','G','B',
                      'H','S','V',
                      'Hasil']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(Y)
    #--------------------------------------------------------------------
    # LOAD ALL IMAGES
    path = glob.glob("Dataset/Train/Data Primer/Amarylis/*.png")
    no =0

    #RGB MULAI
    for img in path:
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

        no +=1

        print('\n Data Ke ', no, ': ', R,G,B,H,S,V)

      

        #MASUKKAN CSV
        d={'R':R,'G':G,'B':B,
           'H':H,'S':S,'V':V,
           'Hasil':0}


        with open('train.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V',
                        'Hasil']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(d)

    #-----------------------------------------------------------------------------------------------------------------
    # LOAD ALL IMAGES
    path = glob.glob("Dataset/Train/Data Primer/Petunia/*.png")

    #RGB MULAI
    for img in path:
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

        no +=1

        print('\n Data Ke ', no, ': ', R,G,B,H,S,V)

    #MASUKKAN CSV
        e={'R':R,'G':G,'B':B,
           'H':H,'S':S,'V':V,
           'Hasil':1}


        with open('train.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V',
                        'Hasil']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(e)

    #--------------------------------------------------------------------------------------------------------------------

    # LOAD ALL IMAGES
    path = glob.glob("Dataset/Train/Data Sekunder/kansas/*.png")

    #RGB MULAI
    for img in path:
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

        no +=1

        print('\n Data Ke ', no, ': ', R,G,B,H,S,V)
        
    #MASUKKAN CSV
        f={'R':R,'G':G,'B':B,
           'H':H,'S':S,'V':V,
           'Hasil':2}


        with open('train.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V',
                        'Hasil']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(f)

    #-------------------------------------------------------------------------------------------------------------------

    # LOAD ALL IMAGES
    path = glob.glob("Dataset/Train/Data Sekunder/marguerite/*.png")

    #RGB MULAI
    for img in path:
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

        no +=1

        print('\n Data Ke ', no, ': ', R,G,B,H,S,V)

    #MASUKKAN CSV
        g={'R':R,'G':G,'B':B,
           'H':H,'S':S,'V':V,
           'Hasil':3}


        with open('train.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V',
                        'Hasil']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(g)
    #--------------------------------------------------------------------
    data1= pd.read_csv('train.csv')
    x_train=data1.iloc[:, 0:-1].values
    y_train=data1.iloc[:, -1].values
    print(data1.shape)
    print(data1.head())
#-------------------------------------------------------------------------------------------------------------------#
def data_test():
    global x_test, y_test
    
    os.remove('test.csv')
    #JUDUL CSV
    Z={'R':1,'G':2,'B':3,
       'H':4,'S':5,'V':6,
       'Hasil':7}


    with open('test.csv', 'a', newline=None) as file:
        fieldnames = ['R','G','B',
                      'H','S','V',
                      'Hasil']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(Z)
    #--------------------------------------------------------------------
    # LOAD ALL IMAGES
    path = glob.glob("Dataset/Test/Amarylis/*.png")
    no=0

    #RGB MULAI
    for img in path:
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

        no +=1

        print('\n Data Ke ', no, ': ', R,G,B,H,S,V)

    #MASUKKAN CSV
        h={'R':R,'G':G,'B':B,
           'H':H,'S':S,'V':V,
           'Hasil':0}


        with open('test.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V',
                        'Hasil']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(h)

    #--------------------------------------------------------------------------------------------------------------------

    # LOAD ALL IMAGES
    path = glob.glob("Dataset/Test/Petunia/*.png")

    #RGB MULAI
    for img in path:
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

        no +=1

        print('\n Data Ke ', no, ': ', R,G,B,H,S,V)

    #MASUKKAN CSV
        m={'R':R,'G':G,'B':B,
           'H':H,'S':S,'V':V,
           'Hasil':1}


        with open('test.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V',
                        'Hasil']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(m)

    #----------------------------------------------------------------------------------------------------------------

    # LOAD ALL IMAGES
    path = glob.glob("Dataset/Test/Kansas/*.png")

    #RGB MULAI
    for img in path:
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

        no +=1

        print('\n Data Ke ', no, ': ', R,G,B,H,S,V)

    #MASUKKAN CSV
        n={'R':R,'G':G,'B':B,
           'H':H,'S':S,'V':V,
           'Hasil':2}


        with open('test.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V',
                        'Hasil']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(n)

    #--------------------------------------------------------------------------------------------------------------------

    # LOAD ALL IMAGES
    path = glob.glob("Dataset/Test/Marguerite/*.png")

    #RGB MULAI
    for img in path:
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

        no +=1

        print('\n Data Ke ', no, ': ', R,G,B,H,S,V)

    #MASUKKAN CSV
        o={'R':R,'G':G,'B':B,
           'H':H,'S':S,'V':V,
           'Hasil':3}


        with open('test.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V',
                        'Hasil']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(o)
    #--------------------------------------------------------------------
    data2= pd.read_csv('test.csv')
    x_test=data2.iloc[:, 0:-1].values
    y_test=data2.iloc[:, -1].values
    print(data2.shape)
    print(data2.head())
#-------------------------------------------------------------------------------------------------------------------#
def normalisasi():
    global x_train, x_test, sc
    print("Dimensi data :\n")
    print("X train \t X test \t Y train \t Y test")  
    print("%s \t %s \t %s \t %s" % (x_train.shape, x_test.shape, y_train.shape, y_test.shape))
    print('y test:', y_test)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
#-------------------------------------------------------------------------------------------------------------------#
def training():
    global ann, history
    ann = Sequential()
    ann.add(Input(shape=(6,)))
    ann.add(Dense(5, activation='sigmoid'))
    ann.add(Dense(4, activation='sigmoid'))
    ann.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])
    history = ann.fit(x_train,y_train, batch_size= 20, epochs=600)
    ann.save('ann_model.h5')
#-------------------------------------------------------------------------------------------------------------------#
def grafik():
    plt.plot(history.history['accuracy'])
    plt.title('model_accuracy')

    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('model_loss ')

    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()
#-------------------------------------------------------------------------------------------------------------------#
def prediksi_test():
    global y_pred
    data1= pd.read_csv('train.csv')
    x_train=data1.iloc[:, 0:-1].values
    y_train=data1.iloc[:, -1].values
    data2= pd.read_csv('test.csv')
    x_test=data2.iloc[:, 0:-1].values
    y_test=data2.iloc[:, -1].values
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    ann = load_model('ann_model.h5')
    y_pred = ann.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    print('confusion matrix Data Test= \n' , confusion_matrix(y_test, y_pred))
    print(' \n')
    print('Prediksi Data Test:\t\t',y_pred)
    print('Hasil Sebenarnya Data Test:\t',y_test)
    print(' \n')
    print('Accuracy = ' , accuracy_score(y_test, y_pred)*100, '%')
    print('classification report: \n', classification_report(y_test, y_pred))
#-------------------------------------------------------------------------------------------------------------------#
def GUI():
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    from PIL import Image, ImageTk

    #--------------------------------------------------------------------------------------------------------------------------#
    def open_file():

        """Open a file for editing."""
        global filepath

        filepath = askopenfilename(

            filetypes=[("Images", "*.png")]

        )
        if filepath:
            pict = Image.open(filepath)
            image = pict.resize((450, 350), Image.ANTIALIAS)
            gambar = ImageTk.PhotoImage(image)
            lblImage.config(image=gambar)
            lblImage.image = gambar

    #--------------------------------------------------------------------------------------------------------------------------#    
    def ekstraksi():


        global hasilekstraksi
        img=filepath
        img_bgr = cv2.imread(img)
        B = np.mean(img_bgr[:,:,0])
        G = np.mean(img_bgr[:,:,1])
        R = np.mean(img_bgr[:,:,2])


                #RGB AKHIR

                #HSV MULAI
        rn= R/(R+G+B)
        gn= G/(R+G+B)
        bn= B/(R+G+B)

        V= max(rn,bn,gn)
        VMIN= min(rn,bn,gn)

        if V==0:
          S=0
        else:
          S= V-(VMIN / V)


        if S==0:
          H=0
        elif V==rn:
          H= (60 * (gn-bn)) / (S*V)
        elif V==gn:
          H= 60 * (2 + ((bn-rn) / (S*V)))
        elif V==bn:
          H= 60 * (4 + ((rn-gn) / (S*V)))


        if H<0:
          H= H+360

    #MASUKKAN CSV
        #JUDUL CSV
        Y={'R':1,'G':2,'B':3,
           'H':4,'S':5,'V':6}
        with open('output.csv', 'a', newline=None) as file:
            fieldnames = ['R','G','B',
                          'H','S','V']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(Y)

        #MASUKKAN DATA
        hasilekstraksi={'R':R,'G':G,'B':B,
                        'H':H,'S':S,'V':V}

        with open('output.csv', 'a', newline=None) as file:
          fieldnames = ['R','G','B',
                        'H','S','V']
          writer = csv.DictWriter(file, fieldnames=fieldnames)
          writer.writerow(hasilekstraksi)
        
        menutxt.set("Ekstraksi Selesai")

    #--------------------------------------------------------------------------------------------------------------------------#
    def klasifikasi():

        data= pd.read_csv('output.csv')
        z_test=data.iloc[:, 0:6].values
        data1= pd.read_csv('train.csv')
        x_train=data1.iloc[:, 0:-1].values
        y_train=data1.iloc[:, -1].values
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        z_test = sc.transform(z_test)
        
        ann = load_model('ann_model.h5')
        z_pred = ann.predict(z_test)
        a_pred = np.argmax(z_pred, axis=1)

        if a_pred == 0:
            menutxt.set("Amarylis")
            os.remove('output.csv')
        elif a_pred == 1:
            menutxt.set("Petunia")
            os.remove('output.csv')
        elif a_pred == 2:
            menutxt.set("Kansas")
            os.remove('output.csv')
        elif a_pred == 3:
            menutxt.set("Marguerite")
            os.remove('output.csv')
        else:
            os.remove('output.csv')
    
    
    def reset():
        os.remove('output.csv')

    #--------------------------------------------------------------------------------------------------------------------------#
    window = tk.Tk()
    # # Code to add widgets will go here...

    window.title("GUI")
    window.geometry("500x500")

    lblImage = tk.Label(window, text='Image Preview')
    lblImage.pack()

    btn_open = tk.Button(window, text="Open", command=open_file).place(x = 40, y = 380)

    btn_eks = tk.Button(window, text="Ekstraksi Ciri", command=ekstraksi).place(x = 220, y = 380)

    btn_kelas = tk.Button(window, text="Klasifikasi", command=klasifikasi).place(x = 390, y = 380)
    
    btn_reset = tk.Button(window, text="Reset Ekstrasi", command=reset).place(x = 40, y = 420)

    lblnama = tk.Label(window, text="Nama Bunga: ").place(x = 180, y = 420)
    menutxt = tk.StringVar()
    lblkelas = tk.Label(window, textvariable=menutxt).place(x = 310, y = 420)

    window.mainloop()
#-------------------------------------------------------------------------------------------------------------------#
#Program Utama
print ('\n\t Tugas Akhir')
print ('\n')
print('\t Nama: Hendrik Eko Nurdianto')
print('\t NIM: 5160411031')
print ('\t----------------------------------------------------------------')
menu_utama()


# In[ ]:





# In[ ]:




