# -*- coding: utf-8 -*-
"""
%This is an code for CNN architecture used in LEARNet: Dyanamic imagingnetwork for micro expression recognition
%{@autors: Verma, M., Vipparthi, S.K., Singh, G. and Murala, S., 2019. 
%Title: LEARNet Dynamic Imaging Network for Micro Expression Recognition.
%Publication: IEEE Transaction on image Processinga, 2019
%This code is written by Monu Verma @ CSE, MNIT, Jaipur (INDIA)
%For any question you can contact us at monuverm.cv@gmail.com
%For more details you can also visit us: https://visionintelligence.github.io/

%----specify size of input-------

@author: Monu Verma
"""
from keras.models import Model
from keras.layers import Input,concatenate,Flatten,Dense,add,BatchNormalization,Dropout
from keras.layers.convolutional import Conv2D


def build(height=112,width=112,channels=3,classes =8):

    im =Input(shape=(112,112,3))
    Conv_S=Conv2D(16, (3,3), activation='relu', padding='same', strides=2, name='Conv_S')(im)
    #-------------------------------------------------------------------
    
    Conv_1_1=Conv2D(16, (1,1), activation='relu', padding='same', strides=2, name='Conv_1_1')(Conv_S)
    Conv_1_2=Conv2D(32, (3,3), activation='relu', padding='same', strides=2, name='Conv_1_2')(Conv_1_1)
    Conv_1_3=Conv2D(64, (5,5), activation='relu', padding='same', strides=2, name='Conv_1_3')(Conv_1_2)
    #------------------------------------------------------------------
    
    Conv_2_1=Conv2D(16, (1,1), activation='relu', padding='same', strides=2, name='Conv_2_1')(Conv_S)
    add_2_1=add([Conv_1_1, Conv_2_1])
    batch_r11=BatchNormalization()(add_2_1)
    Conv_2_2=Conv2D(32, (3,3), activation='relu', padding='same', strides=2, name='Conv_2_2')(batch_r11)
    add_2_2=add([Conv_1_2, Conv_2_2])
    batch_r12=BatchNormalization()(add_2_2)
    Conv_x_2=Conv2D(64, (5,5), activation='relu', padding='same', strides=2, name='Conv_x_2')(batch_r12)
    #------------------------------------------------------------------
    
    Conv_3_1=Conv2D(16, (1,1), activation='relu', padding='same', strides=2, name='Conv_3_1')(Conv_S)
    Conv_3_2=Conv2D(32, (3,3), activation='relu', padding='same', strides=2, name='Conv_3_2')(Conv_3_1)
    Conv_3_3=Conv2D(64, (5,5), activation='relu', padding='same', strides=2, name='Conv_3_3')(Conv_3_2)
    #------------------------------------------------------------------
    
    Conv_4_1=Conv2D(16, (1,1), activation='relu', padding='same', strides=2, name='Conv_4_1')(Conv_S)
    add_4_1=add([Conv_3_1, Conv_4_1])
    batch_r13=BatchNormalization()(add_4_1)
    Conv_4_2=Conv2D(32, (3,3), activation='relu', padding='same', strides=2, name='Conv_4_2')(batch_r13)
    add_4_2=add([Conv_3_2, Conv_4_2])
    batch_r14=BatchNormalization()(add_4_2)
    Conv_x_4=Conv2D(64, (5,5), activation='relu', padding='same', strides=2, name='Conv_x_4')(batch_r14)
   
    #--------------------------------------------------------
    concta1=concatenate([Conv_1_3, Conv_x_2, Conv_3_3, Conv_x_4])
    batch_X=BatchNormalization()(concta1)
    
    #-----------------------------------------------------#    
    Conv_5_1=Conv2D(256, (3,3), activation='relu', padding='same', strides=2, name='Conv_5_1')(batch_X)
    #-----Fully Connected layer--------
    F1=Flatten()(Conv_5_1)
    FC1=Dense(256,activation='relu')(F1)
    drop=Dropout(0.5)(FC1)
    
    #------clasisfication layer-------
    
    out = Dense(classes, activation='softmax')(drop)
    
    model = Model(inputs=[im],outputs= out)
    return model
    
    
    
    

