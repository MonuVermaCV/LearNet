#!/usr/bin/env python3
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
import LEARNet_Model
model =LEARNet_Model.build(height=112,width=112,channels=3,classes =8)
model.summary()
