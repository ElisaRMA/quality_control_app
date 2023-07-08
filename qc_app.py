import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import os

os.environ['PATH'] += ';' + r'D:\Program Files\R\R-4.0.5\\bin\Rscript'


st.header('🎈 Quality control application')


st.sidebar.markdown('''
About
Tis is a tool to allow quality control of two medicinal plants: Maytenus ilicifolia and Mikania laevigada

Athors:

First steps on how to use it:
 analyze your samples using a UHPLC-MS, ideally with the following parameters:
 transform the .raw data into mzXML via MSConvert using the code
 zip the files into a folder
 upload the folder into this application using the 'browse' button. 
 Click run and wait for the result to be shown!

''')



#path_file = os.path.exists(path2script)
#path_command = os.path.exists(command)
#st.title(path_file)
#st.title(path_command)


st.subheader('1. testing R')

uploaded_files = st.file_uploader('Choose a mzXML file', type='mzXML', accept_multiple_files=True, help='Only mzXML files fill be accepted')


xcms_xset1 = os.getcwd() + "\\xcms_xset1.R"
xcms = os.getcwd() + "\\xcms.R"

command = "D:\Program Files\R\R-4.0.5\\bin\Rscript"

if st.button('Run XCMS'):
    with st.spinner('Wait for it...'):
        process1 = subprocess.Popen([command, xcms_xset1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result1 = process1.communicate()
    st.success('Done!')

    st.write(result1) 

with st.expander('See script'):
  code1 = '''
  
  xcms

  mtcars
  '''
 
  st.code(code1, language='R')






