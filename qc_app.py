import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import glob
import io
import pickle
import time
from tabulate import tabulate
from sklearn.svm import SVC
#from sklearn import metrics

#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import PredefinedSplit
#from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------- Python pipeline functions ----------- #
st.set_page_config(layout="wide")


import subprocess
import os

@st.cache_resource
def install_bioc_packages():
    # Create a directory your app can write to
    os.makedirs("R_libs", exist_ok=True)
    
    r_code = """
    # Set a custom library path
    .libPaths(c("./R_libs", .libPaths()))
    
    # Install BiocManager if not present
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = "./R_libs")
    }
    
    # Use Bioconductor version 3.12, which is compatible with R 4.0.x
    BiocManager::install(version = "3.12", lib = "./R_libs")
    
    # Install xcms version 3.10.1
    BiocManager::install("xcms", version = "3.10.1", lib = "./R_libs", update = FALSE, ask = FALSE)
    
    # Install CAMERA version 1.44.0
    BiocManager::install("CAMERA", version = "1.44.0", lib = "./R_libs", update = FALSE, ask = FALSE)
    
    # Verify installations
    packageVersion("xcms")
    packageVersion("CAMERA")
    """
    
    result = subprocess.run(["Rscript", "-e", r_code], capture_output=True, text=True)




@st.cache_resource
def rounder(dataframe):
    '''
    
    Round the values for the columns mz, mzmin, mzmax, rt, rtmin and rtmax of the 
    Dataframe passed.
    
    
    Parameters
    ----------
    dataframe : pandas DataFrame, default None
        DataFrame object with mz, mzmin, mzmax, rt, rtmin and rtmax columns to round.
    
    '''
    
    table = dataframe 
    
    table.mz = table.mz.round(0).astype(int)
    table.mzmin = table.mzmin.round(0).astype(int)
    table.mzmax = table.mzmax.round(0).astype(int)
    
    table.rt = table.rt.div(60).round(1)
    table.rtmin = table.rtmin.div(60).round(1)
    table.rtmax = table.rtmax.div(60).round(1)
    
    return table

@st.cache_resource
def feature_correspondance (ref_data, target_data):
    
    '''
    
    Generates the feature names on the "mz_rt" pattern on the target_data based on the 
    feature names on ref_data.
    
    The columns mz, mzmin, mzmax, rt, rtmin and rtmax on ref_data are used as reference to 
    create the feature names, by comparing the values between columns.
    
    First, the mz values on the target data are tested against the range of mzmin and mzmax on 
    the ref_data, line by line. If a correspondance is found, the testing moves forward.
    
    Next, the testing happens between rt, rtmin and rtmax on target data against the rtmin and 
    rtmax on the ref_data. If a correspondance is found, the feature name on ref_data is applied 
    to such row on the target_data. 
    
    Before everything, the data is ordered by the npeaks columns so that, in case of +1 correspondance, 
    only the feature with higher npeaks is used.
    
    
    Parameters
    ----------
    ref_data : pandas DataFrame, default None
        DataFrame object with mz, mzmin, mzmax, rt, rtmin and rtmax columns to use as reference
        and a feature columns with the feature names in the pattern mz_rt.
    
    target_data : pandas DataFrame, default None
        DataFrame object with mz, mzmin, mzmax, rt, rtmin and rtmax columns to round.
    
    '''
    
    # column to be populated
    target_data['features'] = np.nan

    # sorts the values in order to use the feature name of  features with higher npeaks.
    # sometimes, the preprocessing generated 'duplicates' of almost the same feature. 
    # We ignore those later on
    target_data = target_data.sort_values('npeaks', ascending=False,ignore_index=True)
    ref_data = ref_data.sort_values('npeaks', ascending=False,ignore_index=True)

    for i in range(len(target_data)):
        for j in range(len(ref_data)):

            # check if mz is in right range.
            if ((target_data.loc[i,'mz'] <= ref_data.loc[j,'mzmax']) 
                  & (target_data.loc[i,'mz'] >= ref_data.loc[j,'mzmin'])):

                # if it is, then proceed. Any rt value from the target data must be in the range of 
                # rtmax and rtmin of the reference data:

                if (
                    ((target_data.loc[i,'rt'] <= ref_data.loc[j,'rtmax']) 
                      & (target_data.loc[i,'rt'] >= ref_data.loc[j,'rtmin'])) or

                   ((target_data.loc[i,'rtmin'] <= ref_data.loc[j,'rtmax']) 
                      & (target_data.loc[i,'rtmin'] >= ref_data.loc[j,'rtmin'])) or

                   ((target_data.loc[i,'rtmax'] <= ref_data.loc[j,'rtmax']) 
                    & (target_data.loc[i,'rtmax'] >= ref_data.loc[j,'rtmin']))
                ):

                    # if the condition is met, then the FIRST feature name is extracted and given 
                    # to the target data.
                    # The one with higher npeak will be used, due to the sorting at the beggining
                    target_data.loc[i,'features'] = ref_data.loc[j,'features']
                    
                # this brakes the second if statement. There could be multiple matches on rt 
                break
                
    return target_data

@st.cache_resource
def data_cleaning(ref_data, target_data):
        
    '''
    
    The process that creates the feature can generate duplicate names. The filtering is done by sorting the
    dataframes based on the npeaks columns and  dropping the duplicates, keeping the one with the higher npeaks
    
    The function also drops unnecessary columns such as 'mz', 'mzmin', 'mzmax', 'rt','rtmin', 'rtmax', 
    'npeaks','NEG_GROUP' and 'POS_GROUP' at the end of the process.
    
    
    Parameters
    ----------
    ref_data : pandas DataFrame, default None
        DataFrame object used as reference on the feature_correspondance function.
    
    target_data : pandas DataFrame, default None
        DataFrame object that passed trough the feature_correspondance function.
    
    '''   
        
    # the removal is based on the npeaks column. The feature with more npeaks, is kept.
    target_data = target_data.sort_values('npeaks', ascending=False).drop_duplicates('features').sort_index()
    ref_data = ref_data.sort_values('npeaks', ascending=False).drop_duplicates('features').sort_index()

    # dropping unnecessary columns
    target_data = target_data.drop(['mz', 'mzmin', 'mzmax', 'rt', 
                                  'rtmin', 'rtmax', 'npeaks'], axis=1)

    ref_data = ref_data.drop(['mz', 'mzmin', 'mzmax', 'rt', 
                                      'rtmin', 'rtmax', 'npeaks','NEG_GROUP', 'POS_GROUP'], axis=1)
    
    return ref_data, target_data

@st.cache_resource
def data_prep(ref_data,target_data):
        
    '''
    
    During the process that creates the feature (using feature_correspondance function), 
    the ref_data is used to create the feature names on target_data. However, the target_data
    might not have, for a given feature on ref_data, a good correspondance. In such cases, the 
    feature name is set to NAN. These features need to be dropped. 
    
    For the ref_data, then, some feature_names won't appear in target_data and since both need to
    contain the same feautures, these also need to be dropped.
    
    The purpose of the function is to make both datasets equal in terms of features. The same number 
    and types of features need to appear in both datasets as the ref_data was used for the training 
    and the target_data will go trough the prediction steps.
    
        
    Parameters
    ----------
    ref_data : pandas DataFrame, default None
        DataFrame object used as reference on the feature_correspondance function.
    
    target_data : pandas DataFrame, default None
        DataFrame object that passed trough the feature_correspondance function.
    
    '''   
    
    
    # val set might have some feature that don't fit in any range - their feature names will be nan, so need to remove
    # train might have some features that wont appear in the val. So, create them in val and set them to zero. 
    # first, set index on both to be the features, so its possible to do that.
    
    ref_data= ref_data.set_index('features')
    target_data = target_data.dropna().set_index('features') # dropping na and making feature as index
    
    # set method to get the set of index values that are unique 
    # subtracting the sets to get the different indexes. 
    # concat method to concatenate train and val
    # filling the missing values on the concatenation with 0 using the fillna method.

    unique_indexes = list(set(ref_data.index) - set(target_data.index))
    target_data = pd.concat([target_data, pd.DataFrame(index=unique_indexes, columns=target_data.columns)], sort=True).fillna(0)

    # order both val and train features equally
    # sort the features - the model needs them at the same sequence
    ref_data = ref_data.reset_index().sort_values(by='features')
    target_data = target_data.reset_index().sort_values(by='index')    
    
    return target_data

@st.cache_data
def load_model_maytenus():
    '''
    
    model_path: path to model file. Eg: 'C:/Users/name/Documents/dev/model.pkl'
    
    '''
    pickled_model = pickle.load(open('model_maytenus.pkl', 'rb'))
    return pickled_model

@st.cache_data
def load_model_mikania():
    '''
    
    model_path: path to model file. Eg: 'C:/Users/name/Documents/dev/model.pkl'
    
    '''
    pickled_model = pickle.load(open('model_mikania.pkl', 'rb'))
    return pickled_model

@st.cache_resource
def load_refdata_mikania():
    '''
    returns the data used for training. It will be a reference data to create the feature names of input data.
    '''
    return pd.read_csv('ref_data_mikania.csv')

@st.cache_resource
def load_refdata_maytenus():
    '''
    returns the data used for training. It will be a reference data to create the feature names of input data.
    '''
    return pd.read_csv('ref_data_maytenus.csv')

# ----------- objects to run locally ----------- #

# add Rscript into path variable
os.environ['PATH'] += ';' + r'D:\Program Files\R\R-4.0.5\\bin\Rscript'

# ----------- App ----------- #

st.header('MedPlant-AI: an AI-based Quality Control tool for Medicinal Plants')

st.sidebar.markdown('''

Welcome to the MedPlant-AI!

                    
This is an AI-driven tool designed for quality control of two medicinal plants: *Maytenus ilicifolia* and *Mikania laevigata*.
The purpose of MedPlant-AI is to allow the analysis of multiple samples at once, so that the quality control process becomes faster and more reliable. 
This tool is meant to be used after the metabolomics analysis of the plant material using UHPLC-MS, on the mzXML data converted. 

                                              

How to use MedPlant-AI:                                     


1. Analyze your samples with replicates, using an HPLC-MS equipment, ideally using the method especified on the paper or on the GitHub repository [link]. 
For the extractions, follow the Brazillian Fitoterapy formulary, in which aqueous extract is instructed for *Maytenus ilicifolia* and hydroethanolic extract is instructed for *Mikania laevigata*.
For the chromatographic method, the most important aspect is the ionization and detection mode. Use only the negative ionization mode and the 'centroid' detection method. 

                    
2. Convert the .raw data into mzXML via msconvert, a command line tool. The code for the transformation is on the GitHub repository.

                    
3. Split the converted mzXML into folders, one for each sample and its replicates. Then, place these folder into a directory, zip this directory and upload it to the MedPlant-AI.
st.sidebar.image                    
                   

''')

st.sidebar.info('''To understand more about this tool and how to use it, we recommend you to access the GitHub [repository](https://github.com/ElisaRMA/quality_control_app).''')

st.subheader('1. Preprocessing metabolomics data')

option = st.selectbox(
    'Select the species to perform quality control:',
    ('Maytenus ilicifolia', 'Mikania laevigata'))

#st.write(option)


# files need to be in a zipped folder
uploaded_files = st.file_uploader('Choose a zipped folder with subfolders for each sample. Each file also needs to be in the mzXML format.', type='zip', accept_multiple_files=False, help='Only rar files are accepted')

# to get the xcms R scripts
xcms_may = os.getcwd() + "/xcms_may.R"
xcms_mik = os.getcwd() + "/xcms_mik.R"

# R.exe on local machine
#command = "D:\Program Files\R\R-4.0.5\\bin\Rscript"

# object placeholder to tbe filled later
output_folder = "output"
output_folder_mik = "output_mik"

if st.button('Run XCMS') and uploaded_files is not None:
    if option == 'Maytenus ilicifolia':
    # Read the uploaded zip folder
        zip_file_bytes = uploaded_files.getvalue()
    
    # loading symbol 
        with st.spinner('Please wait ...'):
        
        # Create the output folder, unzips content to it, runs r script
            os.makedirs(output_folder, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(zip_file_bytes), 'r') as zip_ref:
                zip_ref.extractall(output_folder)
        
            install_bioc_packages()
            process1 = subprocess.run(["Rscript", xcms_may], stdout=subprocess.PIPE, cwd=output_folder)
            st.write(process1.stdout)

        st.success('Done! This is the data that will be used for the Machine Learning model:')

    # Find the generated CSV file
        csv_files = glob.glob(os.path.join(output_folder, '*.csv'))
        if len(csv_files) > 0:
            csv_file = csv_files[0]
            input_data = pd.read_csv(csv_file, index_col=[0])

        # gets the folder names to delete the columns on the csv
            subfolder_names = [name for name in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, name))]

            for subfolder_name in subfolder_names:
                st.write(subfolder_name)
                subfolder_path = os.path.join(output_folder, subfolder_name)
                folder_names = [name for name in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, name))]

        # stores the data in session and drops the folder names from the df
            st.session_state.input_data = input_data.drop(folder_names,axis=1)
            st.dataframe(input_data.drop(folder_names,axis=1))  

####### fix! download does not have the data and it deletes the rest of the output #######
            st.download_button(
            "Download CSV",
            csv_file,
            os.path.basename(csv_file),
            "text/csv",
            key='download-csv'
            )
        else:
            st.warning('No CSV file found in the output.')
    else:
            # Read the uploaded zip folder
        zip_file_bytes = uploaded_files.getvalue()

    # loading symbol 
        with st.spinner('Please wait ...'):
        
        # Create the output folder, unzips content to it, runs r script
            os.makedirs(output_folder_mik, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(zip_file_bytes), 'r') as zip_ref:
                zip_ref.extractall(output_folder_mik)
        
            process1 = subprocess.run(["Rscript", xcms_mik], stdout=subprocess.PIPE, cwd=output_folder_mik)
    
        st.success('Done! This is the data that will be used for the Machine Learning model:')

    # Find the generated CSV file
        csv_files = glob.glob(os.path.join(output_folder_mik, '*.csv'))

        if len(csv_files) > 0:
            csv_file = csv_files[0]
            input_data = pd.read_csv(csv_file, index_col=[0])

        # gets the folder names to delete the columns un the csv
            subfolder_names = [name for name in os.listdir(output_folder_mik) if os.path.isdir(os.path.join(output_folder_mik, name))]

            for subfolder_name in subfolder_names:
            #st.write(subfolder_name)
                subfolder_path = os.path.join(output_folder_mik, subfolder_name)
                folder_names = [name for name in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, name))]

        # stores the data in session and drops the folder names from the df
            st.session_state.input_data = input_data.drop(folder_names,axis=1)
            st.dataframe(input_data.drop(folder_names,axis=1))  

####### fix! download does not have the data and it deletes the rest of the output #######
            st.download_button(
            "Download CSV",
            csv_file,
            os.path.basename(csv_file),
            "text/csv",
            key='download-csv'
            )
        else:
            st.warning('No CSV file found in the output.')

with st.expander('See xcms script'):
    if option == 'Maytenus ilicifolia':
        code1 = '''
  
library(xcms)
library(CAMERA)

xset <- xcmsSet( 
        method   = "matchedFilter",
        fwhm     = 29.4,
        snthresh = 16.1595968, #16.1595968
        step     = 1,
        steps    = 12,
        sigma    = (29.4/2.3548), #12.4851367419738,
        max      = 5,
        mzdiff   = -11, # -11 WAS THE STANDARD
        index    = FALSE)

xset2 <- retcor( 
        xset,
        method         = "obiwarp",
        plottype       = "none",
        distFunc       = "cor_opt",
        profStep       = 1,
        response       = 1,
        gapInit        = 0.26,
        gapExtend      = 2.1,
        factorDiag     = 2,
        factorGap      = 1,
        localAlignment = 0)

xset3 <- group( 
        xset2,
        method  = "density",
        bw      = 50,
        mzwid   = 1,
        minfrac = 0.1,
        minsamp = 1,
        max     = 50)

xset4 <- fillPeaks(xset3)

an <- xsAnnotate(xset4)

anF <- groupFWHM(an, perfwhm = 0.6)

anI <- findIsotopes(anF, mzabs=0.01)

anIC <- groupCorr(anI, cor_eic_th=0.1)

anFA <- findAdducts(anIC, polarity="negative")

write.csv(getPeaklist(anIC), file="data.csv") # generates a table of features
'''
        st.code(code1, language='R')
    else:
        code2 = ''' 
library(xcms)
library(CAMERA)

xset <- xcmsSet( 
        method   = "matchedFilter",
        fwhm     = 28,#7.5
        snthresh = 3,
        step     = 1, 
        steps    = 6,
        sigma    = 3.18498386274843,
        max      = 3,# 5
        mzdiff   = 1, 
        index    = FALSE)

xset2 <- retcor( 
        xset,
        method         = "obiwarp",
        plottype       = "none",
        distFunc       = "cor_opt",
        profStep       = 1,
        response       = 1,
        gapInit        = 0,
        gapExtend      = 2.7,
        factorDiag     = 2,
        factorGap      = 1,
        localAlignment = 0)

xset3 <- group( 
        xset2,
        method  = "density",
        bw      = 22,
        mzwid   = 1,
        minfrac = 0.3,
        minsamp = 1,
        max     = 50)

an <- xsAnnotate(xset4)

anF <- groupFWHM(an, perfwhm = 0.6)

anI <- findIsotopes(anF, mzabs=0.01)

anIC <- groupCorr(anI, cor_eic_th=0.75)

anFA <- findAdducts(anIC, polarity="negative")

write.csv(getPeaklist(anIC), file='data.csv') # generates a table of features
 '''
        st.code(code2, language='R')

st.subheader('2. Sample classification')

# checkbox / selector on the species
# if one is selected, run the code below, 
# if another is selected, run the other code


if option == 'Maytenus ilicifolia':
    if st.button('Run Machine Learning Prediction for Maytenus ilicifolia'):

        if 'input_data' in st.session_state:
            with st.spinner('Please wait...'):
                time.sleep(3)
                input_data = st.session_state.input_data
                st.dataframe(input_data)

# feature engineering pipeline
                ref_training_data = pd.read_csv('ref_data_maytenus.csv')
                input_data_rounded = rounder(input_data.drop(['isotopes', 'adduct','pcgroup'], axis=1))
                input_data_feat = feature_correspondance(ref_training_data, input_data_rounded)
                ref_data,input_data_clean = data_cleaning(ref_training_data,input_data_feat)
                input_data_prep = data_prep(ref_training_data,input_data_clean)
                input_data_prep = input_data_prep.set_index('index').T

# feature selection
#                features = ['830_291.0', '831_278.4', '739_256.6', '832_275.7', '833_291.0', '561_236.1', '688_300.8', '756_241.1', '593_276.4', '525_290.2',
# '691_300.9', '691_299.4', '834_230.7', '579_224.5', '690_299.8', '533_250.6', '289_226.5', '836_273.8', '564_239.2', '851_228.7', '195_44.1', '463_281.8', 
# '577_177.9', '610_261.6', '464_252.1', '560_299.9', '273_260.2',  '272_256.7', '526_360.5', '849_190.0', '740_259.2', '517_326.3', '453_221.5', '353_222.5', 
# '833_166.8', '210_47.9', '561_208.6', '548_256.2', '344_329.6', '545_257.3', '439_374.3', '594_271.9', '191_222.3', '579_225.8', '192_47.4', '410_366.5', 
# '515_327.3', '368_264.2', '193_46.8', '562_259.5', '194_45.3', '423_409.8', '393_165.1', '578_221.3', '452_222.3', '902_221.9', '574_146.2', '329_390.6',
# '219_44.1', '118_109.9', '610_259.2', '393_174.8', '305_248.6', '612_308.9', '221_44.6', '220_44.4', '178_42.8', '848_182.7', '222_45.2',
# '691_433.7', '918_205.7', '181_42.1', '303_47.8', '293_312.1', '336_227.5', '396_178.4', '487_396.1', '727_361.7', '547_256.5', '707_222.0', '274_261.2', '835_202.6',
# '479_229.8', '257_328.2', '293_443.9', '378_43.7', '164_211.0', '778_255.7', '690_432.3', '245_247.7', '294_443.5', '692_432.8', '381_43.3', 
# '290_195.7', '483_366.5', '725_314.3', '866_204.8', '329_192.2', '295_441.3', '217_42.8', '382_43.7', '652_283.6', '428_314.9', '209_46.9', 
# '328_382.3', '704_436.3', '311_431.5', '278_43.9', '850_216.6', '429_44.1', '277_43.1', '705_436.4', '335_229.4', '538_43.7', '379_43.1',
#  '476_47.3', '234_48.4', '669_437.4', '203_91.5', '312_164.2', '309_127.7', '369_228.8', '615_44.8', '380_43.3', '271_41.4', '474_46.0', 
#  '433_265.2', '327_193.6', '535_44.2', '370_212.5', '597_205.6', '536_43.8', '276_41.6', '723_301.1', '207_328.6', '626_130.0', '868_226.2', 
#  '456_168.3', '264_47.6', '233_49.0', '777_258.4', '355_198.2', '475_46.4', '326_238.5', '371_210.2', '287_72.7', '133_55.4', '607_43.4', '310_426.4',
# '311_151.2', '348_203.0', '353_197.1', '432_262.4']
                
                #input_data_model = input_data_prep['features']

                input_data_model = input_data_prep

                st.dataframe(input_data_model)
        
# prediction
                model_maytenus = load_model_maytenus()    

                result = pd.DataFrame(input_data_model.index)

                result['prediction'] = model_maytenus.predict_proba(input_data_model)[:, 1]

                result.rename(columns={0:'sample'},inplace=True)

                #result.loc[result.prediction > 0, 'prediction'] = '*Maytenus ilicifolia*'
                #result.loc[result.prediction == 0, 'prediction'] = 'Unknown'

# processing the result to show
                result_markdown = tabulate(result, headers='keys', tablefmt='pipe')
                st.success('Done! Here is the sample classification:')
                st.markdown(result_markdown, unsafe_allow_html=True)
else:   
    if st.button('Run Machine Learning Prediction for Mikania laevigata'):

        if 'input_data' in st.session_state:
            with st.spinner('Please wait...'):
                time.sleep(3)
                input_data = st.session_state.input_data
                st.dataframe(input_data)

# feature engineering pipeline
                ref_training_data = pd.read_csv('ref_data_mikania.csv')
                input_data_rounded = rounder(input_data.drop(['isotopes', 'adduct','pcgroup'], axis=1))
                input_data_feat = feature_correspondance(ref_training_data, input_data_rounded)
                ref_data,input_data_clean = data_cleaning(ref_training_data,input_data_feat)
                input_data_prep = data_prep(ref_training_data,input_data_clean)
                input_data_prep = input_data_prep.set_index('index').T

# feature selection
                #features = ['1000_338.1', '119_337.1', '121_320.5', '136_572.4', '163_247.9', '163_337.4', 
                #            '165_230.9', '165_321.0', '181_42.3', '191_176.4', '204_46.9', '210_44.6', 
                #            '216_43.0', '217_43.2', '241_572.8', '278_530.2', '302_547.3', '306_209.5', 
                #            '318_433.7', '326_181.2', '326_247.9', '328_231.5', '336_390.4', '338_337.0', 
                #            '342_40.8', '350_172.5', '354_323.2', '362_247.4', '372_250.1', '378_43.5', 
                #            '388_39.7', '388_515.4', '390_515.1', '400_522.1', '406_328.8', '410_348.7', 
                #            '422_526.5', '424_380.9', '434_279.6', '440_356.3', '442_356.4', '448_530.3', 
                #            '450_298.3', '472_40.0', '478_319.0', '482_269.5', '484_349.2', '488_348.4', 
                #            '490_371.4', '492_344.6', '494_299.4', '500_338.9', '514_350.0', '516_323.2', 
                #            '526_269.6', '530_342.2', '534_211.1', '534_43.5', '548_530.5', '550_343.8', 
                #            '560_332.3', '572_353.1', '574_377.6', '594_310.9', '610_280.1', '612_257.3', 
                #            '626_333.6', '640_284.1', '652_181.2', '652_247.9', '652_319.3', '654_247.9', 
                #            '655_331.2', '656_231.0', '674_247.9', '680_179.4', '708_176.2', '794_545.8', 
                #            '817_546.4', '874_353.3', '876_360.2', '904_381.6', '918_353.8', '920_360.1', 
                #            '923_330.0', '940_376.4', '946_391.7', '962_367.5']

                #input_data_model = input_data_prep[features]
                input_data_model = input_data_prep

                st.dataframe(input_data_model)
        
# prediction
                model_mikania = load_model_mikania()    

                result = pd.DataFrame(input_data_model.index)

                result['prediction'] = model_mikania.predict_proba(input_data_model)[:, 1]

                result.rename(columns={0:'sample'},inplace=True)

                #result.loc[result.prediction > 0, 'prediction'] = '*Mikania laevigata*'
                #result.loc[result.prediction == 0, 'prediction'] = 'Unknown'

# processing the result to show
                result_markdown = tabulate(result, headers='keys', tablefmt='pipe')
                st.success('Done! Here is the sample classification:')
                st.markdown(result_markdown, unsafe_allow_html=True)
