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
def load_model():
    '''
    
    model_path: path to model file. Eg: 'C:/Users/name/Documents/dev/model.pkl'
    
    '''
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    return pickled_model

@st.cache_resource
def load_refdata():
    '''
    returns the data used for training. It will be a reference data to create the feature names of input data.
    '''
    return pd.read_csv('ref_data.csv')

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

# files need to be in a zipped folder
uploaded_files = st.file_uploader('Choose a zipped folder with subfolders for each sample. Each file also needs to be in the mzXML format.', type='zip', accept_multiple_files=False, help='Only rar files are accepted')

# to get the xcms R scripts
xcms = os.getcwd() + "\\xcms.R"

# R.exe on local machine
command = "D:\Program Files\R\R-4.0.5\\bin\Rscript"

# object placeholder to tbe filled later
output_folder = "output"

# if the button is pressed and something was uploaded, continue
if st.button('Run XCMS') and uploaded_files is not None:
    
    # Read the uploaded zip folder
    zip_file_bytes = uploaded_files.getvalue()
    
    # loading symbol - keeps running as the script is running
    with st.spinner('Please wait ...'):
        
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
            
        # Extract the uploaded zip file contents into the folder
        with zipfile.ZipFile(io.BytesIO(zip_file_bytes), 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        
        # Run XCMS on the extracted files - outputfolder has the subfolders for each sample and their replicates
        process1 = subprocess.run([command, xcms], stdout=subprocess.PIPE, cwd=output_folder)
    
    st.success('Done! This is the data that will be used for the Machine Learning model:')
    #st.write(process1.stdout) 

    # Find the generated CSV file
    csv_files = glob.glob(os.path.join(output_folder, '*.csv'))
    

    if len(csv_files) > 0:
        csv_file = csv_files[0]
        input_data = pd.read_csv(csv_file, index_col=[0])
        #st.session_state.input_data = input_data
        #st.dataframe(input_data)  # Display the DataFrame

        subfolder_names = [name for name in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, name))]

        # Print the subfolder names and the folder names inside each subfolder
        #st.write("Subfolder names:")
        for subfolder_name in subfolder_names:
            #st.write(subfolder_name)
            subfolder_path = os.path.join(output_folder, subfolder_name)
            folder_names = [name for name in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, name))]
            #st.write(folder_names)
            #st.write("Folder names inside", subfolder_name, ":")
            #for folder_name in folder_names:
            #    st.write(folder_name)
        st.session_state.input_data = input_data.drop(folder_names,axis=1)
        st.dataframe(input_data.drop(folder_names,axis=1))  # Display the DataFrame


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

st.subheader('2. Sample classification')

if st.button('Run Machine Learning Prediction'):
    if 'input_data' in st.session_state:
        with st.spinner('Please wait...'):
            time.sleep(3)
            input_data = st.session_state.input_data
            st.dataframe(input_data)

# load the training dataset - its used as standard for the feature name generation
            ref_training_data = pd.read_csv('ref_data.csv')

# Using the data from previous steps
            input_data_rounded = rounder(input_data.drop(['isotopes', 'adduct','pcgroup'], axis=1))

# create the feature names on val set
            input_data_feat = feature_correspondance(ref_training_data, input_data_rounded)

# clean and prep val set
            ref_data,input_data_clean = data_cleaning(ref_training_data,input_data_feat)
            input_data_prep = data_prep(ref_training_data,input_data_clean)

            input_data_prep = input_data_prep.set_index('index').T

            features = ['103_57.3', '117_112.9', '129_35.7', '129_36.0', '129_36.6', '130_39.3', '137_201.8', 
            '138_202.4', '153_204.6', '161_317.4', '163_269.1', '163_282.9', '163_300.4', '164_207.3', 
            '164_239.7', '164_272.8', '165_295.0', '166_255.8', '173_71.3', '179_235.4', '181_39.5', 
            '187_339.3', '187_339.9', '191_211.3', '191_42.7', '195_43.1', '195_43.4', '205_270.0', 
            '206_277.8', '206_281.9', '207_328.8', '210_46.4', '210_47.6', '210_50.8', '215_41.2', 
            '222_212.3', '227_38.6', '232_70.8', '235_45.1', '245_36.0', '245_39.9', '246_47.9', 
            '259_172.1', '261_168.0', '261_193.0', '272_223.2', '272_274.4', '273_251.5', '273_255.5', 
            '279_291.7', '284_240.6', '285_198.0', '286_197.6', '287_85.7', '289_221.7', '293_315.1', 
            '294_444.9', '294_445.0', '296_65.2', '303_57.0', '304_49.7', '305_51.1', '305_51.9', '305_53.5', 
            '309_127.8', '309_424.2', '309_425.2', '311_429.4', '315_149.8', '325_210.6', '326_422.0', 
            '326_424.1', '327_236.3', '327_390.9', '327_391.7', '327_394.3', '327_394.8', '329_37.4', 
            '335_197.7', '335_222.5', '335_224.7', '337_45.9', '343_343.2', '345_210.5', '349_208.8', 
            '350_206.5', '353_212.4', '367_257.4', '368_277.5', '369_214.7', '370_229.0', '371_274.6', 
            '372_181.3', '372_261.1', '373_230.4', '391_164.5', '391_168.0', '392_210.8', '393_222.4', 
            '397_395.0', '410_233.3', '411_300.5', '411_304.5', '412_290.8', '412_301.8', '414_355.3', 
            '425_403.5', '431_273.7', '439_372.1', '440_44.7', '442_374.0', '446_232.5', '446_234.9', 
            '447_298.1', '450_246.2', '450_264.1', '458_236.5', '458_245.2', '463_279.9', '463_294.3', 
            '465_256.2', '470_226.5', '474_334.3', '475_273.3', '477_330.9', '481_278.3', '491_362.0', 
            '493_373.1', '494_375.2', '494_383.8', '495_354.2', '496_326.4', '498_252.2', '498_284.9', 
            '498_288.7', '499_314.1', '500_236.3', '500_283.0', '500_325.1', '501_284.6', '501_296.7', 
            '502_263.1', '502_287.7', '504_149.1', '504_365.5', '508_289.0', '508_335.7', '509_253.8', 
            '509_269.1', '510_367.3', '512_296.4', '514_257.0', '514_258.6', '515_255.2', '517_321.8', 
            '521_217.7', '521_271.9', '522_295.2', '523_324.1', '525_280.2', '528_214.9', '529_343.5', 
            '532_258.7', '533_240.9', '533_250.0', '533_256.4', '535_349.5', '540_200.4', '542_376.7', 
            '545_309.7', '552_285.4', '553_278.7', '553_285.3', '560_310.8', '561_231.5', '561_281.6', 
            '561_285.3', '562_267.8', '562_276.6', '563_216.0', '563_234.6', '563_328.2', '565_163.7', 
            '568_276.7', '575_143.4', '575_146.5', '576_139.2', '576_248.8', '577_182.0', '577_206.7', 
            '577_207.9', '578_187.4', '579_224.3', '579_257.3', '581_226.3', '581_247.1', '581_260.7', 
            '582_252.8', '582_266.4', '583_264.9', '593_272.7', '594_245.7', '596_203.2', '596_204.4', 
            '596_207.0', '596_274.5', '598_246.3', '598_246.9', '603_251.9', '609_259.8', '609_371.5', 
            '611_367.3', '611_382.6', '612_333.9', '612_334.5', '625_134.7', '636_337.9', '660_205.6', 
            '666_325.1', '670_439.9', '673_316.7', '687_268.2', '688_296.4', '688_303.5', '689_268.0', 
            '689_275.0', '689_275.5', '689_283.9', '689_432.3', '690_241.9', '690_245.2', '690_432.3', 
            '690_432.5', '697_259.3', '697_272.9', '697_282.1', '708_436.0', '716_425.4', '716_440.1', 
            '725_307.1', '726_312.1', '726_314.9', '736_125.5', '739_255.0', '740_262.1', '741_261.6', 
            '741_277.4', '742_251.5', '742_256.3', '742_282.0', '746_421.1', '747_418.6', '747_431.6', 
            '755_239.5', '755_268.8', '758_210.5', '758_229.4', '759_210.2', '761_211.7', '761_399.3', 
            '764_338.9', '764_345.9', '776_256.2', '799_56.5', '800_345.4', '800_55.8', '800_57.0', 
            '800_59.6', '806_228.2', '807_227.5', '817_303.1', '817_320.2', '824_247.5', '824_270.2', 
            '825_293.0', '825_300.0', '825_314.2', '826_288.5', '826_305.2', '827_288.0', '832_232.3', 
            '832_250.1', '833_249.1', '833_258.8', '833_259.1', '834_224.6', '834_228.6', '834_264.1', 
            '835_234.5', '835_237.7', '835_296.1', '836_304.1', '836_353.4', '837_261.8', '837_294.3', 
            '839_244.6', '840_248.9', '841_274.9', '847_177.9', '847_216.8', '847_220.5', '848_186.8', 
            '848_188.3', '849_186.5', '849_216.4', '849_217.4', '849_220.7', '850_235.8', '851_202.5', 
            '851_221.5', '851_232.7', '852_199.0', '852_226.9', '852_259.4', '855_257.4', '856_213.8', 
            '863_180.3', '865_182.1', '866_186.0', '867_171.1', '868_195.0', '868_228.4', '869_195.5', 
            '869_204.4', '870_207.3', '871_175.3', '889_187.7']

            input_data_model = input_data_prep[features]

            st.dataframe(input_data_model)
        

            model = load_model()    

            result = pd.DataFrame(input_data_model.index)

            result['prediction'] = model.predict(input_data_model)
            result.rename(columns={0:'sample'},inplace=True)
            result.loc[result.prediction > 0, 'prediction'] = '*Maytenus ilicifolia*'
            result.loc[result.prediction == 0, 'prediction'] = 'Unknown'

            result_markdown = tabulate(result, headers='keys', tablefmt='pipe')

            st.success('Done! Here is the sample classification:')
            st.markdown(result_markdown, unsafe_allow_html=True)