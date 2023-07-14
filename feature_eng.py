# creates the feature name with the mz and rt
def feature_name_creation(xcms_file_path):
    table = pd.read_csv(xcms_file_path, index_col=[0]) 
    
    # no need for decimal on m/z (low resolution) and only one decimal for rt
    table.mz = table.mz.round(0).astype(int)
    table.rt = table.rt.round(1)

    # creating the feature name: mz_rt
    features = table["mz"].astype(str) + "_" + table["rt"].astype(str)
    table.insert(0, 'features', features) # first column
    
    # drop as we don't know how many columns the table will have. Drop the known ones. 
    # There should only be the 'features' column and the samples
    
    table_clean = table.drop(['isotopes', 'adduct','pcgroup'], axis=1) #'npeaks','NEG_GROUP', 'POS_GROUP',
    
    return table_clean


# rounds the mz and rt columns along with its min and max

def rounder(dataframe):
    table = dataframe 
    
    table.mz = table.mz.round(0).astype(int)
    table.mzmin = table.mzmin.round(0).astype(int)
    table.mzmax = table.mzmax.round(0).astype(int)
    
    table.rt = table.rt.div(60).round(1)
    table.rtmin = table.rtmin.div(60).round(1)
    table.rtmax = table.rtmax.div(60).round(1)

    
    return table


def feature_correspondance (ref_data, target_data):
    
    # column to be populated
    target_data['features'] = np.nan

    # sorts the values in order to use the feature name of  features with higher npeaks.
    # sometimes, the preprocessing generated 'duplicates' of almost the same feature. We ignore those later on
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

                    # if the condition is met, then the FIRST feature name is extracted and given to the target data.
                    # The one with higher npeak will be used, due to sorting at the beggining
                    target_data.loc[i,'features'] = ref_data.loc[j,'features']
                    
                # this brakes the second if statement. There could be multiple matches on rt 
                break
    return target_data



