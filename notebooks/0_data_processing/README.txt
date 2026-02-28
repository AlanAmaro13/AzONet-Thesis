Hello there!

* All the dataframes have the extension .pkl. 

Notebook 0: Convert the 101 samples into an usable DataFrame, here the thickness correction has already be done. 
Notebook 1: Convert the 15 samples into an usable DataFrame, here the thickess is corrected using: (df_junto['Espesor'] - 17.897827)/1.1521356
Notebook 2: Convert the 45 samples into an usable DataFrame, here the thickess is corrected using: (df_junto['Espesor'] - 17.897827)/1.1521356

After all these three notebook, you will end up with three dataframes, the following notebook is used to join them. 

Notebook 3: This notebook unifies the three previous dataframes. 

Let's select only the 145 with the better results, this is done in: 

Notebook 4: This notebook select only 145 samples from a pair of indexes, this notebook generates the used graphics. 

The final results are summarized in Notebook 4, we generate the histogram for 200, 100 and 50 nm, along with the Porcentual Error plot. The selected samples are saved in dataframe_spectrum_thickness_145_final.pkl, that is our glorified dataframe. 
