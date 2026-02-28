In the previous folder: 0, we have created a DF with the Spectrum and the Thickness, the purpose of this folder is to fit the experimental curves with the theoretical model for T. To do this, we consider several algorithms, such as: Basin Hopping, Differential Evolution and Direct. The results of each optimization are saved as a numpy array with the following structure: 

Index, Thickness, R1, R2, Sellemier Coefficients (5), absorption coefficients (3), ne and MSE.

And the arrays are saved in the folder: ../../results/SciPy_IIM/

Notebook 1: This notebook uses the Basin-Hopping algorith to fit the curves. 
Notebook 2: This notebook uses the Differential Evolution algorithm to fit the curves. 
Notebook 3: This ntoebook uses the Direct algorithm to fit the curves. 

Notebook 4: This notebooks presents a comparison of all the previous results. The final graphics are done here. There is a added section for the Ng plot. 

