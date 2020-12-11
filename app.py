
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import time

from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import GridSearchCV, train_test_split

"""
# Gold Nanorods Size Prediction 

**_Instruction_**:

1. Upload .csv file

2. Hit the prediction button 
"""

path = "./data/"

@st.cache
def load_data(nrows):
    data = pd.read_csv(path + 'SPP+.csv', nrows=nrows)
    return data

data_load_state = st.text('Loading data...')
data = load_data(1000)
data_load_state.text("Training dara loading done! (using st.cache)")

if st.checkbox('Show all training data'):
    st.subheader('Training data')
    st.write(data)

# Real data input
st.header('Experimental Data Input')

# # Manual data input
# if st.button("Manual Input"):
#     @st.cache(allow_output_mutation=True)
#     def get_data():
#         return []
#
#     Resonance_energy = st.text_input("resonance_energy")
#     Linewidth = st.text_input("linewidth")
#
#     if st.button("Add row"):
#         get_data().append({"resonance_energy": Resonance_energy, "linewidth": Linewidth})
#
#     if st.button("Reset"):
#         get_data().clear()
#
#     Exp_data = pd.DataFrame(get_data())

uploaded_file = st.file_uploader("Choose your data", type="csv")

if uploaded_file:
    dataframe = pd.read_csv(uploaded_file)
    exp_data = pd.DataFrame(dataframe)
    Exp_width = exp_data['Width']
    Exp_length = exp_data['Length']
    Exp_data = exp_data.drop(['Wavelength', 'FWHM', 'Width', 'Length', 'MaxCscat'], axis=1)
    # Exp_data = exp_data.drop(['Wavelength', 'FWHM', 'SNR', 'MaxCscat'], axis=1)
    # Exp_data = exp_data.drop(['E_res', 'Linewidth', 'SNR', 'MaxCscat'], axis=1)

    st.write(Exp_data)

# arranging features from original dataset for model learning
x = data.drop(['Wavelength (nm)', 'Width (nm)', 'AspectRatio', 'Length (nm)', 'Linewidth (nm)', 'MaxCscat'], axis=1)
# x = data.drop(['E_res', 'Width', 'AspectRatio', 'Length', 'FWHM', 'MaxCscat'], axis=1)
w_y = data['Width (nm)']
l_y = data['Length (nm)']

# parameters for GridSearchCV class
param_grid = {'max_depth': range(1, 31)}

# Initialize GridSearchCV class
wgs = GridSearchCV(estimator=DTR(),
                   param_grid=param_grid,
                   cv=10, scoring='neg_mean_squared_error')
lgs = GridSearchCV(estimator=DTR(),
                   param_grid=param_grid,
                   cv=10, scoring='neg_mean_squared_error')

wgs.fit(x, w_y)
lgs.fit(x, l_y)

st.header('Prediction')

if st.button("Prediction!!!"):

    wexp_y_pred = wgs.predict(Exp_data)
    lexp_y_pred = lgs.predict(Exp_data)

    st.dataframe(({"True_Width": Exp_width, "Predicted_Width": wexp_y_pred, "True_Length": Exp_length, "Predicted_Length": lexp_y_pred}))
    # st.dataframe(({"Predicted_Width": wexp_y_pred, "Predicted_Length": lexp_y_pred}))

st.header('Evaluation')

if st.button("Evaluation!!!"):

    wexp_y_pred = wgs.predict(Exp_data)
    lexp_y_pred = lgs.predict(Exp_data)

    st.subheader('Prediction error plot')
    # Improved spontaneous Plot
    fig, ax1 = plt.subplots(figsize=(3.33, 3.33), dpi=300)

    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.grid']=False
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['axes.edgecolor']='black'
    plt.rcParams['axes.labelpad'] = 0
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams['legend.edgecolor'] = 'black'

    x = np.arange(-1, 181)

    ax1.plot([-1, 181], [-1, 181], color='black', Linewidth=0.5, label='Error 0 % line')
    ax1.plot(x, 1.08 * x, color='black', linestyle='dashed', Linewidth=0.5, label='Error 8 % line', alpha=0.75)
    ax1.plot(x, 0.92 * x, color='black', linestyle='dashed', Linewidth=0.5, alpha=0.75)

    ax1.scatter(Exp_width, wexp_y_pred, label='Width', s=3, c='b')
    ax1.scatter(Exp_length, lexp_y_pred, label='Length', s=3, c='r')

    ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0, fontsize=7.5)
    ax1.set_xticks(np.arange(0, 180 + 1, 20))
    ax1.set_yticks(np.arange(0, 180 + 1, 30))

    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, 180)
    ax1.set_xlabel('Measured Size (nm)', fontsize=12)
    ax1.set_ylabel('Predicted Size (nm)', fontsize=12)
    ax1.tick_params(labelsize=12)
    st.pyplot(fig)

    st.subheader('Relative Error')

    wdif = (np.abs(Exp_width.T - wexp_y_pred) / Exp_width.T) * 100
    ldif = (np.abs(Exp_length.T - lexp_y_pred) / Exp_length.T) * 100

    fig, ax1 = plt.subplots(figsize=(3.33, 3.33),dpi=300)
    ax1.hist(wdif, alpha=0.5, label='Width', color='b', bins='auto', ec='b')
    n1, bins1, patches1 = ax1.hist(ldif, alpha=0.5, label = 'Length', color='r', bins='auto', ec='r')
    n2, bins2, patches2 = ax1.hist(wdif, alpha=0, bins='auto')

    y2 = np.add.accumulate(n1) / n1.sum()
    x2 = np.convolve(bins1, np.ones(2) / 2, mode="same")[1:]
    ax2 = ax1.twinx()
    lines = ax2.plot(x2, y2, ls='-', color='r')

    ax1.set_xlabel('Relative Error (%)', fontsize=12)
    ax1.set_ylabel('Occurrences', fontsize=12)
    ax2.set_ylabel('Cumulative Ratio', fontsize=12)
    y4 = np.add.accumulate(n2) / n2.sum()
    x4 = np.convolve(bins2, np.ones(2) / 2, mode="same")[1:]
    lines = ax2.plot(x4, y4, ls='-', color='b')

    st.pyplot(fig)

if st.button("Plot"):

    wexp_y_pred = wgs.predict(Exp_data)
    lexp_y_pred = lgs.predict(Exp_data)

    fig, ax = plt.subplots(figsize=(3.33, 3.33), dpi=300)
    x = np.arange(0, len(wexp_y_pred))
    plt.plot(x, wexp_y_pred, c='r', label='Predicted width change')
    plt.plot(x, lexp_y_pred, c='b', label='Predicted length change')

    plt.plot(len(wexp_y_pred), 35, marker='o', markersize=3, c='r', label='Width from SEM')
    plt.plot(len(lexp_y_pred), 56, marker='o', markersize=3, c='b', label='Length from SEM')

    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Size (nm)', fontsize=12)
    plt.legend()

    # SEMseg
    # plt.plot(len(wexp_y_pred), 107.6359, marker='o', markersize=3)
    # plt.plot(len(wexp_y_pred), 62.4794, marker='o', markersize=3)

    st.pyplot(fig)
