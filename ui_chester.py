#how to run :   streamlit run ui_chester.py --server.port 3333
#Info       :   use for BMP business game 
#by         :   Santi T.
#rev        :   0

import streamlit as st
import pandas as pd
import numpy as np
import glob, base64

st.set_page_config(page_title="Chester Analysis", layout="wide", initial_sidebar_state="expanded")
df_all = pd.read_csv("csv/result.csv",index_col=["round","year","company","pars"])
df_product = pd.read_csv("csv/product.csv",index_col=["round","year","segment","name"])
df_seg = pd.read_csv("csv/segment.csv",index_col=["round","year","segment"])

#utils
segments = {"Trad":"Traditional","Low":"Low End","High":"High End","Pfmn":"Performance","Size":"Size"}
companys = {"A":"Andrews","B":"Baldwin","C":"Chester","D":"Digby","E":"Erie","F":"Ferris"}
def sep_name_segment(x) :    
    return (x[0],segments[x[1]] if len(x)>1 else None, companys[x[0][0]])

def clean_number(text):
    import locale
    locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
    tbl = str.maketrans('(', '-', 'R$ )')
    return locale.atof(text.translate(tbl))

def get_df_download_link(df, file_name="device_conf_export.csv", file_label='csv file'):
    import csv
    csv_file = df.to_csv(index=False)   #, sep='\t', quoting=csv.QUOTE_NONNUMERIC
    b64 = base64.b64encode(csv_file.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download {file_label}</a>'
    return href

def load_data_from_reports() :    
    global df_all,df_product
    df_all, df_product = pd.DataFrame(), pd.DataFrame()
    files = glob.glob("excel/[!~]*.xlsx" )    
    #for f in files[0:1]:
    for f in files:
        with st.spinner(f'Loading {f}'):
            #read page 1 - financial stat.
            df = pd.read_excel(f, "Table 1", header=None)
            #get round and year
            rnd, yr = int(df.iloc[0,0].split("\n")[0].split(":")[1]), int(df.iloc[0,0].split("\n")[2])       
            row_start = df[df.iloc[:,0]=="Selected Financial Statistics"].index[0]+1
            df_finance = pd.read_excel(f, 0, header=row_start, nrows=12, index_col=0)
            cols = [c for c in df_finance.columns if not c.startswith("Unna")]
            df_finance = df_finance[cols]
            df_finance = df_finance.stack().reset_index()
            df_finance.columns = ["pars","company","value"]
            df_finance.value = df_finance.value.astype(float)
            df_finance[["round","year"]] = (rnd, yr)
            #convert money to 1000$ unit
            filter_cond = df_finance.pars.isin(["Sales","EBIT","Profits","Cumulative Profit"])
            df_finance.loc[filter_cond, "value"] = df_finance.loc[filter_cond, "value"]/1000
           
            #read page 3 - financial summary
            df = pd.read_excel(f, "Table 3", header=1, index_col=0)
            df = df.apply(lambda x : pd.to_numeric(x, errors='coerce'), axis=1).astype(float)
            df = df.dropna()
            df_finance_sum = df.stack().reset_index()
            df_finance_sum.columns = ["pars","company","value"]
            df_finance_sum[["round","year"]] = (rnd, yr)
            
            #read page 2 - stock                
            df = pd.read_excel(f, "Table 2", header=2, index_col=0, nrows=6)            
            cols = [c for c in df.columns if not c.startswith("Unna")]
            df_st = df[cols].copy()
            if "Close" in df_st.columns :
                df_st.columns = ["Stock Market Close","Stock Market Change","MarketCap shares","Yield","P/E"]
                df_st[["Stock Market Close","Stock Market Change"]] = df_st[["Stock Market Close","Stock Market Change"]].astype(float)
                df_st[["MarketCap shares","MarketCap per share","Book Value","EPS","Dividend"]] = df_st["MarketCap shares"].apply(lambda x : (clean_number(x.split()[0])/1000, 
                                                                                                        clean_number(x.split()[1]), 
                                                                                                        clean_number(x.split()[2]),
                                                                                                        clean_number(x.split()[3]),
                                                                                                        clean_number(x.split()[4]),
                                                                                                        )).apply(pd.Series)
                df_st[["Yield","P/E"]] = df_st[["Yield","P/E"]].astype(float)
            if len(df_st.columns)==5 :
                df_st.columns = ["Stock Market Close","MarketCap shares","EPS","Yield","P/E"]
                df_st[["Stock Market Close","Stock Market Change"]] = df_st["Stock Market Close"].apply(lambda x : (clean_number(x.split()[0]), 
                                                                                                                clean_number(x.split()[-1]))).apply(pd.Series)
                df_st[["MarketCap shares","MarketCap per share","Book Value"]] = df_st["MarketCap shares"].apply(lambda x : (clean_number(x.split()[0])/1000, 
                                                                                                        clean_number(x.split()[1]), 
                                                                                                        clean_number(x.split()[2]))).apply(pd.Series)
                df_st[["EPS","Dividend"]] = df_st["EPS"].apply(lambda x : (clean_number(x.split()[0]), clean_number(x.split()[1]))).apply(pd.Series)
                df_st[["Yield","P/E"]] = df_st[["Yield","P/E"]].astype(float)
            elif len(df_st.columns)==4 :
                df_st.columns = ["Stock Market Close","MarketCap shares","Yield","P/E"]
                df_st[["Stock Market Close","Stock Market Change"]] = df_st["Stock Market Close"].apply(lambda x : (clean_number(x.split()[0]), 
                                                                                                                clean_number(x.split()[-1]))).apply(pd.Series)
                df_st[["MarketCap shares","MarketCap per share","Book Value","EPS","Dividend"]] = df_st["MarketCap shares"].apply(lambda x : (clean_number(x.split()[0])/1000, 
                                                                                                        clean_number(x.split()[1]), 
                                                                                                        clean_number(x.split()[2]),
                                                                                                        clean_number(x.split()[3]),
                                                                                                        clean_number(x.split()[4]),
                                                                                                        )).apply(pd.Series)
                df_st[["Yield","P/E"]] = df_st[["Yield","P/E"]].astype(float)            
            df_st = df_st.stack().reset_index()
            df_st.columns = ["company","pars","value"]
            df_st[["round","year"]] = (rnd, yr)
            df_st.value = df_st.value.astype(float)                         
            #combine
            df_all = df_all.append(df_finance)
            df_all = df_all.append(df_finance_sum)
            df_all = df_all.append(df_st)
            
            #read page 4 - product           
            df = pd.read_excel(f, "Table 4", header=6, parse_dates=["Date"])
            df.columns = ["name","UnitSold","UnitInvent","RevDate","Age","MTBF","Pfmm","Size","Price","MatCost","LaborCost","ContrMarg","SecShiftOverTime","AutomationNextRd","CapNextRd","PlantUtilz"]
            df = df.dropna()
            df[["name","segment","company"]] = df["name"].str.split().apply(lambda x : sep_name_segment(x)).apply(pd.Series)            
            df[["round","year"]] = (rnd, yr)
            df_product = df_product.append(df)            
    df_all = df_all.set_index(["round","year","company","pars"])
    df_all = df_all[~df_all.index.duplicated(keep="first")].copy()    
    print(df_product)
    df_product = df_product.set_index(["round","year","segment"]).join(df_seg, how="left").reset_index()
    
    df_product = df_product[~df_product.index.duplicated(keep="first")].copy()
    df_product = df_product.set_index(["round","year","company","segment","name"])
    
    df_all.to_csv("csv/result.csv", index=True)
    df_product.to_csv("csv/product.csv", index=True)
    st.success("Done")

#************ UI
sidebar_logo = st.sidebar.container()
sidebar_src = st.sidebar.container()
sidebar_menu = st.sidebar.expander("MENU", expanded=True)
sidebar_cmd = st.sidebar.expander("COMMAND", expanded=True)
main_header = st.container()

#initial load data
with sidebar_logo :
    pic_path = f"./pic/logo.png"
    st.image(pic_path, "", use_column_width=True) 

with sidebar_menu :
    main_menu = ["DATA","TREND","Other"]
    menu = st.radio("Analysis", main_menu, index=1)  
    
with sidebar_cmd :    
    load_btn = st.button("Re-load from the Annual Reports")
    if load_btn :
        load_data_from_reports()
    
    
with main_header :
    if (main_menu.index(menu)==0) :
        with st.expander(f"RESULT", expanded=True) :
            df = df_all.reset_index()
            st.write(df)  
            st.write(f"Rows = {df.shape[0]:,} | Cols = {df.shape[1]:,}") 
            st.markdown(get_df_download_link(df, file_name=f"result.csv", file_label="csv file"), unsafe_allow_html=True)
        with st.expander(f"PRODUCT", expanded=True) :
            df = df_product.reset_index()
            st.write(df)  
            st.write(f"Rows = {df.shape[0]:,} | Cols = {df.shape[1]:,}") 
            st.markdown(get_df_download_link(df, file_name=f"result.csv", file_label="csv file"), unsafe_allow_html=True)
        with st.expander(f"SEGMENTATION", expanded=True) :
            df = df_seg.reset_index()
            st.write(df)  
            st.write(f"Rows = {df.shape[0]:,} | Cols = {df.shape[1]:,}") 
            st.markdown(get_df_download_link(df, file_name=f"segment.csv", file_label="csv file"), unsafe_allow_html=True)