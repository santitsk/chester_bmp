#how to run :   streamlit run ui_chester.py --server.port 3333
#Info       :   use for BMP business game 
#by         :   Santi T.
#rev        :   0

import streamlit as st
import pandas as pd
import numpy as np
import glob, base64
import locale
locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
import plotly.express as px

st.set_page_config(page_title="Chester Analysis", layout="wide", initial_sidebar_state="expanded")
#df_result = pd.read_csv("csv/result.csv")  #,index_col=["round","year","company","pars"]
#df_product = pd.read_csv("csv/product.csv")   #,index_col=["round","year","segment","name"]
df_seg = pd.read_csv("csv/segment.csv")  #,index_col=["round","year","segment"]

#utils
segments = {"Trad":"Traditional","Low":"Low End","High":"High End","Pfmn":"Performance","Size":"Size"}
companys = {"A":"Andrews","B":"Baldwin","C":"Chester","D":"Digby","E":"Erie","F":"Ferris"}
def sep_name_segment(x) :    
    return (x[0],segments[x[1]] if len(x)>1 else "NA", companys[x[0][0]])

def clean_number(text):
    tbl = str.maketrans('(', '-', 'R$ )')
    return locale.atof(text.translate(tbl))

def get_df_download_link(df, file_name="device_conf_export.csv", file_label='csv file'):
    import csv
    csv_file = df.to_csv(index=False)   #, sep='\t', quoting=csv.QUOTE_NONNUMERIC
    b64 = base64.b64encode(csv_file.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download {file_label}</a>'
    return href

@st.cache(allow_output_mutation=True)
def load_data_from_reports() :    
    #global df_result,df_product
    df_result, df_product = pd.DataFrame(), pd.DataFrame()
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
            df_result = df_result.append(df_finance)
            df_result = df_result.append(df_finance_sum)
            df_result = df_result.append(df_st)
            
            #read page 4 - product           
            df = pd.read_excel(f, "Table 4", header=6, parse_dates=["Date"])
            df.columns = ["name","UnitSold","UnitInvent","RevDate","Age","MTBF","Pfmm","Size","Price","MatCost","LaborCost","ContrMarg","SecShiftOverTime","AutomationNextRd","CapNextRd","PlantUtilz"]
            
            df = df.dropna()
            df[["name","segment","company"]] = df["name"].str.split().apply(lambda x : sep_name_segment(x)).apply(pd.Series)            
            df[["round","year"]] = (rnd, yr)
            df_product = df_product.append(df)            
    df_result = df_result.set_index(["round","year","company","pars"])
    df_result = df_result[~df_result.index.duplicated(keep="first")].copy()   
    df_result = df_result.sort_index() 
    

    
    #df_id["company"], df_id["name"],df_id["Pfmm"],df_id["Size"],df_id["radius"] = "ALL", df_id["segment"]+"_ideal",df_id["performance_ideal"],df_id["size_ideal"],10   
    #df_result =  df_na.append(df_p).append(df_id).append(df_product)
    
    
    
    #print(df_product)
    round_max = df_product["round"].max() #5
    round_year = df_product["year"].max() #2026
    #create next year (forecast)
    df_n1y = df_product[(df_product['round']==round_max)&(df_product["RevDate"].dt.year<round_year+1)].copy()
    df_n1y["round"] = df_n1y["round"] + 1
    df_n1y["year"] = df_n1y["year"] + 1
    df_n1y["Age"] = df_n1y["Age"] + 1
   #df_n1y.to_csv("test.csv")
    
    #create next year middle (forecast)
    df_n1y_m = df_product[(df_product['RevDate'].dt.year==round_year+1)].copy()
    df_n1y_m1 = df_n1y_m.copy()
    #df_n1y_m1.to_csv("test.csv")
    
    df_n1y_m1["round"] = df_n1y_m1["round"] + 0.5  
    df_n1y_m1["year"] = df_n1y_m1["year"] + 0.5
    df_n1y_m1["Age"] = (df_n1y_m1["Age"] + 0.5)/2
    df_n1y_m1 = df_n1y_m1.set_index(["segment","round","name"])    
    #@@@@@@@@@@@@@@@@@@@@@@   adjust in next year   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    df_n1y_m1.loc[("Performance", round_max+0.5, "C_PI"), ["Pfmm","Size","Age"]] = (15.4, 11.8, 1.2)
    df_n1y_m1.loc[("Low End", round_max+0.5, "Cedar"), ["Pfmm","Size","Age"]] = (4.7, 15.3, 4.9)
    
    
    df_n1y_m1 = df_n1y_m1.reset_index()    
    df_n1y_m2 = df_n1y_m1.copy()
    df_n1y_m2["round"] = df_n1y_m1["round"] + 0.5  
    df_n1y_m2["year"] = df_n1y_m1["year"] + 0.5
    df_n1y_m2["Age"] = df_n1y_m1["Age"] + 0.5
    
    #df_product
    df_product = df_product.append(df_n1y).append(df_n1y_m1).append(df_n1y_m2)  
    df_product.to_csv("test.csv")
    #df_product[["Pfmm","Size"]] = df_product[["Pfmm","Size"]].fillna(method="ffill")
    
    df_product = df_product.set_index(["round","year","segment"]).join(df_seg.set_index(["round","year","segment"]), how="left").reset_index()    
    df_product = df_product[~df_product.index.duplicated(keep="first")].copy()
    df_product = df_product.set_index(["round","year","company","segment","name"])
    df_product = df_product.sort_index()
    df_product["radius"] = 1
    
    
    
    
    #df_result.to_csv("csv/result.csv", index=True)
    #df_product.to_csv("csv/product.csv", index=True)
    
    df_result = df_result.reset_index()
    df_product = df_product.reset_index()
    #st.success("Done")
    return df_result, df_product

def prepare_segment() :    
    #workaround to make plotly show new product 
    df_na = df_product[df_product.segment=="NA"].copy()
    df_na["round"],df_na["Pfmm"],df_na["Size"],df_na["radius"] = 0,0,0,0  

    df = df_product[df_product.segment!="NA"].copy()
    df = df.set_index(["round","year","segment"])
    
    #create segment data    
    df = df[~df.index.duplicated(keep="first")].reset_index().copy()
    df_p = df.copy()
    df_p["company"], df_p["name"],df_p["Pfmm"],df_p["Size"],df_p["radius"] = "ALL", df_p["segment"],df_p["performance_center"],df_p["size_center"],100
     
    
    #create ideal data    
    df_id = df.copy()
    df_id["company"], df_id["name"],df_id["Pfmm"],df_id["Size"],df_id["radius"] = "ALL", df_id["segment"]+"_ideal",df_id["performance_ideal"],df_id["size_ideal"],10   
    df_result =  df_na.append(df_p).append(df_id).append(df_product)
    
    df_result['distance'] = np.linalg.norm(df_result[['Pfmm','Size']].values - df_result[['performance_ideal','size_ideal']].values, axis=1)
    df_result['distance_ideal'] = 0 
    return df_result

#************ UI
sidebar_logo = st.sidebar.container()
sidebar_src = st.sidebar.container()
sidebar_menu = st.sidebar.expander("MENU", expanded=True)
sidebar_cmd = st.sidebar.expander("COMMAND", expanded=True)
main_header = st.container()

#initial load data
df_result, df_product = load_data_from_reports()
df_xy = prepare_segment()
product_list = df_product.name.unique().tolist()
company_list = df_product.company.unique().tolist()
segment_list = df_product.segment.unique().tolist()

with sidebar_logo :
    pic_path = f"./pic/logo.png"
    st.image(pic_path, "", use_column_width=True) 

with sidebar_menu :
    main_menu = ["DATA","TREND","PERFORMANCE CHART"]
    menu = st.radio("Analysis", main_menu, index=1)  
    
with sidebar_cmd :    
    load_btn = st.button("Re-load from the Annual Reports")
    if load_btn :
        #clear caches (will reload MO, PO from the databases)
        st.legacy_caching.clear_cache() 

with main_header :
    if (main_menu.index(menu)==0) :
        with st.expander(f"RESULT", expanded=True) :
            df = df_result
            st.write(df)  
            st.write(f"Rows = {df.shape[0]:,} | Cols = {df.shape[1]:,}") 
            st.markdown(get_df_download_link(df, file_name=f"result.csv", file_label="csv file"), unsafe_allow_html=True)
        with st.expander(f"PRODUCT", expanded=True) :
            df = df_product
            st.write(df)  
            st.write(f"Rows = {df.shape[0]:,} | Cols = {df.shape[1]:,}") 
            st.markdown(get_df_download_link(df, file_name=f"result.csv", file_label="csv file"), unsafe_allow_html=True)
        with st.expander(f"SEGMENTATION", expanded=True) :
            df = df_seg
            st.write(df)  
            st.write(f"Rows = {df.shape[0]:,} | Cols = {df.shape[1]:,}") 
            st.markdown(get_df_download_link(df, file_name=f"segment.csv", file_label="csv file"), unsafe_allow_html=True)
        with st.expander(f"PERFORMANCE CHART", expanded=True) :
            df = df_xy
            st.write(df)  
            st.write(f"Rows = {df.shape[0]:,} | Cols = {df.shape[1]:,}") 
            st.markdown(get_df_download_link(df, file_name=f"performance_chart.csv", file_label="csv file"), unsafe_allow_html=True)
    if (main_menu.index(menu)==1) :       
        pars_product = [c for c in df_product.columns if c not in ["company","segment","name","round","year"]]
        cols_width = [0.2,0.9]        
        with st.expander(f"PLOT - COMPANY RESULT", expanded=True) :
            company_sel = st.multiselect("COMPANY ", company_list, default=company_list)
            pars_list = df_result.pars.unique().tolist()
            df_result["round"] = df_result["round"].astype(int)           
            
            col1, col2 = st.columns(cols_width) 
            pars_sel = col1.selectbox("Index 1", pars_list, index=pars_list.index("Sales"))            
            df_result_flt = df_result[df_result.company.isin(company_sel)&(df_result.pars==pars_sel)]
            fig = px.line(df_result_flt, x="round", y="value", color='company', markers=True)
            col2.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(cols_width)
            pars_sel = col1.selectbox("Index 2", pars_list, index=pars_list.index("Cumulative Profit"))  
            df_result_flt = df_result[df_result.company.isin(company_sel)&(df_result.pars==pars_sel)]
            fig = px.line(df_result_flt, x="round", y="value", color='company', markers=True)
            col2.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(cols_width)
            pars_sel = col1.selectbox("Index 3", pars_list, index=pars_list.index("ROE"))  
            df_result_flt = df_result[df_result.company.isin(company_sel)&(df_result.pars==pars_sel)]
            fig = px.line(df_result_flt, x="round", y="value", color='company', markers=True)
            col2.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(cols_width)
            pars_sel = col1.selectbox("Index 4", pars_list, index=pars_list.index("Inventory"))  
            df_result_flt = df_result[df_result.company.isin(company_sel)&(df_result.pars==pars_sel)]
            fig = px.line(df_result_flt, x="round", y="value", color='company', markers=True)
            col2.plotly_chart(fig, use_container_width=True)
        with st.expander(f"PLOT - PRODUCT", expanded=True) :            
            company_sel = st.multiselect("COMPANY", company_list, default=company_list)  
                 
            col1, col2 = st.columns(cols_width)
            segment_sel = col1.selectbox("Segment 1", segment_list, index=segment_list.index("Traditional"))        
            par_product_sel = col1.selectbox("Index 1", pars_product, index=pars_product.index("ContrMarg"))    
            df_product_flt = df_product[df_product.company.isin(company_sel)&(df_product.segment==segment_sel)]            
            fig = px.line(df_product_flt, x="round", y=par_product_sel, color='name', markers=True)
            col2.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(cols_width)
            segment_sel = col1.selectbox("Segment 2", segment_list, index=segment_list.index("High End"))        
            par_product_sel = col1.selectbox("Index 2", pars_product, index=pars_product.index("ContrMarg"))    
            df_product_flt = df_product[df_product.company.isin(company_sel)&(df_product.segment==segment_sel)]            
            fig = px.line(df_product_flt, x="round", y=par_product_sel, color='name', markers=True)
            col2.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(cols_width)
            segment_sel = col1.selectbox("Segment 3", segment_list, index=segment_list.index("Low End"))        
            par_product_sel = col1.selectbox("Index 3", pars_product, index=pars_product.index("ContrMarg"))    
            df_product_flt = df_product[df_product.company.isin(company_sel)&(df_product.segment==segment_sel)]            
            fig = px.line(df_product_flt, x="round", y=par_product_sel, color='name', markers=True)
            col2.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(cols_width)
            segment_sel = col1.selectbox("Segment 4", segment_list, index=segment_list.index("Performance"))        
            par_product_sel = col1.selectbox("Index 4", pars_product, index=pars_product.index("ContrMarg"))    
            df_product_flt = df_product[df_product.company.isin(company_sel)&(df_product.segment==segment_sel)]            
            fig = px.line(df_product_flt, x="round", y=par_product_sel, color='name', markers=True)
            col2.plotly_chart(fig, use_container_width=True)            

    if (main_menu.index(menu)==2) :
        with st.expander(f"PERFORMANCE CHART", expanded=True) : 
            cols_width = [0.3,0.7]
            col1, col2 = st.columns(cols_width)        
            company_sel = col1.multiselect("COMPANY", company_list, default=company_list)          
            df_xy_flt = df_xy[df_xy.company.isin([*company_sel,"ALL"])].copy()
            #st.write(df_xy_flt)
            fig = px.scatter(df_xy_flt, x="Pfmm", y="Size", animation_frame="round", animation_group="segment",
                             size="radius", color="name", hover_name="name",size_max=120, height=750, width=750, range_x=[0,15], range_y=[5,20])
            col2.plotly_chart(fig, use_container_width=True)
        with st.expander(f"POSITION CHART", expanded=True) : 
            cols_width = [0.3,0.7]
            col1, col2 = st.columns(cols_width)        
            company_sel = col1.multiselect("COMPANY 1", company_list, default=company_list)          
            segment_sel = col1.selectbox("SEGMENT 1", segment_list, index=segment_list.index("Traditional"))    
            
            df_dist_flt = df_xy[df_xy.company.isin(company_sel)&(df_xy.segment==segment_sel)].copy()
            
            #st.write(df_xy_flt)
            
            df_dist_flt["year"] = df_dist_flt["year"]+1
            #fig = px.scatter(df_dist_flt, x="year", y="distance", color='name').update_traces(mode="lines+markers")
            fig = px.line(df_dist_flt[df_dist_flt.distance.notna()], x="year", y="distance", color='name', hover_data=["Pfmm", "Age"], markers=True)
            #col2.write(df_dist_flt)
            #fig.update_traces(textposition="bottom right")
            fig.add_scatter(x=df_dist_flt['year'], y=df_dist_flt['distance_ideal'], line=dict(color='royalblue', width=4, dash='dot'))
            col2.plotly_chart(fig, use_container_width=True)
            #df_dist_flt=pd.melt(df_dist_flt, id_vars=['round', 'name'], value_vars=['Age', 'AGE_ideal'])
            
            fig = px.line(df_dist_flt[df_dist_flt.Age.notna()], x="year", y="Age", color='name',  markers=True)  #hover_data=["distance", "Age"],
            fig.add_scatter(x=df_dist_flt['year'], y=df_dist_flt['AGE_ideal'], line=dict(color='royalblue', width=4, dash='dot'))
            col2.plotly_chart(fig, use_container_width=True)
            
             