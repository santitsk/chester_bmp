
#how to run :   streamlit run ui_vibml.py --server.port 7776
#by         :   Santi T.

from sqlalchemy.sql.expression import column
import streamlit as st
#from streamlit import caching
import pandas as pd
import numpy as np
import os.path
import requests, json, time
from sqlalchemy import create_engine

cached_path = "cached/"

#from logging import error
#from altair.vegalite.v4.schema.core import Header
#import sqlalchemy


#from datetime import date, datetime
#from datetime import timedelta
#from dateutil.relativedelta import relativedelta
#from sqlalchemy import create_engine
#from sqlalchemy.sql.elements import Null
#from styleframe import StyleFrame, Styler

import sys
sys.path.append("..")
import importlib
from mylib import lib_sap as mysap
importlib.reload(mysap)
from mylib import lib_config as mycfg
importlib.reload(mycfg)
from mylib import lib_streamlit as myst
importlib.reload(myst)
from mylib import lib_vibration as myvib
importlib.reload(myvib)
from styleframe import StyleFrame, Styler

#configuration
import configparser 
cfg = configparser.ConfigParser()
cfg.read('../vibml_predict/config.ini')                  #read common configuration
ip = cfg["prediction_api"]["ip"]
port = str(cfg["prediction_api"]["port"])
influxdb_server_id = int(cfg["influxdb_input"]["influxdb_server_id"])
meas_ovl, meas_ovl_spd = cfg["influxdb_input"]["meas_ovl"], cfg["influxdb_input"]["meas_ovl_spd"]
meas_sp, meas_ts = cfg["influxdb_input"]["meas_sp"], cfg["influxdb_input"]["meas_ts"]
meas_pred_no = cfg["influxdb_output"]["meas_pred_no"]
meas_pred = {'pred_problem' : f'vml_pred_problem{meas_pred_no}',
             'point_summary': f'vml_point_summary{meas_pred_no}'}
pred_problem_meas = meas_pred["pred_problem"]

#prediction API
headers = {'Content-type': 'application/json', 'charset': 'utf-8'}
api_baseline_path = f"http://{ip}:{port}/api/calc_baseline"  

#******************setting
main_menu = ["Point Configuration","ML Configuration","Last Speed Calc / ML Predict"]
src_list = ["SKF","WIZARD","SYSTEM1"]
skf_srv_list = ["SKF-BP1","SKF-BP2","SKF-BP3","SKF-WS1","SKF-WS2","SKF-WS3"]
# main_menu = ["MO/PO with Work Center","MO this month but no PO","PO Last 3, Next 3"]
cols_width_2_short = [0.12, 1]
cols_width_2_wide = [0.2, 1]
cols_width_4 = [0.5, 0.4, 0.4, 0.4]

#******************Streamlit
st.set_page_config(page_title="VIB ML Configuration", layout="wide", initial_sidebar_state="expanded")

#load data
@st.cache(allow_output_mutation=True)
def get_connection_conf_mysql():
    config_ip, config_user, config_pw = "10.28.58.29", "mysql", "mysql" 
    return create_engine(f"mysql+mysqldb://{config_user}:{config_pw}@{config_ip}/buffer_conf?charset=utf8", pool_pre_ping=True)

@st.cache(allow_output_mutation=True)
def load_vibml_point_cfg_skf():
    #return df_skf_points, df_skf_points_all
    return mycfg.vibml_point_cfg_skf()  

@st.cache(allow_output_mutation=True)
def load_vibml_bearing_freq():
    return mycfg.vibml_bearing_freq()

def load_vibml_ml_cfg(source="skf", server_id=[], only_en=False):
    return mycfg.vibml_ml_cfg(source=source, server_id=server_id, only_en=only_en)

engine = get_connection_conf_mysql()

#************ UI
sidebar_logo = st.sidebar.container()
sidebar_src = st.sidebar.container()
sidebar_menu = st.sidebar.expander("MENU", expanded=True)
sidebar_cmd = st.sidebar.container()
main_header = st.container()

#initial load data
with sidebar_logo :
    pic_path = f"./pic/logo.png"
    st.image(pic_path, "", use_column_width=True)    
    
with sidebar_src:
    src = st.selectbox("Source", src_list)
  
with sidebar_menu :
    menu = st.radio("Configuration", main_menu, index=1)       

with sidebar_cmd :
    _, _ = st.text("  "), st.text("  ")   
    clear_btn = st.button("Reload All")
    if clear_btn :
        #clear caches (will reload MO, PO from the databases)
        st.legacy_caching.clear_cache()       
        
with main_header :    
    src_lower = src.lower()
    # MENU 1 :  Point Conf from SKF
    if (main_menu.index(menu)==0) and (src_lower=="skf") :                
        col1, col2 = st.columns(cols_width_2_short)   
        skf_srv_sel = st.multiselect("SKF server", skf_srv_list, default=skf_srv_list)   
        with st.expander(f"EXISTING CONFIGURATION", expanded=True) :
            #st.header("Existing configuration")
            #load SKF point conf
            df_skf_points, df_skf_points_all = load_vibml_point_cfg_skf()
            df_skf_points_ft = df_skf_points.loc[df_skf_points.index.get_level_values(0).isin(skf_srv_sel)]
            st.write(df_skf_points_ft.head(30).reset_index())               
            
            #load SKF bearing freq
            df_bearings = load_vibml_bearing_freq()
            df_bearings = df_bearings[df_bearings.source==src_lower]
            df_bearings_ft = df_bearings.loc[df_bearings.index.get_level_values(0).isin(skf_srv_sel)]
            
            col1, col2, col3, col4 = st.columns(cols_width_4)
            col1.write(f"Point {src} : Rows = {df_skf_points_ft.shape[0]:,} | Cols = {df_skf_points_ft.shape[1]:,}") 
            request_dl = col2.button("Request to Download")
            if request_dl :
                with st.spinner(f'Converting to csv files'):
                    col3.markdown(myst.get_df_download_link(df_skf_points_ft.reset_index(), 
                                                        file_name=f"point_cfg_{src_lower}.csv", file_label="point_cfg csv file"), unsafe_allow_html=True)       
                    col4.markdown(myst.get_df_download_link(df_bearings_ft.reset_index(), 
                                                        file_name=f"bearing_freq_{src_lower}.csv", file_label="bearing_freq csv file"), unsafe_allow_html=True)  
        with st.expander(f"UPDATE CONFIGURATION", expanded=True) :
            if src_lower=="skf" :  
                #st.header("Update")           
                col1, col2 = st.columns(cols_width_2_short)            
                reload_sap = col1.checkbox("Reload SAP", value=False)
                initialize_cfg = col1.checkbox("Initial cfg.", value=False)
                update_conf = col2.button("Update Conf.")
                if update_conf :       
                    if initialize_cfg :
                        df_skf_points, df_skf_points_all = pd.DataFrame(), pd.DataFrame()                    
                    with st.spinner('Loading SAP FL/EQ'):
                        df_FL, df_EQ = mysap.read_FL_EQ(reload_sap)
                    with st.spinner('Loading SAP FL Section'):
                        df_FL_section = mysap.read_FL_section(reload_sap, df_FL)
                    with st.spinner('Loading SAP EQ Object'):
                        df_EQ_object = mysap.read_EQ_object_type(reload_sap, df_EQ)
                    df_skf_srvs = mycfg.server_list() 
                    #update_srvs = skf_srv_list if skf_srv=="ALL" else [skf_srv]                                         
                    for update_srv in skf_srv_sel :
                        srv_id = df_skf_srvs.loc[df_skf_srvs.server_shortname==update_srv].index[0]
                        with st.spinner(f'Loading SKF Engine Conf. of {update_srv}'):
                            engine_vib, pars = mycfg.skf_server_engine(srv_id)
                        with st.spinner(f'Loading SKF Bearing Freq. of {update_srv}'):
                            df_bearing = myvib.get_rpm_by_point_id(engine_vib, update_srv, pars, 0)
                            df_bearings.bearing_name = df_bearings.bearing_name.str.replace(" ", "")
                            df_bearings.bearing_name = df_bearings.bearing_name.str.replace(" ", "")
                            df_bearings.bearing_name = df_bearings.bearing_name.str.replace(" ", "")
                            print(df_bearings[df_bearings.bearing_name.str.contains(" ")])
                            if df_bearing.shape[0]>0 :
                                df_bearings = df_bearings.loc[df_bearings.index.get_level_values(0)!=update_srv]
                                df_bearings = df_bearings.append(df_bearing , sort=False).sort_index()
                        with st.spinner(f'Loading SKF Unit/Measmode/FL/EQ/Alarm of {update_srv}'):
                            df_unit = myvib.get_skf_unit(engine_vib, pars)
                            df_measmode = myvib.get_skf_measmode(engine_vib, pars)
                            df_fl_eq = myvib.get_skf_fl_eq(engine_vib, pars)
                            df_alarm = myvib.get_skf_alarm(engine_vib, pars)
                        with st.spinner(f'Loading SKF Path of {update_srv}'):
                            df_path = myvib.get_skf_path(engine_vib, pars)
                        with st.spinner(f'Join SKF data of {update_srv}'):
                            df = df_path.join(df_fl_eq, how='inner').join(df_unit, how='inner').join(df_measmode, how='left').join(df_alarm, how='left')
                            df = df.reset_index().merge(df_FL_section, on='SEGMENTNAME', how='left').merge(df_EQ_object, on='ASSETNAME', how='left')
                            filter_cond = ~df.measmode_cfg.isna()
                            df.loc[filter_cond, 'measmode_cfg'] = df.loc[filter_cond, 'measmode_cfg'].astype(int)
                            df.loc[filter_cond, 'ALERTHI_peak'] = df.loc[filter_cond,:].apply(lambda x : x['ALERTHI']*pars['measmode'][x['measmode_cfg']], axis=1)
                            df.loc[filter_cond, 'DANGERHI_peak'] = df.loc[filter_cond,:].apply(lambda x : x['DANGERHI']*pars['measmode'][x['measmode_cfg']], axis=1)
                            df.insert(0, 'server_name', pars['server_name'])
                            df.insert(0, 'server_id', pars['server_id'])
                            df = df.set_index(['server_name','point_id'])
                            if df.shape[0]>0 :
                                df_skf_points = df_skf_points.loc[df_skf_points.index.get_level_values(0)!=pars['server_name']]
                                df_skf_points = df_skf_points.append(df , sort=False).sort_index()             
                        with st.spinner(f'Join SKF data (all) of {update_srv}'):
                            df_all = df_path.join(df_unit, how='inner').join(df_measmode, how='left').join(df_alarm, how='left')
                            df_all = df_all.reset_index()         
                            filter_cond = ~df_all.measmode_cfg.isna()
                            df_all.loc[filter_cond, 'measmode_cfg'] = df_all.loc[filter_cond, 'measmode_cfg'].astype(int)
                            df_all.loc[filter_cond, 'ALERTHI_peak'] = df_all.loc[filter_cond,:].apply(lambda x : x['ALERTHI']*pars['measmode'][x['measmode_cfg']], axis=1)
                            df_all.loc[filter_cond, 'DANGERHI_peak'] = df_all.loc[filter_cond,:].apply(lambda x : x['DANGERHI']*pars['measmode'][x['measmode_cfg']], axis=1)
                            df_all.insert(0, 'server_name', pars['server_name'])
                            df_all.insert(0, 'server_id', pars['server_id'])
                            df_all = df_all.set_index(['server_name','point_id'])
                            if df_all.shape[0]>0 :
                                df_skf_points_all = df_skf_points_all.loc[df_skf_points_all.index.get_level_values(0)!=pars['server_name']]
                                df_skf_points_all = df_skf_points_all.append(df_all , sort=False).sort_index()          
                    with st.spinner(f'Write SKF mapping bearing freq to mysql'):           
                        cols = ["bearing_name"]
                        df_map_bearings = df_bearings[cols].reset_index().set_index(["server_name","point_id","bearing_name"]).sort_index()                    
                        df_map_bearings = df_map_bearings.loc[~df_map_bearings.index.duplicated(keep="last")] 
                        df_map_bearings = df_map_bearings.reset_index()
                        df_map_bearings.point_id = df_map_bearings.point_id.astype(str)
                        mycfg.vibml_update_table(src_lower, f"vml_mapping_bearing_auto_{src_lower}", df_map_bearings, if_exists='replace', 
                                                primary=["source","server_name","point_id","bearing_name"], varchar_index=["source","server_name","point_id","bearing_name"])
                    with st.spinner(f'Write SKF bearing freq to mysql'):     
                        df_bearings = df_bearings.reset_index().set_index("bearing_name").sort_index()
                        cols = ["BearingRpm_BPFO","BearingRpm_BPFI","BearingRpm_BSF","BearingRpm_FTF"]
                        df_bearings = df_bearings.loc[~df_bearings.index.duplicated(keep="last"), cols] 
                        df_bearings = df_bearings.reset_index()
                        mycfg.vibml_update_table(src_lower, f"vml_bearing_freq_auto_{src_lower}", df_bearings, if_exists='replace', 
                                                primary=["source","bearing_name"], varchar_index=["source","bearing_name"])
                    with st.spinner(f'Write SKF point cfg to mysql'):            
                        df_skf_points = df_skf_points.reset_index()
                        df_skf_points.point_id = df_skf_points.point_id.astype(str)
                        mycfg.vibml_update_table(src_lower, f"vml_point_cfg_auto_{src_lower}", df_skf_points, if_exists='replace', 
                                                primary=["source","server_name","point_id"], varchar_index=["source","server_name","point_id"])
                    with st.spinner(f'Write SKF point cfg (all) to mysql'):            
                        df_skf_points_all = df_skf_points_all.reset_index()           
                        df_skf_points_all.point_id = df_skf_points_all.point_id.astype(str)                                  
                        mycfg.vibml_update_table(src_lower, f"vml_point_all_auto_{src_lower}", df_skf_points_all, if_exists='replace', 
                                                primary=["source","server_name","point_id"], varchar_index=["source","server_name","point_id"])  
                    st.success(f'Done! : Bearing freq = {df_bearings.shape[0]:,} rows, Point cfg = {df_skf_points.shape[0]:,} rows, Point cfg (all) = {df_skf_points_all.shape[0]:,} row')
    if (main_menu.index(menu)==1) and (src_lower=="skf"):                
        with st.expander("ML configuration via csv file", expanded=True) :
            st.header("Existing ML configuration")
            col1, col2 = st.columns(cols_width_2_wide) 
            only_en = col1.checkbox("Only Enable ML", value=False)
            skf_srv_sel = col2.multiselect("SKF server", skf_srv_list, default=["SKF-BP1"])  
            #load ML point conf                  
            df_ml = load_vibml_ml_cfg(source=src_lower, server_id=skf_srv_sel, only_en=only_en)
            df_ml["start_predict"] = pd.to_datetime(df_ml["start_predict"]).dt.strftime("%Y-%m-%d")
            df_ml["start_spd"] = pd.to_datetime(df_ml["start_spd"]).dt.strftime("%Y-%m-%d")
            df_ml["last_updated"] = pd.to_datetime(df_ml["last_updated"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            cols = [c for c in df_ml.columns if c not in ["lookback_day","current_sd_day"]]
            df_ml = df_ml[cols]
            df_ml = df_ml.sort_values(["server_name","rootpath"], ascending=[True,True])
            st.write(df_ml.head(30).reset_index())                  
                    
            col1, col2, col3, _ = st.columns(cols_width_4)   
            en_ml_cnt = df_ml[df_ml.en_ml==1].shape[0]
            col1.write(f"Rows = {df_ml.shape[0]:,} | Cols = {df_ml.shape[1]:,} | EN ML = {en_ml_cnt:,}")   
            request_dl = col2.button("Request to Download")
            if request_dl :
                with st.spinner(f'Converting to csv files'):
                    df_ml_filter = df_ml.reset_index().copy()
                    df_ml_filter.insert(0,"updated", "")                    
                    df_ml_filter.insert(df_ml_filter.columns.get_loc("start_predict"),"_point_id", "")
                    df_ml_filter["_point_id"] = df_ml_filter["point_id"]                       
                    col3.markdown(myst.get_df_download_link(df_ml_filter, 
                                                        file_name=f"ml_point_cfg_{src_lower}.csv", file_label="ML point_cfg csv file"), unsafe_allow_html=True)
            uploaded_file = st.file_uploader('Upload ML configuration', type="csv")
            if uploaded_file :
                cols_str = ["updated","server_name","point_id","source","unit","point_name","description","rootpath","segment_name","asset_name",
                            "_point_id","plant","section","asset_group","asset_descp","machine_type",
                            "src_type","src_type_mms","src_pars","src_pars_mms","spd_tag","spd_tag_mms","note","start_predict","start_spd"]
                dtype = {c : object for c in cols_str}                    
                df_up = pd.read_csv(uploaded_file, dtype=dtype)                    
                df_up_filter = df_up[df_up.updated.isin(["y","Y","u","U"])&df_up.unit.isin(["gE","mm/s"])].copy()    
                df_up_filter[cols_str] = df_up_filter[cols_str].apply(lambda x: x.str.strip())
                cols_upper = ["plant","section","asset_group","asset_descp"]
                df_up_filter[cols_upper] = df_up_filter[cols_upper].apply(lambda x: x.str.upper())
                cols_int = ["en_ml","vib_type","src_id","src_id_mms"]                           
                df_up_filter[cols_int] = df_up_filter[cols_int].apply(lambda x : pd.to_numeric(x, errors="coerce")).fillna(0).astype(int)
                cols_float = ["mech_hi_x","elec_lo_rpm","rul_threshold","spd_factor","gear_factor","dia_mm"] 
                df_up_filter[cols_float] = df_up_filter[cols_float].apply(lambda x : pd.to_numeric(x, errors="coerce"))
                df_up_filter["start_predict"] = pd.to_datetime(df_up_filter["start_predict"], format="%Y-%m-%d", errors="coerce").dt.strftime("%Y-%m-%d")
                df_up_filter["start_predict"] = df_up_filter["start_predict"].fillna("2020-01-01")
                df_up_filter["start_spd"] = pd.to_datetime(df_up_filter["start_spd"], format="%Y-%m-%d", errors="coerce").dt.strftime("%Y-%m-%d")
                df_up_filter["start_spd"] = df_up_filter["start_spd"].fillna("2017-01-01")
                st.write(df_up_filter.head(30))   
                col1, col2, col3, _ = st.columns(cols_width_4)   
                col1.write(f"Update ML configure {src} : Rows = {df_up_filter.shape[0]:,} | Cols = {df_up_filter.shape[1]:,}") 
                update_ml = col2.button("Update ML configure")     
                if update_ml :
                    with st.spinner(f'Update ML configure to mysql'):                
                        df_result = pd.DataFrame(columns=["server_name","point_id","point_name","result"]).set_index(["server_name","point_id"])
                        for index, row in df_up_filter.iterrows():
                            #print(row)
                            r = mycfg.vibml_update_ml_cfg(row, cols_str)
                            df_result.loc[(row.server_name,row.point_id),["point_name","result"]] = (row.point_name, r)
                    col1.write(df_result)
                    refresh = col2.button("Refresh")                        
                    
        with st.expander("ML configuration one by one", expanded=False) :
            st.header("Coming soon")
    if (main_menu.index(menu)==2) :
        with st.expander(f"LAST SPEED CALCULATION", expanded=True) :
            with st.spinner(f'Loading from Database'):
                df_last_spd = mycfg.point_cfg_calc_spd(source=src_lower)
                df_last_spd["last_calc"] = pd.to_datetime(df_last_spd["last_calc"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                cols = ['vib_type_descp','unit','point_name','src_type','spd_tag','first_spd_avg', 'first_n_spd','last_calc', 'last_spd_avg', 'last_n_spd',
                        'status_text','status_text_ski','rootpath','segment_name', 'asset_name','plant', 'section',
                        'asset_group','asset_descp', 'machine_type','description']
                
                df_last_spd = df_last_spd[cols].reset_index().sort_values(["last_calc"], ascending=[False])
                
                st.write(df_last_spd.head(30).style.format(subset=['first_n_spd', 'first_spd_avg','last_n_spd', 'last_spd_avg'], formatter="{:.0f}"))
                
                col1, col2, col3, _ = st.columns(cols_width_4)   
                no_spd_cnt = df_last_spd[df_last_spd.first_spd_avg.isna()].shape[0]
                zero_spd_cnt = df_last_spd[df_last_spd.first_spd_avg==0].shape[0]
                col1.write(f"Rows = {df_last_spd.shape[0]:,} | Cols = {df_last_spd.shape[1]:,} | No Speed = {no_spd_cnt:,} | Zero Speed = {zero_spd_cnt:,}") 
                request_dl = col2.button("Request to Download")
                if request_dl :
                    with st.spinner(f'Converting to csv files'):                
                        df_last_spd.insert(0,"re-speed", "")                  
                        col3.markdown(myst.get_df_download_link(df_last_spd, 
                                                            file_name=f"last_spd_{src_lower}.csv", file_label="Last speed calc csv file"), unsafe_allow_html=True)
                uploaded_file = st.file_uploader('Upload points need to Re-calculate speed', type="csv")            
                if uploaded_file :
                    try :
                        df_up = pd.read_csv(uploaded_file) 
                        df_up_filter = df_up[df_up["re-speed"].isin(["y","Y","r","R","u","U"])].copy()
                        df_up_filter[["last_calc","last_spd_avg","last_predict","status_text","last_n_predict"]] = None, None, None, None, None
                        st.write(df_up_filter)
                        col1, col2, col3, _ = st.columns(cols_width_4)                 
                        recalc_spd = col1.button("Re-calc Speed")     
                        if recalc_spd :
                            pass
                    except :
                        st.error("Incorrect File upload !!!")
                        
        with st.expander(f"LAST ML PREDICTION ({pred_problem_meas})", expanded=True) :
            with st.spinner(f'Working on Last ML prediction'):
                df_last_predict = mycfg.point_cfg_predict(source=src_lower, pred_problem_meas=pred_problem_meas)
                df_last_predict["last_predict"] = pd.to_datetime(df_last_predict["last_predict"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                cols = ['vib_type_descp','unit','point_name','last_predict', 'status_text', 'last_n_predict',
                        'rootpath','segment_name', 'asset_name','plant', 'section',
                        'asset_group','asset_descp', 'machine_type','description']
                df_last_predict = df_last_predict[cols].reset_index()
                st.write(df_last_predict.head(30).style.format(subset=['last_n_predict'], formatter="{:.0f}"))
                
                col1, col2, col3, _ = st.columns(cols_width_4)   
                col1.write(f"Rows = {df_last_predict.shape[0]:,} | Cols = {df_last_predict.shape[1]:,}") 
                request_dl = col2.button("Request to Download", key="predict")
                if request_dl :
                    with st.spinner(f'- Converting to csv files'):               
                        df_last_predict.insert(0,"re-predict", "")         
                        col3.markdown(myst.get_df_download_link(df_last_predict, 
                                                            file_name=f"last_predict_{src_lower}.csv", file_label="Last ML predict csv file"), unsafe_allow_html=True)
                uploaded_file = st.file_uploader('Upload points need to Re-predict ML', type="csv")
                if uploaded_file :
                    try :
                        cols_str = ["updated","server_name","point_id","source","unit","point_name","rootpath","segment_name", "asset_name",
                                    "plant", "section", "asset_group","asset_descp", "machine_type","description"]
                        dtype = {c : object for c in cols_str}  
                        df_up = pd.read_csv(uploaded_file, dtype=dtype) 
                        df_up_filter = df_up[df_up["re-predict"].isin(["y","Y","r","R","u","U"])].copy()
                        df_up_filter[["last_predict","status_text","last_n_predict"]] = None, None, None
                        st.write(df_up_filter)
                        col1, col2, col3, _ = st.columns(cols_width_4)                        
                        col1.write(f"Rows = {df_up_filter.shape[0]:,} | Cols = {df_up_filter.shape[1]-1:,}") 
                        re_predict = col2.button("Re-predict ML") 
                        if re_predict :
                            df_up_filter = df_up_filter.rename(columns={'server_name' :'server_id'}) 
                            
                            #update last predict to mysql
                            
                            
                            #to API - re-calc baseline specific points
                            with st.spinner(f'... Send to Prediction API to re-calc baseline'):
                                cols = ["server_id", "point_id"]
                                payload = {'ids' : df_up_filter[cols].to_json(orient = "records")}                       
                                timeout = 10*df_up_filter.shape[0]  
                                status, status_text = 1, "send to re-calc baseline"                            
                                try :
                                    r = requests.post(api_baseline_path, data=json.dumps(payload), headers=headers, timeout=timeout)
                                    if r and (r.json()['status']) :
                                        df_r = pd.read_json(r.json()['results'])
                                        st.write(df_r)                                            
                                        st.markdown(myst.get_df_download_link(df_r, file_name=f"baseline_status_{src_lower}.csv", 
                                                                            file_label="Baseline update status csv file"), unsafe_allow_html=True)
                                    else :
                                        st.error("Bad response", )
                                except :
                                    st.error("API no response")
                    except :
                        st.error("Incorrect File upload !!!")
                            
        with st.expander(f"BASELINE CALCULATION", expanded=True) :
            col1, col2, col3, _ = st.columns(cols_width_4) 
            re_baseline = col1.button("Re-calc All baseline")     
            if re_baseline :
                #to API - re-calc baseline all points
                with st.spinner(f'... Send to Prediction API to re-calc baseline (All Points)'):
                    df_blank = pd.DataFrame(columns=["server_id","point_id"])
                    payload = {'ids' : df_blank.to_json(orient = "records")}                       
                    timeout = 800
                    status, status_text = 1, "send to re-calc baseline"                            
                    try :
                        r = requests.post(api_baseline_path, data=json.dumps(payload), headers=headers, timeout=timeout)
                        if r and (r.json()['status']) :
                            df_r = pd.read_json(r.json()['results'])
                            st.write(df_r)                                            
                            st.markdown(myst.get_df_download_link(df_r, file_name=f"baseline_status_all_{src_lower}.csv", 
                                                                    file_label="Baseline update status (all) csv file"), unsafe_allow_html=True)
                        else :
                            st.error("Bad response", )
                    except :
                        st.error("API no response")