import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import numpy as np
from datetime import date
import os 

# -------------------------------------------------------------------
# FUNCTIES VOOR DATA INLADEN EN OPSCHONEN
# -------------------------------------------------------------------

@st.cache_data(show_spinner="Bezig met inladen en opschonen van historische data...")
def load_all_data(directory_path='.'):
    """
    Zoekt alle CSV-bestanden, groepeert ze per station-ID (uit de bestandsnaam) 
    en retourneert een dictionary van DataFrames.
    
    LET OP: Gebruikt 'glob(..., recursive=True)' om nu ook in submappen te zoeken.
    """
    # *** AANPASSING VOOR RECURSIEVE ZOEKTOCHT: **/ *.csv en recursive=True ***
    all_files = glob.glob(directory_path + "/**/*.csv", recursive=True)
    
    if not all_files:
        return None, "Geen CSV-bestanden gevonden in de map."

    station_dfs_raw = {}
    
    def clean_value(series):
        series = series.astype(str).str.replace(' °C| °%| °hPa', '', regex=True)
        series = series.str.replace(' ', '').str.replace(',', '', regex=True)
        series = series.replace('--', pd.NA)
        return pd.to_numeric(series, errors='coerce')

    COLUMNS = [
        "Date", "Temp_High_C", "Temp_Avg_C", "Temp_Low_C",
        "DewP_High_C", "DewP_Avg_C", "DewP_Low_C",
        "Hum_High_P", "Hum_Avg_P", "Hum_Low_P",
        "Speed_High", "Speed_Avg", "Speed_Low",
        "Pres_High_hPa", "Pres_Low_hPa", "Precip_Sum"
    ]
    
    # Bepaal de Station ID op basis van de bestandsnaam
    def get_station_id(filename):
        base = os.path.basename(filename)
        if '_' in base:
            # Gebruik het deel voor de eerste underscore als ID
            return base.split('_')[0]
        else:
            # Standaard ID voor bestanden zonder underscore
            return "Station_Default"

    for filename in all_files:
        try:
            station_id = get_station_id(filename)
            
            df = pd.read_csv(filename, skiprows=[1])
            df.columns = COLUMNS
            
            cols_to_clean = df.columns.drop('Date')
            df[cols_to_clean] = df[cols_to_clean].apply(clean_value)
            
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            df = df.set_index('Date')
            
            if station_id not in station_dfs_raw:
                station_dfs_raw[station_id] = []
            station_dfs_raw[station_id].append(df)
            
        except Exception as e:
            st.warning(f"Fout bij het inlezen van bestand {filename} (fout: {e}). Bestand overgeslagen.")
            
    final_station_dfs = {}
    total_files = 0
    
    # Consolideer alle bestanden per station tot één DataFrame
    for station_id, list_of_dfs in station_dfs_raw.items():
        total_files += len(list_of_dfs)
        final_df = pd.concat(list_of_dfs).sort_index()
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        
        final_df['Pres_Avg_hPa'] = (final_df['Pres_High_hPa'] + final_df['Pres_Low_hPa']) / 2
        
        numeric_cols = final_df.select_dtypes(include=np.number).columns
        final_df[numeric_cols] = final_df[numeric_cols].astype(float)
        
        final_station_dfs[station_id] = final_df
    
    return final_station_dfs, f"Succesvol {total_files} maandbestanden van {len(final_station_dfs)} stations geconsolideerd."


# -------------------------------------------------------------------
# FUNCTIE: VINDT AANEENGESLOTEN PERIODES
# -------------------------------------------------------------------
def find_consecutive_periods(df_filtered, min_days, temp_column):
    """Vindt en retourneert aaneengesloten periodes in een gefilterde DataFrame."""
    
    if df_filtered.empty:
        return pd.DataFrame(), 0

    df_groups = df_filtered.copy()
    
    # Bepaal of dit de start van een nieuwe periode is
    df_groups['new_period'] = df_groups.index.to_series().diff().dt.days.fillna(0) > 1
    df_groups['group_id'] = df_groups['new_period'].cumsum()
    df_groups['Date'] = df_groups.index.to_series() 
    
    # Aggregatie
    periods = df_groups.groupby('group_id').agg(
        StartDatum=('Date', 'min'),
        EindDatum=('Date', 'max'),
        Duur=('Date', 'size'),
        Gemiddelde_Temp_Periode=(temp_column, 'mean')
    )
    
    # Filter op de minimale duur
    periods = periods[periods['Duur'] >= min_days]
    
    # Formatteer voor weergave
    periods['StartDatum'] = periods['StartDatum'].dt.strftime('%d-%m-%Y')
    periods['EindDatum'] = periods['EindDatum'].dt.strftime('%d-%m-%Y')
    periods['Gemiddelde_Temp_Periode'] = periods['Gemiddelde_Temp_Periode'].apply(lambda x: f"{x:.1f} °C")

    total_periods = len(periods)
    
    return periods.reset_index(drop=True), total_periods


# -------------------------------------------------------------------
# HOOFD SCRIPT: DATA INLADEN EN APP LAYOUT
# -------------------------------------------------------------------

st.set_page_config(layout="wide") 
st.title("🌡️ Historische Weer Data Analyse")

# 1. Data Inladen
# Hier geef je aan dat de data in de submap 'data' staat.
all_stations_data, message = load_all_data('data') 

if all_stations_data is None or not all_stations_data:
    st.error(f"Fout: Geen bruikbare data gevonden. {message}")
    st.stop()

# 2. Station Selection
available_stations = sorted(all_stations_data.keys())

st.sidebar.header("Configuratie")
st.sidebar.markdown("---")


# Expander voor Stations Selectie
with st.sidebar.expander("Kies Weerstation", expanded=True):
    if len(available_stations) > 1:
        selected_station = st.selectbox(
            "Selecteer Weerstation voor Analyse:",
            available_stations,
            key="station_selector"
        )
    elif available_stations:
        selected_station = available_stations[0]
        st.info(f"Huidige Analyse: **{selected_station}** (Enige station)")
    else:
        st.error("Geen bruikbare stationsdata gevonden na inladen.")
        st.stop()
    
    df_full = all_stations_data[selected_station]
    st.markdown(f"**Geselecteerd:** {selected_station}")


# -------------------------------------------------------------------
# 3. TABBLAD NAVIGATIE
# -------------------------------------------------------------------

tab1, tab2 = st.tabs(["Maand/Jaar Analyse", "Historische Zoekfunctie"])


# -------------------------------------------------------------------
# FUNCTIE 1: MAAND/JAAR ANALYSE 
# -------------------------------------------------------------------

# Expander voor Maand/Jaar Analyse Filters
with st.sidebar.expander("Maand/Jaar Analyse Filters", expanded=True):
    analysis_type = st.radio(
        "Kies Analyse Niveau:",
        ["Maand", "Jaar"],
        key="analysis_level"
    )

    if analysis_type == "Maand":
        # df_full wordt nu dynamisch gefilterd
        df_full['JaarMaand'] = df_full.index.to_period('M')
        available_periods = sorted(df_full['JaarMaand'].unique().astype(str), reverse=True)

        selected_period_str = st.selectbox(
            "Selecteer Maand:",
            available_periods,
            key="monthly_selector"
        )
        
        df_selected = df_full[df_full.index.to_period('M') == selected_period_str]
        selected_period = pd.to_datetime(selected_period_str)
        titel_periode = selected_period.strftime('%B %Y')
        
    else:
        available_years = sorted(df_full.index.year.unique(), reverse=True)

        selected_year = st.selectbox(
            "Selecteer Jaar:",
            available_years,
            key="yearly_selector"
        )
        
        df_selected = df_full[df_full.index.year == selected_year]
        titel_periode = str(selected_year)

    df_month = df_selected

with tab1:
    st.header(f"Gedetailleerde Analyse voor {titel_periode} (Station: {selected_station})")
    st.markdown("---")
    
    if df_month.empty:
        st.warning(f"Geen data gevonden voor de geselecteerde periode: {titel_periode}.")
    else:
        avg_temp = df_month['Temp_Avg_C'].mean()
        max_temp = df_month['Temp_High_C'].max()
        min_temp = df_month['Temp_Low_C'].min()
        
        max_date_str = df_month['Temp_High_C'].idxmax().strftime('%d-%m-%Y')
        min_date_str = df_month['Temp_Low_C'].idxmin().strftime('%d-%m-%Y')

        avg_pres_month = df_month['Pres_Avg_hPa'].mean() 
        
        st.subheader("Overzicht en Kerncijfers van de Periode")
        col1, col2, col3, col4 = st.columns(4) 

        with col1:
            st.metric(label="Gemiddelde Temp", value=f"{avg_temp:.1f} °C")

        with col2:
            st.metric(
                label="Hoogste Max Temp",
                value=f"{max_temp:.1f} °C",
                delta=f"Op: {max_date_str}" if analysis_type == "Jaar" else f"Op: {df_month['Temp_High_C'].idxmax().strftime('%d-%m')}"
            )

        with col3:
            st.metric(
                label="Laagste Min Temp",
                value=f"{min_temp:.1f} °C",
                delta=f"Op: {min_date_str}" if analysis_type == "Jaar" else f"Op: {df_month['Temp_Low_C'].idxmin().strftime('%d-%m')}"
            )

        with col4:
            st.metric(label="Gemiddelde Luchtdruk", value=f"{avg_pres_month:.2f} hPa")

        st.markdown("---")

        st.subheader(f"Temperatuur Trend ({titel_periode})")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=df_month.index, y=df_month['Temp_High_C'], mode='lines+markers', name="Max Temp", line=dict(color='red')))
        fig_temp.add_trace(go.Scatter(x=df_month.index, y=df_month['Temp_Avg_C'], mode='lines+markers', name="Gem Temp", line=dict(color='blue', dash='dash')))
        fig_temp.add_trace(go.Scatter(x=df_month.index, y=df_month['Temp_Low_C'], mode='lines+markers', name="Min Temp", line=dict(color='green')))
        fig_temp.update_layout(
            title=f"Temperatuurverloop in {titel_periode}",
            xaxis_title="Datum", yaxis_title="Temperatuur (°C)", hovermode="x unified", template="plotly_white"
        )
        st.plotly_chart(fig_temp, use_container_width=True)


# -------------------------------------------------------------------
# FUNCTIE 2: HISTORISCHE ZOEKEN
# -------------------------------------------------------------------

# Expander voor Historische Zoekfilters
with st.sidebar.expander("Historische Zoekfilters", expanded=False):
    st.markdown(f"**1. Zoekperiode**")
    period_options = ["Onbeperkt (Volledige Database)", "Selecteer Jaar", "Selecteer Maand", "Aangepaste Datums"]
    period_type = st.selectbox("1. Zoekperiode", period_options, key="period_select_hist")
    
    # df_full is nu de data van het geselecteerde station
    df_filtered_time = df_full.copy()
    
    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()
    
    if period_type == "Selecteer Jaar":
        available_years = sorted(df_full.index.year.unique())
        st.selectbox("Kies Jaar:", available_years, key="hist_year")
        df_filtered_time = df_full[df_full.index.year == st.session_state.hist_year]
    
    elif period_type == "Selecteer Maand":
        df_temp_filter = df_full.copy()
        df_temp_filter['JaarMaand'] = df_temp_filter.index.to_period('M').astype(str)
        available_periods = sorted(df_temp_filter['JaarMaand'].unique(), reverse=True)
        
        selected_period_str = st.selectbox("Kies Jaar en Maand (YYYY-MM):", available_periods, key="hist_year_month")
        
        df_filtered_time = df_temp_filter[df_temp_filter['JaarMaand'] == selected_period_str].drop(columns=['JaarMaand'])

    elif period_type == "Aangepaste Datums":
        date_range = st.date_input(
            "Kies Start- en Einddatum:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="hist_dates"
        )
        st.caption("Let op: het getoonde datumformaat is afhankelijk van uw browserinstellingen.")
        
        if len(date_range) == 2:
             df_filtered_time = df_full.loc[str(date_range[0]):str(date_range[1])]

    st.markdown("---")
    st.markdown(f"**2. Filtertype & Drempel**")

    filter_mode = st.radio(
        "Zoeken op:",
        ["Losse Dagen", "Aaneengesloten Periode", "Hellmann Getal Berekenen"],
        key="filter_mode"
    )

    # Initialiseer variabelen buiten de filter-modus
    temp_column = 'Temp_Avg_C'
    comparison = "Lager dan (<=)" # Default voor Hellmann
    temp_threshold = 0.0
    min_consecutive_days = 0

    # -------------------------------------------------------------------
    # B. DYNAMISCHE FILTERS (Binnen de Expander)
    # -------------------------------------------------------------------

    if filter_mode == "Hellmann Getal Berekenen":
        st.markdown("---")
        st.markdown(f"**3. Hellmann Berekening**")
        st.markdown("**Basis:** Absolute som van Gemiddelde Dagtemp $\\le 0.0$ °C.")
        
    elif filter_mode == "Aaneengesloten Periode":
        st.markdown("---")
        st.markdown(f"**3. Periodedrempel**")
        min_consecutive_days = st.number_input(
            "Min. aaneengesloten dagen:",
            min_value=2,
            value=3,
            step=1,
            key="min_days_period"
        )
        st.markdown("---")
        st.markdown(f"**4. Temperatuurfilter**")
        temp_type = st.selectbox(
            "Meetwaarde:",
            ["Max Temp (Temp_High_C)", "Min Temp (Temp_Low_C)", "Gemiddelde Temp (Temp_Avg_C)"],
            key="temp_type_period" 
        )
        # Extract the actual column name from the full string
        temp_column = temp_type.split(" (")[1][:-1] if " (" in temp_type else 'Temp_Avg_C'
        
        comparison = st.radio("Vergelijking:", ["Hoger dan (>=)", "Lager dan (<=)"], key="comparison_period") 
        
        temp_threshold = st.number_input(
            "Temperatuur (°C):",
            value=15.0, 
            step=0.5,
            key="temp_threshold_period"
        )

    else: # Losse Dagen
        st.markdown("---")
        st.markdown(f"**4. Temperatuurfilter**")
        temp_type = st.selectbox(
            "Meetwaarde:",
            ["Max Temp (Temp_High_C)", "Min Temp (Temp_Low_C)", "Gemiddelde Temp (Temp_Avg_C)"],
            key="temp_type_days"
        )
        # Extract the actual column name from the full string
        temp_column = temp_type.split(" (")[1][:-1] if " (" in temp_type else 'Temp_Avg_C'
        
        comparison = st.radio("Vergelijking:", ["Hoger dan (>=)", "Lager dan (<=)"], key="comparison_days")
        
        temp_threshold = st.number_input(
            "Temperatuur (°C):",
            value=15.0, 
            step=0.5,
            key="temp_threshold_days"
        )


# -------------------------------------------------------------------
# C. FILTEREN (Uitvoering)
# -------------------------------------------------------------------

df_filtered_condition = df_filtered_time.copy()

# Apply temperature filter based on mode
if filter_mode == "Hellmann Getal Berekenen":
    # Hellmann is always Temp_Avg_C <= 0.0
    df_filtered_condition = df_filtered_condition[df_filtered_condition['Temp_Avg_C'] <= 0.0].copy()
elif comparison == "Hoger dan (>=)":
    df_filtered_condition = df_filtered_condition[df_filtered_condition[temp_column] >= temp_threshold].copy()
elif comparison == "Lager dan (<=)":
    df_filtered_condition = df_filtered_condition[df_filtered_condition[temp_column] <= temp_threshold].copy()


# -------------------------------------------------------------------
# D. RESULTATEN WEERGAVE (Rechts in Hoofdscherm - Tab 2)
# -------------------------------------------------------------------

with tab2:
    st.header(f"Historische Zoekfunctie: Resultaten (Station: {selected_station})")
    
    # -------------------------------------------------------------
    # TOEGEVOEGDE LOGICA: SAMENVATTING VAN DE ZOEKSPECIFICATIES
    # -------------------------------------------------------------
    search_summary_parts = []
    
    if not df_filtered_time.empty:
        min_res_date = df_filtered_time.index.min().strftime('%d-%m-%Y')
        max_res_date = df_filtered_time.index.max().strftime('%d-%m-%Y')
        search_summary_parts.append(f"**Periode:** {min_res_date} t/m {max_res_date}")
    
    if filter_mode == "Hellmann Getal Berekenen":
        search_summary_parts.append("**Modus:** Hellmann Getal Berekenen (Gem. Temp $\\le$ 0.0 °C)")
    
    elif filter_mode == "Aaneengesloten Periode":
        # Get label/comparison from the session state keys used in the sidebar
        temp_label = st.session_state.temp_type_period.split(" (")[0]
        comp_symbol = ">=" if comparison == "Hoger dan (>=)" else "<="
        
        search_summary_parts.append(f"**Modus:** Aaneengesloten Periode ({min_consecutive_days}+ dagen)")
        search_summary_parts.append(f"**Drempel:** {temp_label} {comp_symbol} **{temp_threshold:.1f} °C**")
    
    elif filter_mode == "Losse Dagen":
        # Get label/comparison from the session state keys used in the sidebar
        temp_label = st.session_state.temp_type_days.split(" (")[0]
        comp_symbol = ">=" if comparison == "Hoger dan (>=)" else "<="
        
        search_summary_parts.append("**Modus:** Losse Dagen")
        search_summary_parts.append(f"**Drempel:** {temp_label} {comp_symbol} **{temp_threshold:.1f} °C**")

    # Display the summary
    if search_summary_parts:
        st.subheader("Zoekopdracht Samenvatting")
        st.info(" | ".join(search_summary_parts))
    st.markdown("---")
    # -------------------------------------------------------------
    # EINDE TOEGEVOEGDE LOGICA
    # -------------------------------------------------------------

    if df_filtered_condition.empty and (period_type != "Selecteer Maand" or 'hist_year_month' not in st.session_state):
        st.info("Selecteer eerst een filter om resultaten te zien.")
        
    elif filter_mode == "Losse Dagen":
        # Logic for Single Days
        df_final = df_filtered_condition
        
        st.subheader(f"Resultaten ({len(df_final)} losse dagen gevonden)")

        if len(df_final) == 0:
            st.info("Geen dagen gevonden die voldoen aan de ingestelde filters.")
        else:
            start_date_str = df_final.index.min().strftime('%d-%m-%Y')
            end_date_str = df_final.index.max().strftime('%d-%m-%Y')
            st.metric(
                label="Totaal aantal dagen gevonden:",
                value=f"{len(df_final)} dagen",
                delta=f"Periode: {start_date_str} t/m {end_date_str}"
            )
            st.subheader("Gevonden Dagen")
            df_display = df_final.copy()
            df_display.index = df_display.index.strftime('%d-%m-%Y')
            df_display.index.name = "Datum"
            st.dataframe(df_display[['Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C', 'Hum_Avg_P', 'Pres_Avg_hPa', 'Precip_Sum']], use_container_width=True)
            
            st.subheader("Temperatuurdistributie van de Gevonden Dagen")
            
            # Determine the correct label for the histogram x-axis
            temp_label = st.session_state.temp_type_days.split(" (")[0]
            
            fig_hist = px.histogram(
                df_final, x=temp_column, nbins=30, 
                title=f"Temperatuurdistributie ({temp_label})",
                labels={temp_column: f"{temp_label} (°C)"}, 
                template="plotly_white"
            )
            fig_hist.add_vline(x=temp_threshold, line_width=2, line_dash="dash", line_color="red", annotation_text="Drempel")
            st.plotly_chart(fig_hist, use_container_width=True)


    elif filter_mode == "Aaneengesloten Periode":
        # Logic for Consecutive Periods (Temperature Threshold)
        
        df_periods, total_periods = find_consecutive_periods(df_filtered_condition, min_consecutive_days, temp_column)

        st.subheader(f"Resultaten ({total_periods} periodes gevonden)")

        if df_periods.empty:
            st.info(f"Geen aaneengesloten periodes van {min_consecutive_days} of meer dagen gevonden die aan de filters voldoen.")
        else:
            totaal_dagen = df_periods['Duur'].sum()
            st.metric(
                label=f"Totaal {min_consecutive_days}+ dagen periodes gevonden:",
                value=f"{total_periods} periodes",
                delta=f"Totaal: {totaal_dagen} dagen"
            )
            df_periods.rename(columns={'StartDatum': 'Startdatum', 'EindDatum': 'Einddatum', 'Duur': 'Aantal Dagen'}, inplace=True)
            st.subheader("Gevonden Periodes")
            st.dataframe(df_periods, use_container_width=True)
            fig_bar = px.bar(
                df_periods, x=df_periods.index, y='Aantal Dagen', color='Gemiddelde_Temp_Periode',
                title="Gevonden Periodes per Duur en Gem. Temp",
                labels={'x': 'Periode Index', 'Aantal Dagen': 'Aantal Dagen'},
                hover_data=['Startdatum', 'Einddatum', 'Gemiddelde_Temp_Periode'], template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)


    elif filter_mode == "Hellmann Getal Berekenen":
        # Logic for Hellmann Number Calculation (Fixed period summation)
        
        hellmann_value = abs(df_filtered_condition['Temp_Avg_C'].sum())
        aantal_vorstdagen = len(df_filtered_condition)
        
        st.subheader("Hellmann Getal Resultaat")
        
        if aantal_vorstdagen == 0:
            st.info("Geen dagen met een negatieve gemiddelde temperatuur gevonden in de geselecteerde periode.")
        else:
            
            st.metric(
                label="Hellmann Getal voor de Periode:",
                value=f"{hellmann_value:.1f}",
                delta=f"Totaal: {aantal_vorstdagen} vorstdagen"
            )
            
            st.subheader("Alle Vorstdagen (Tgem $\\le$ 0.0 °C)")
            
            # Toon de vorstdagen
            df_display = df_filtered_condition[['Temp_Avg_C', 'Temp_Low_C']].copy()
            df_display['Vorstbijdrage'] = df_display['Temp_Avg_C'] # Voor de duidelijkheid
            df_display['Vorstbijdrage'] = df_display['Vorstbijdrage'].apply(lambda x: f"{x:.1f} °C")
            df_display.rename(columns={'Temp_Low_C': 'Min Temp (°C)', 'Temp_Avg_C': 'Gem Temp (°C)', 'Vorstbijdrage': 'Vorstbijdrage'}, inplace=True)
            df_display.index = df_display.index.strftime('%d-%m-%Y')
            df_display.index.name = "Datum"

            st.dataframe(df_display, use_container_width=True)
            
            # Visualisatie
            fig_bar = px.bar(
                df_filtered_condition, 
                x=df_filtered_condition.index, 
                y='Temp_Avg_C', 
                title="Negatieve Gemiddelde Dagtemperaturen (Bijdrage aan Hellmann Getal)",
                labels={'Temp_Avg_C': 'Gemiddelde Temp (°C)', 'index': 'Datum'},
                template="plotly_white",
                color=df_filtered_condition['Temp_Avg_C'],
                color_continuous_scale=px.colors.sequential.thermal_r # Koude kleuren
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)


# Console message at the bottom of the sidebar
st.sidebar.markdown("---")
st.sidebar.caption(message)

# End of the script
