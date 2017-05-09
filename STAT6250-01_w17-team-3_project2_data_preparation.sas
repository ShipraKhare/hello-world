*******************************************************************************;
**************** 80-character banner for column width reference ***************;
* (set window width to banner width to calibrate line length to 80 characters *;
*******************************************************************************;

* 
[Team Number] team-3
--
[Dataset 1 Name] mar_16
[Dataset Description] Average BART ridership for the month of March 2016
[Experimental Unit Description] BART Stations
[Number of Observations] 277
[Number of Features] 47
[Data Source]  http://www.bart.gov/sites/default/files/docs/ridership_2016.zip
[Data Dictionary]  http://www.bart.gov/sites/default/files/docs/Station_Names.xls
[Unique ID Schema] The first column in the data set, labeled UID (3-letter code
    for Unique Identification), numeric data type.
--
[Dataset 2 Name] Jan1_16
[Dataset Description] Hourly ridership report for January 1, 2016
[Experimental Unit Description] Number of riders, origin and destination with
    respect to hour traveled.
[Number of Observations] 21,934
[Number of Features] 5
[Data Source] http://64.111.127.166/origin-destination/date-hour-soo-dest-2016.csv.gz
[Data Dictionary] DATE: date of ride, HOUR: the hour of the ride, ENTRY: station
    of entry, EXIT: station of exit, NUM: number of riders who traveled from
    and to.
[Unique ID Schema] Columns Date, Hour, Entry, and Exit uniquely identify each 
    observation, as a composit unique ID.
--
[Dataset 3 Name] Mar31_16
[Dataset Description] Hourly ridership report for March 31, 2016
[Experimental Unit Description] Number of riders, origin and destination with 
    respect to hour traveled.
[Number of Observations] 29,155
[Number of Features] 5
[Data Source]  http://64.111.127.166/origin-destination/date-hour-soo-dest-2016.csv.gz
[Data Dictionary] DATE: date of ride, HOUR: the hour of the ride, ENTRY: station
    of entry, EXIT: station of exit, NUM: number of riders who traveled from and
    to.
[Unique ID Schema] Columns Date, Hour, Entry, and Exit uniquely identify each 
    observation, as a composit unique ID.
--
[Dataset 4 Name] Station_Names
[Dataset Description] Data dictionary reference of two-letter and four-letter 
    station identifiers keys used in entry and exit data sets.
[Experimental Unit Description] Station name equated to two and four letter 
    abbreviations.
[Number of Observations] 46
[Number of Features] 3
[Data Source]  http://www.bart.gov/sites/default/files/docs/Station_Names.xls
[Data Dictionary] This file is a name/value lookup table, data dictionary 
    for variable names.
[Unique ID Schema] The column Code_2 is the unique id.
--
[Dataset 5 Name] HomeOrigin
[Dataset Description] Ridership demographic study conducted by BART.
[Experimental Unit Description] BART station
[Number of Observations] 44
[Number of Features] 22
[Data Source]  http://www.bart.gov/about/reports/profile
[Data Dictionary]
              UID            - ID representing each station
              NO_V           - No Vehicle
              MORE_V         - More than one vehicle
              HIGH_FREQ      - High frequency Uses bart more than 5 times in a week
              AB_AV_FREQ     - Above Averaga frequency 3-4 times in a week
              AV_FREQ        - Avergae frequence 1-2 times a week
              BEL_AV_FREQ    - Below Frequencey 1-3 days per month
              LOW_FREQ       - Low frequency less than once  a month
              I_V_LOW        - Income very low (Under $25,000)
              I_LOW          - Income Low ($25,000 - $34,999
              I_BEL_AVG      - Income below Average($35,000 - $39,999)
              I_AVG          - Income Average($40,000 -49,999)
              I_AB_AVG       - Income above Average($50,000 -$59,999)
              I_HIGH         - Income high($60,000 -$74,999)
              I_V_HIGH       - Income very high($75,000 -$99,999)
              I_B_TOP        - Income below top($100,000 -$149,999)
              I_TOP          - Income Top($150,000+)
              Male           - Male
              Female         - Female
              Walk           - Walk
              Workplace      - work related travel
              W_BUS_APP      - appointment related to work
[Unique ID Schema] Column 1, UID
--
[Dataset 6 Name] Weekday_Entry_Eastbay_2016
[Dataset Description] Eastbay only weekday station entry
[Experimental Unit Description] Stations passenger numbers on weekday in the 
    east bay
[Number of Observations] 552
[Number of Features] 33
[Data Source] https://www.bart.gov/about/reports/ridership
[Data Dictionary] https://www.bart.gov/sites/default/files/docs/Station_Names.xls
[Unique ID Schema] UID
--
[Dataset 7 Name] sep23_16
[Dataset Description] Hourly ridership report for Sept. 23, 2016
[Experimental Unit Description] Number of riders, origin and destination with 
    respect to hour traveled.
[Number of Observations] 28,825
[Number of Features] 5
[Data Source]  http://64.111.127.166/origin-destination/date-hour-soo-dest-2016.csv.gz
[Data Dictionary] DATE: date of ride, HOUR: the hour of the ride, ENTRY: station
    of entry, EXIT: station of exit, NUM: number of riders who traveled from and
    to.
[Unique ID Schema] Columns Date, Hour, Entry, and Exit uniquely identify each 
    observation, as a composit unique ID.
--
[Dataset 8 Name] sep30_16
[Dataset Description] Hourly ridership report for Sept. 30, 2016
[Experimental Unit Description] Number of riders, origin and destination with 
    respect to hour traveled.
[Number of Observations] 28,832
[Number of Features] 5
[Data Source]  http://64.111.127.166/origin-destination/date-hour-soo-dest-2016.csv.gz
[Data Dictionary] DATE: date of ride, HOUR: the hour of the ride, ENTRY: station
    of entry, EXIT: station of exit, NUM: number of riders who traveled from and
    to.
[Unique ID Schema] Columns Date, Hour, Entry, and Exit uniquely identify each 
    observation, as a composit unique ID.
--
;

* setup environmental parameters;
%let mar16DatasetURL =
  /*https://github.com/stat6250/team-3_project2/blob/master/data/mar_16.xls?raw=true*/
    http://filebin.ca/3BljJfFK6653/mar_16.xls
;

%let homeOriginURL =
  /*https://github.com/stat6250/team-3_project2/blob/master/data/HomeOrigin.xls?raw=true*/
    http://filebin.ca/3CqUplYkveQw/HomeOrigin.xls
;

%let stnNameURL =
  /*https://github.com/stat6250/team-3_project2/blob/master/data/Station_Names.xls?raw=true*/
    http://filebin.ca/3CqVKzJN9RWB/Station_Names.xls
;

%let jan1URL =
  /*https://github.com/stat6250/team-3_project2/blob/master/data/Jan1_16.xls?raw=true*/
    http://filebin.ca/3CQT37enXRRb/Jan1_16.xls
;

%let mar31URL =
  /*https://github.com/stat6250/team-3_project2/blob/master/data/mar31_16.xls?raw=true*/
    http://filebin.ca/3CQaYKMNsSfI/mar31_16.xls
;

%let sep23URL =
  /*https://github.com/stat6250/team-3_project2/blob/master/data/sep23_16.xls?raw=true*/
    http://filebin.ca/3Csmgwrd3mbm/sep23_16.xls
;

%let sep30URL =
  /*https://github.com/stat6250/team-3_project2/blob/master/data/sep30_16.xls?raw=true*/
    http://filebin.ca/3Csn2UX5ultW/sep30_16.xls
;

%let ebEntryURL =
  /*https://github.com/stat6250/team-3_project2/blob/master/data/Weekday_Entry_Eastbay_2016.xls?raw=true*/
    http://filebin.ca/3CQehaxIvEiW/Weekday_Entry_Eastbay_2016.xls
;

* Loading raw datafile via Internet;
filename mar16 TEMP;
filename hOrgTEMP TEMP;
filename stnName TEMP;
filename jan1_TMP TEMP;
filename mr31_TMP TEMP;
filename ebay_TMP TEMP;
filename s23_TMP TEMP;
filename s30_TMP TEMP;

proc http
    method="get" 
    url="&mar16DatasetURL." 
    out=mar16
    ;
run;

proc http
    method="get" 
    url="&homeOriginURL." 
    out=hOrgTEMP
    ;
run;

proc http
    method="get" 
    url="&stnNameURL."
    out=stnName
    ;
run;

proc http
    method="get" 
    url="&jan1URL."
    out=jan1_TMP
    ;
run;

proc http
    method="get" 
    url="&mar31URL."
    out=mr31_TMP
    ;
run;

proc http
    method="get" 
    url="&ebEntryURL."
    out=ebay_TMP
    ;
run;

proc http
    method="get" 
    url="&sep23URL."
    out=s23_TMP
    ;
run;

proc http
    method="get" 
    url="&sep30URL."
    out=s30_TMP
    ;
run;

proc import
    file=mar16
    out=bart_raw
    dbms=xls
    ;
run;

proc import
    file=hOrgTEMP
    out=ho_raw
    dbms=xls
    ;
run;

proc import
    file=stnName
    out=stnName_raw
    dbms=xls
    ;
run;

proc import
    file=jan1_TMP
    out=jan1_raw
    dbms=xls
    ;
run;

proc import
    file=mr31_TMP
    out=mr31_raw
    dbms=xls
    ;
run;

proc import
    file=ebay_TMP
    out=ebay_raw
    dbms=xls
    ;
run;

proc import
    file=s23_TMP
    out=s23_raw
    dbms=xls
    ;
run;

proc import
    file=s30_TMP
    out=s30_raw
    dbms=xls
    ;
run;

filename mar16 clear;
filename hOrgTEMP clear;
filename stnName clear;
filename jan1_TMP clear;
filename mr31_TMP clear;
filename ebay_TMP clear;
filename s23_TMP clear;
filename s30_TMP clear;

* Build Data Set.  All data fields will be visible and accessed by field name.;
data barf;
    retain
        UID
        RM
        EN
        EP
        NB
        BK
        AS
        MA
        l9
        l2
        LM
        FV
        CL
        SL
        BF
        HY
        SH
        UC
        FM
        CN
        PH
        WC
        LF
        OR
        RR
        OW
        EM
        MT
        PL
        CC
        l6
        z4
        GP
        BP
        DC
        CM
        CV
        ED
        NC
        WP
        SS
        SB
        SO
        MB
        WD
        OA
        WS
    ;
    keep
        UID
        RM
        EN
        EP
        NB
        BK
        AS
        MA
        l9
        l2
        LM
        FV
        CL
        SL
        BF
        HY
        SH
        UC
        FM
        CN
        PH
        WC
        LF
        OR
        RR
        OW
        EM
        MT
        PL
        CC
        l6
        z4
        GP
        BP
        DC
        CM
        CV
        ED
        NC
        WP
        SS
        SB
        SO
        MB
        WD
        OA
        WS
    ;
    set bart_raw;
run;

data ho;
    retain
        UID
        Code_2
        Name
        NO_V
        MORE_V
        HIGH_FREQ
        AB_AV_FREQ
        AV_FREQ
        BEL_AV_FREQ
        LOW_FREQ
        I_V_LOW
        I_LOW
        I_BEL_AVG
        I_AVG
        I_AB_AVG
        I_HIGH
        I_V_HIGH
        I_B_TOP
        I_TOP
        Male
        Female
        Walk
        Workplace
        W_BUS_APP
    ;
    keep
        UID
        Code_2
        Name
        NO_V
        MORE_V
        HIGH_FREQ
        AB_AV_FREQ
        AV_FREQ
        BEL_AV_FREQ
        LOW_FREQ
        I_V_LOW
        I_LOW
        I_BEL_AVG
        I_AVG
        I_AB_AVG
        I_HIGH
        I_V_HIGH
        I_B_TOP
        I_TOP
        Male
        Female
        Walk
        Workplace
        W_BUS_APP
    ;
    set ho_raw;
run;

data stn;
    retain
        Code_2
        Code_4
        Name
    ;
    keep
        Code_2
        Code_4
        Name
    ;
    set stnName_raw;
run;

data jan1;
    retain
        DATE
        HOUR
        ENTRY
        EXIT
        NUM
    ;
    keep
        DATE
        HOUR
        ENTRY
        EXIT
        NUM
    ;
    set jan1_raw;
run;

data mar31;
    retain
        DATE
        HOUR
        ENTRY
        EXIT
        NUM
    ;
    keep
        DATE
        HOUR
        ENTRY
        EXIT
        NUM
    ;
    set mr31_raw;
run;

data ebay;
    retain
        UID
        RM
        EN
        EP
        NB
        BK
        AS
        MA
        l9
        l2
        LM
        FV
        CL
        SL
        BF
        HY
        SH
        UC
        FM
        CN
        PH
        WC
        LF
        OR
        RR
        OW
        CV
        ED
        NC
        WP
        WD
        OA
        WS
    ;
    keep
        UID
        RM
        EN
        EP
        NB
        BK
        AS
        MA
        l9
        l2
        LM
        FV
        CL
        SL
        BF
        HY
        SH
        UC
        FM
        CN
        PH
        WC
        LF
        OR
        RR
        OW
        CV
        ED
        NC
        WP
        WD
        OA
        WS
    ;
    set ebay_raw;
run;

data sep23;
    retain
        DATE
        HOUR
        ENTRY
        EXIT
        NUM
    ;
    keep
        DATE
        HOUR
        ENTRY
        EXIT
        NUM
    ;
    set s23_raw;
run;

data sep30;
    retain
        DATE
        HOUR
        ENTRY
        EXIT
        NUM
    ;
    keep
        DATE
        HOUR
        ENTRY
        EXIT
        NUM
    ;
    set s30_raw;
run;

* Vertical concatenation of hourly ridership - for two days of data: January 
  1st and March 31;
data jan_mar;
    set jan1 mar31;
run;

* Vertically stacking observations for Home Origin data in order to match number
  of observations in Bay Area Ridership File (a.k.a. barf), preprocessing necessary
  for next data step;
data mult_cat;
    set ho ho ho ho ho ho;
run;

* Horizontal one-to-one merging Home Orgin data with barf data file;
data barf_interlv;
    set work.barf;
    set work.mult_cat;
run;

* Sorting data by descending workplace;
proc sort data = barf_interlv out = barf_interlv_wrkplace_sort ;
    by DESCENDING workplace;
run;

* Sorting data by descending South Hayward Station;
proc sort data = barf_interlv out = barf_interlv_SH ;
    by DESCENDING SH;
run;

* Creating data obj specializes on first three hours of day for Embr station;
data jan_mar_EMBR;            
    set jan1 mar31;
    where ENTRY = 'EMBR' and HOUR in (0 1 2);
run;

* Sorting BART arrival data during a Giants baseball game;
proc sort data=work.sep30 out=work.arrv;
    by exit;
run;
* Sorting BART arrival data during Giants baseball off season;
proc sort data=work.mar31 out=work.arrv_off;
    by exit;
run;

* The next proc sort and two data steps create a SAS data object which represent
  the number of people exiting BART during the morning weekday rush hour in the 
  financial district, San Francisco;
proc sort data=work.sep30 out=work.temp_a;
    by exit hour;
run;
data work.f_dist_stns_rush;
    set work.temp_a;
    where exit in ('MONT', 'EMBR') and hour in (7, 8, 9);
run;

data work.f_dist_stns_rush_pm;
    set work.temp_a;
    where exit in ('MONT', 'EMBR') and hour in (16, 17, 18);
run;

data work.civic_center_am_rush;
    set work.temp_a;
    where exit in ('CIVC') and hour in (7, 8, 9);
run;

data work.merge_am_rush;
    set work.f_dist_stns_rush work.civic_center_am_rush;
run;

* Creates a SAS data set which represents the number of people exiting    ;
* the Montgomery and Embarcadaro BART stations on a typical work day, AM. ;
data work.m_rush;
    set work.f_dist_stns_rush;
    by exit hour;
    if first.exit then tot_num=0;
    tot_num+num;
    if last.exit and last.hour;
run;

* Creates a SAS data set which represents the number of people exiting    ;
* the Montgomery and Embarcadaro BART stations on a typical work day, PM. ;
data work.pm_rush;
    set work.f_dist_stns_rush_pm;
    by exit hour;
    if first.exit then tot_num=0;
    tot_num+num;
    if last.exit and last.hour;
run;

* The next four data steps incrementally collate merge and calculate percent;
* change.  These four data steps are candidates for data step consolidation,;
* but for now they work together and should be considered a unit.           ;
data pop_avg_off;
    set work.arrv_off;
    by hour;
    where exit in ('MONT', 'EMBR', 'POWL') and hour in (14);
    do;
        no_game_ridership+num;
    end;
    if last.hour then
    do;
        output;
    end;
run;
data pop_avg_game;
    set work.arrv;
    by hour;
    where exit in ('MONT', 'EMBR', 'POWL') and hour in (14);
    do;
        game_ridership+num;
    end;
    if last.hour then
    do;
        output;
    end;
run;
data match_merge_records;
    set pop_avg_game;
    set pop_avg_off;
    prcnt_chng = ((game_ridership - no_game_ridership)/no_game_ridership) * 100;
    keep prcnt_chng;
run;
data compare_am_pm_rush;
    merge work.m_rush (rename=(tot_num=am_tot_num)) work.pm_rush (rename=(tot_num=pm_tot_num));
    percent_change = ((pm_tot_num-am_tot_num)/am_tot_num)*100;
    keep exit percent_change;
run;