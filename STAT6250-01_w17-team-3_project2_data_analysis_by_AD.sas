*******************************************************************************;
**************** 80-character banner for column width reference ***************;
* (set window width to banner width to calibrate line length to 80 characters *;
*******************************************************************************;


*
This file uses the following analytic dataset to address several research
questions regarding ridership data on the Bay Area Rapid Transit system, BART.
Dataset Name: barf, ho, barf_interlv, arrv created in external file
STAT6250-01_w17-team-3_project2_data_preparation.sas, which is assumed to be
in the same directory as this file.
See included file for dataset properties
;

%let dataPrepFileName = STAT6250-01_w17-team-3_project2_data_preparation.sas;
%let sasUEFilePrefix = team-3_project2;


%macro setup;
%if
	&SYSSCP. = WIN
%then
	%do;
		X "cd ""%substr(%sysget(SAS_EXECFILEPATH),1,%eval(%length(%sysget(SAS_EXECFILEPATH))-%length(%sysget(SAS_EXECFILENAME))))""";			
		%include ".\&dataPrepFileName.";
	%end;
%else
	%do;
		%include "~/&sasUEFilePrefix./&dataPrepFileName.";
	%end;
%mend;
%setup;

*******************************************************************************;
* Research Question Correlation analysis between  percentage of riders
   with no vehicle and percentage of high frequency riders;
*******************************************************************************;
TITLE1 "Research Question: For each station is there is a correlation between
the percentage of riders with no vehicle and percentage of high frequency riders"
;

TITLE2 "Rationale:This would help us to find if the percentage of riders with no
vehicle are also part of high frequency riders"
;

FOOTNOTE1 " We cannot make any statistical conclusion based on the output  since the p-value for the results are really high"
;

*
Methodology: This uses a proc corr to find the correlation between percentage
of riders with no vehicles and percentage of high frequency riders who
are taking bart in each station"
;

PROC CORR 
	DATA = barf_interlv;
	VAR NO_V HIGH_FREQ;
RUN;

TITLE;
FOOTNOTE;


*******************************************************************************;
* Research Question Analysis Find the stations with minimum and maximum percentage
   of low frequency riders;
*******************************************************************************;

TITLE2 "Research Question: Find the stations which has the highest percentage 
of low frequency riders ?"
;

TITLE2 "Rationale:This would help the business to see where there are less 
frequent riders and see if something needs to be improved in that 
station like make it clean, building elevators or other facilities 
which would help us improve the station in general to increase the frequent riders."
;

FOOTNOTE1 "The above tables shows that Dublin/Pleasonton station has highest percentage 
of low frequency rider"
;

*
Methodology: 1.PROC MEANS to find min and max values 
;

PROC MEANS 
	DATA=barf_interlv NOPRINT MEAN;
	ID name;
	VAR LOW_FREQ;
	OUTPUT 
    OUT =  TEMP_MEAN_LOW_FREQ (DROP = _TYPE_ _FREQ_  NAME); 
RUN;

PROC PRINT 
	DATA= TEMP_MEAN_LOW_FREQ NOOBS LABEL ;
    WHERE _STAT_ = 'MAX';
    LABEL _STAT_ = 'Highest percentage of low frequency riders';
    BY _STAT_;
RUN;
	
title;
footnote;

*******************************************************************************;
* Research Question highest percentage of walkers to BART;
*******************************************************************************;

TITLE1 "Research Question: Find the station which has highest percentage of people walking to 
station ?"
;
 
TITLE2 "Rationale:This would help the public transport to introduce more 
transportation facilities."
;

FOOTNOTE1 "24th St. Mission station has highest percentage of people 
walking to the BART station"
;

*
Methodology: Use PROC MEANS to calculate the max percentages of walkers to BART
;

PROC MEANS 
	DATA=barf_interlv NOPRINT MEAN;
	ID name;
	VAR Walk;
	OUTPUT 
    OUT =   TEMP_WALK_MEAN (DROP = _TYPE_ _FREQ_ name ); 
RUN;

PROC PRINT 
	DATA=TEMP_WALK_MEAN NOOBS LABEL ;
    WHERE _STAT_ = 'MAX';
    LABEL _STAT_ = 'Highest percentage of walkers to Bart station';
    BY _STAT_;
RUN;
	
TITLE;
FOOTNOTE;



	