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
* Research Question Analysis Starting Point                                    ;
*******************************************************************************;
*
Methodology: Preprocessing within the data preparation file will sort on
exit data, creating a nice work.arrv and work.arrv_off SAS data sets upon which
one can specify the variables upon which to display.  Using a branch conditional
(that is an if-then statement) one can then specify hour and station
of egress.  Summing exits we find the number exiting the most popular
stations during the time for a Giants baseball game in San Francisco.
Then we will do the same thing for a non-ballgame day, and conduct ratio
analysis, percent change in risership.
;

title1
"Research Question: Does BART ridership significantly differ from normal weekday operations as compared to operations during an afternoon when the San Francisco Giants baseball team is having a downtown baseball game at ATT Park?"
;

title2
"This should help identify projections for economic opportunity for local vendors."
;

proc print data=work.match_merge_records label noobs;
    label
        prcnt_chng = "Percent Change of BART Ridership During Baseball Game vs Normal Weekday:"
    ;
run;

title;

proc print data=work.pop_avg_off (keep = no_game_ridership) label noobs;
    label
        no_game_ridership = "Normal BART Exits - Weekday"
    ;
run;

footnote1 bcolor=aquamarine bold underlin=1
"This study measures percent change in arrival ridership from the three San Francisco financial district BART stations (i.e. Montgomery, Embarcadero, and Powell stations), an hour before the opening pitch.  Game started at 3:15 PM, BART station arrival time for the entire 2 o'clock hour."
;

footnote2 bcolor=aquamarine bold underlin=1
"Game day: September 30, 2016.  Control day: March 31, 2016"
;

footnote3 bcolor=aquamarine bold underlin=1
"As one can see this percent change in relation to the actual number of station exits during these two different days only accounts for 242 extra passengers on game day, as compared to any other weekday when nothing special is happening in San Francisco downtown."
;

proc print data=work.pop_avg_game (keep = game_ridership) label noobs;
    label
        game_ridership = "Game Day BART Exits - Weekday"
    ;
run;

footnote;

*******************************************************************************;
* Research Question Analysis Starting Point                                    ;
*******************************************************************************;
*
Methodology: Here we will focus on both EMBR and West Oakland stations, as 
between these two stations is only the transbay tunnel.  Once calculating
ridership load between the two stations we can compare loading and capacity
with respect to morning and evening commute.
;

title1
"Research Question: How do ridership station exits compare (i.e. % change) for both Embarcadero and Montgomery Street stations with respect to morning and evening rush hours?"
;

title2 
"Rationale: As the Transbay tunnel is the single nexus within the BART system it is important to know travel loads to forecast general maintenance."
;

proc print data=work.m_rush (keep = exit tot_num) label noobs;
    label
        tot_num = "Total # of Morning Exits"
        exit = "Station"
    ;
run;

title;

proc print data=work.pm_rush (keep = exit tot_num) label noobs;
    label
        tot_num = "Total # of Evening Exits"
        exit = "Station"
    ;
run;

footnote1 bcolor=aquamarine bold underlin=1
"Morning rush hour: 7-9am. Evening rush hour: 4-6pm."
;

proc print data=work.compare_am_pm_rush (keep = exit percent_change) label noobs;
    label
        percent_change = "% Change - Ridership"
        exit           = "Station"
    ;
run;

footnote;

*******************************************************************************;
* Research Question Analysis Starting Point                                    ;
*******************************************************************************;
*
Methodology: Sorting and using proc means with variance option it will be
possible to compute the variance between these two stations.
;

title1
"Research Question: provide a statistical 5-number summary for AM rush hour commute station exits between San Francisco financial district and San Francisco’s civic center; does the distribution appear to be normal?"
;

title2 
"Rationale: Identifying the type of distribution when analyzing specific BART stations is the first step for any kind of further analysis, like determining passenger station exit probability."
;

footnote1 bcolor=aquamarine bold underlin=1
"We see that the distribution is skewed to the left, indicating disproportionate station exits during AM rush hour BART traffic.  More passengers exit in the financial district in the morning as compared to the civic center."
;

proc means data=work.civic_center_am_rush mean min q1 median q3 max;
   var hour num;
run;

title;
footnote;