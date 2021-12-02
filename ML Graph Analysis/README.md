**CFG ML analysis for malware detection/classificiation**  

Currently, the program supports generating many CFGs from malware and benign files (do not exist on this repository)  

`angrg.py` - runs graph analysis and pulls files from hardcoded malware and benign directories. Contains most of the code for this project  

> `python3 -o angrg.py`  

`run_experiment.py` - iterates through several malware repositories to collect data. This is usually just for much longer experiments to run overnight  

> `python3 run_experiment.py`  

`file_reports` contains logging information about files that passed/failed. This seems to be a little broken right now.  
