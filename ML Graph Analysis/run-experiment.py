
import angrg as ang
import time

# This file is if you want to run a larger, longer machine learning experiment. It will collect more data points for malware directories

def main(): 

	start_time = time.time()

	# Ideally you want to iterate through several different malware directories for each experiment, 
	#  but the computer freezes up eventually, so right now I am doing 1 at a time
	directories = [ "8ae" ] #, "8a3", "8a4", "8a5", "8a6", "8a7", "8a8","8a9","8aa","8ab","8ac","8ad","8ae","8af","8ae"]


	for each_directory in directories:
		# Runs main function 
		ang.graph_analysis(each_directory)
		
	print("--- Runtime of program is %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
	main()
