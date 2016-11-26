## PARSE ALL FILES
# Unzip + Parse all the .gz files on the specified folder
#
# Usage: python run_parse.py path/to/inputfolder/ path/to/outputfolder/
#

import parsing_tools as prs
import sys
import os

def main():

	# Check if the input/output paths were specified
	if len(sys.argv) < 3:
		print('\n\033[31mERROR:\nPlease specify a input folder and an output folder...\033[0m\n')
		print('\n\033[35m\nExample of usage: python run_parse.py path/to/inputfolder/ path/to/outputfolder/\033[0m\n')
		sys.exit()

	input_folder = sys.argv[1]
	output_folder = sys.argv[2]

	# Check if the given folders exist
	if not os.path.isdir(input_folder):
		print( '\n\033[31mERROR:\nInput folder {} not found\033[0m\n'.format(input_folder) )
		sys.exit()
	if not os.path.isdir(output_folder):
		print( '\n\033[31mERROR:\Output folder {} not found\033[0m\n'.format(output_folder) )
		sys.exit()


	prs.parse_files(folder=input_folder, output_folder=output_folder)



if __name__ == '__main__':
	main()
