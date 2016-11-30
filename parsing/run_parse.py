#!/home/nyao111/anaconda2/bin/python
## PARSE ALL FILES
# Unzip + Parse all the .gz files on the specified folder
#

import parsing_tools as prs
import sys

def main():

	# Check if a path was specified
	if len(sys.argv) > 1:
		ZIP_FOLDER_PATH = sys.argv[1]
	else:
		ZIP_FOLDER_PATH = '../data/8k-gz/'

	BASE_FOLDER_PATH = '/'.join( list(filter(lambda x: x, ZIP_FOLDER_PATH.split('/')))[:-1] ) + '/'

	UNZIPPED_FOLDER = '{}unzipped/'.format(BASE_FOLDER_PATH)
	PARSED_FOLDER = '{}parsed/'.format(BASE_FOLDER_PATH)


	# First unzip all files
	prs.parse_files(ZIP_FOLDER_PATH, output_folder=UNZIPPED_FOLDER )

	# Second parse unzipped folder
	prs.parse_8kfolder(UNZIPPED_FOLDER, output_folder=PARSED_FOLDER)



if __name__ == '__main__':
	main()
