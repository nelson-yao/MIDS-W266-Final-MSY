##  DATA PARSER
#  Parses the raw data downloaded from 
#     Oct, 2016


import gzip
import glob
import re
import pandas as pd



## Variable-parsing functions

META_LINES = ['FILE', 'TIME', 'EVENTS', 'ITEM']

# regex templates
reg_FILE = re.compile('FILE:(.*)')
reg_TIME = re.compile('TIME:(.*)')
reg_EVENTS = re.compile('EVENTS:(.*)')
reg_ITEMS = re.compile('ITEM:(.*)')

# A parsing_function generator
def factory_parser(compiled_regex):
	return lambda d: map(lambda x: x.replace(',', '').strip(), re.findall(compiled_regex, d))

get_file = factory_parser(reg_FILE)
get_time = factory_parser(reg_TIME)
get_events = factory_parser(reg_EVENTS)
get_items = factory_parser(reg_ITEMS)

def get_text(doc):
	lines = [ ln for ln in doc.split('\n') if ln != ' ' ]
	g_lines = [ ln for ln in lines if not any(map(lambda x: ln.startswith(x), META_LINES)) ]
	return '\n'.join( g_lines )



## Full file parser function

def parse_8kfile(flatfile, output_folder='../data/parsed/8k-data/'):
	""" Parses a file and outputs it as a csv """

	filename = flatfile.split('/')[-1].split('.')[0]  # extract the filename for output
	
	# Read flat file
	with open(flatfile, 'r') as f:
		data = f.read()

	# Split into documents
	documents = data.replace('<DOCUMENT>', '').split('</DOCUMENT>')

	# Parse documents and create dataframe
	parsed_data = {
		'file': map(get_file, documents),
		'time': map(get_time, documents),
		'events': map(get_events, documents),
		'items': map(get_items, documents),
		'text': map(get_text, documents)
	 }

	struct_data = pd.DataFrame.from_dict(parsed_data)

	# save to csv
	struct_data.to_csv( '{f}/{nm}.csv'.format(f=output_folder, nm=filename) )


## Folder parser

def parse_8kfolder(flatfile_folder, output_folder='../data/parsed/8k-data/'):
	""" Parses the entire folder of flat files """

	files = glob.glob( '{f}/*'.format(f=flatfile_folder) )

	print( '\n\033[34mParsing {} files ...\033[0m'.format(len(files)) )

	map( parse_8kfile, files )

	print( '\033[34mDone! Located in:  \033[32m{}\033[0m'.format(output_folder) )



## Unzip a gzip file

def unzip_files(folder, output_folder='data/unzipped/'):
	""" Unzips all .gz files found in the specified folder """

	files = glob.glob( '{f}/*.gz'.format(f=folder) )

	print( '\n\033[34mUnzipping {} files ...\033[0m'.format(len(files)) )

	for gzip_file in files:

		with gzip.open(gzip_file, 'rb') as f_in:
			_data = f_in.read()

		# replace the '.gz' with '.txt'
		output_file = gzip_file.replace('.gz', '.txt').split('/')[-1]

		with open('{f}{out}'.format(f=output_folder, out=output_file), 'wb') as f_out:
			f_out.write(_data)

	print( '\033[34mDone! Located in:  \033[32m{}\033[0m'.format(output_folder) )


