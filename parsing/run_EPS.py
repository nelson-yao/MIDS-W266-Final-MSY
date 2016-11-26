## PARSE ALL FILES
# Unzip + Parse all the .gz files on the specified folder
#
# Usage: python run_parse.py path/to/inputfolder/ path/to/outputfolder/
#

import pandas as pd
import parsing_EPS_tools as prs

from glob import glob

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


    all_files = glob('{}*.txt'.format(input_folder))

    FAILED = []
    EMPTY = []
    count = 0

    MAJOR = max(len(all_files) // 10, 1)
    MINOR = max(len(all_files) // 100, 1)

    for filename in all_files:

        # Get output path
        outpath = '{fld}{f}.csv'.format(fld=output_folder, f=filename.split('/')[-1].split('.')[0])

        try:
            data = prs.parse_EPS_file(filename)
        except:
            print('X', end='', flush=True)
            FAILED.append(filename)
        else:
            if data is None:
                EMPTY.append(filename)
                print('^', end='', flush=True)
            else:
                # Save the data
                with open(outpath, 'w') as f:
                    data.to_csv(f)

        # Update counter and print processing mark
        count += 1
        if count % MAJOR == 0:
            print('-', end='', flush=True)
            if count % MINOR == 0:
                print('|', end='', flush=True)

    print('\n\n... Done!')

    print('Processed {} files, of which:'.format(count))
    print('\033[31m  - Failed:', len(FAILED), '\033[0m')
    print('\033[33m  - Empty:', len(EMPTY), '\033[0m\n')



if __name__ == '__main__':
    main()
