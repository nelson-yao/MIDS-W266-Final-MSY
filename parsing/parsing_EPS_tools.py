##  DATA PARSER
#  Parses the raw data downloaded from
#     Nov, 2016


import pandas as pd


def parse_EPS_file(filepath):
    """ Parses a given EPS html file (does not have to have a html extension) """
    with open(filepath, 'r') as f:
        raw = f.read()

    # Trim the page to restrict to only one table
    raw_1table = raw.split('U.S. Earnings')[-1]

    # Find the HTML table
    _start = raw_1table.index('<table')
    _stop = raw_1table.index('</table>') + 8
    html_table = raw_1table[_start : _stop]

    # Parse the HTML table
    try:
        data = pd.read_html(html_table)[0]
    except:
        return None

    # Clean the table of internal headers and spaces, remove unnecessary columns
    parsed_EPS = data.loc[~data.ix[:, 1].isin([pd.np.nan, 'Symbol']), data.columns[:5]]
    parsed_EPS.columns = ['Company', 'Symbol', 'Surprise', 'Reported_EPS', 'Consensus_EPS']

    # Add date based on the filename
    parsed_EPS['Date'] = pd.to_datetime(filepath.split('/')[-1].split('.')[0], format='%Y%m%d')

    return parsed_EPS
