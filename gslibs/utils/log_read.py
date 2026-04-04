import os
from typing import List

def read_array(stem, prefix, filespath: str = 'out.txt'):
    """
    Reads all folders in the given stem directory and with prefix 'prefix'
    and returns a iterable of paths the filpaths files.
    """
    if not os.path.exists(stem):
        raise FileNotFoundError(f"Stem directory {stem} does not exist!")
    
    paths = []
    for run in [f for f in os.listdir(stem) if os.path.isdir(os.path.join(stem, f))]:
        if run.startswith(prefix) and os.path.exists(os.path.join(stem, run, filespath)):
            paths.append(os.path.join(stem, run, filespath))

    return paths

def read_sections(filename, section_string, delimiter_char='='):
    """
    Reads sections from a file that after 'section_string' appears within
    lines with 10 or more delimiter characters. The section ends when a line
    with the same number of delimiter characters is found.
    """
    assert os.path.exists(filename), f"File {filename} does not exist!"
    
    sections = []

    insection = False
    inheader = False
    header = None
    with open(filename, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if delimiter_char * 10 in line:
            if inheader:
                inheader = False
                if header == None:
                    continue
                else:
                    start_line = i + 1
                    insection = True
            else:
                if insection:
                    sections.append(lines[start_line:i])
                    insection = False
                    inheader = False
                    header = None
                else:
                    inheader = True
        elif inheader:
            if section_string in line:
                header = line.strip()
            else:
                header = None
                inheader = False
                insection = False
                  
    return sections

def read_correlations(lines: List[str]):
    """
    Reads correlations from a list of lines, looking for the section
    "Correlation of latent space and labels" and returns a DataFrame.
    """
    import pandas as pd

    header = lines[0].strip()
    corrs = []
    columns = []
    for line in lines[1:]:
        line = line.strip().split()
        corrs.append([float(x) for x in line if '.' in x and x.replace('.', '', 1).replace('-','',1).isdigit()])
        columns.append(' '.join([x for x in line if not ('.' in x and x.replace('.', '', 1).replace('-','',1).isdigit())]))
    
    assert len(columns) == len(corrs), "Mismatch between number of columns and correlations."
    assert all([a in header for a in columns]), f"All columns must be present in the header for lines:\n{''.join(lines)}"

    df = pd.DataFrame(corrs, columns=columns)
    df.index = columns
    
    return df

def read_hparam(lines: List[str], param_name: str):
    """
    Reads correlations from a list of lines, looking for the section
    "Correlation of latent space and labels" and returns a DataFrame.
    """
    
    for line in lines:
        if param_name in line:
            beta = float(line.split()[-1])
            return beta
    


    
# def read_one_correlations(lines, delimiter_char='='):

#     start_line = -1
#     end_line = -1
#     for i in range(len(lines)):
#         if lines[i].strip() == delimiter_char * len(lines[i].strip()):
#             if start_line == -1:
#                 start_line = i
#             else:
#                 end_line = i
#                 break

#     if start_line != -1 and end_line != -1:
#         data = ''.join(lines[start_line+1:end_line])
#         data = data.replace("Latent Dimension 0", "Latent-Dimension-0")
#         data = data.replace("Latent Dimension 1", "Latent-Dimension-1")
#         return pd.read_csv(StringIO(data), sep=r"\s+", engine='python', index_col = 0)
#     else:
#         raise Exception("Could not find correlations in file.")


