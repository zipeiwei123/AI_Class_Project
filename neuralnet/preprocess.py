###
# preprocess.py
# Takes in text string and splits it into words, whitespace, and punctuation
#
# Author: Samantha Kerkhoff, samantha_kerkhoff@student.uml.edu
###

import re 

parse_whitespace = re.compile(r'(\s)')
parse_punctuation = re.compile(r'(\W)')

def tokenize(input_str):
    text_array = []
    for word in parse_whitespace.split(input_str):
        if '@' in word:
            continue
        if 'http' in word:
            continue
            
        if len(parse_whitespace.findall(word)) > 0:
            text_array.append(word)
        elif len(parse_punctuation.findall(word)) > 0:
            for item in parse_punctuation.split(word):
                text_array.append(item)
        else:
            text_array.append(word)
    return text_array
    
