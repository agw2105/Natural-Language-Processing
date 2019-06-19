#replace multiple substrings in string or series of strings
import re

def replace(string, substitutions):
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)

#create dictionary of substrings and substitutions for the "substitutions" argument
substitution = {'substring to replace': 'replacement string'}

text_clean = []
for i in text: #text is placeholder for text data to clean
    text_clean.append(replace(i, substitution))