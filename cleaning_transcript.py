import json
import os
import codecs

transcript_file = os.path.join('utt_spk_text_original.tsv')


# Read in the file
with open(transcript_file, 'r', encoding='utf-8',
          errors='ignore') as file :
    file_data = file.read()

print(file_data)

remove_words =  list(map(chr, range(ord('a'), ord('z')+1))) + list(map(chr, range(ord('A'), ord('Z')+1))) + list(map(chr, range(ord('0'), ord('9')+1)))

# Replace the target string
file_data = file_data.replace('\x93', '')
file_data = file_data.replace('\x94', '')
file_data = file_data.replace('\u200c', '')
file_data = file_data.replace('\u200d', '')
file_data = file_data.replace('\u200e', '')
file_data = file_data.replace('œ', '')

for remove_word in remove_words:
    file_data = file_data.replace(remove_word, '')

# Write the file out again
# with open(transcript_file, 'w', encoding='utf-8',
#           errors='ignore') as file:
#     file.write(file_data)

