import json
import os
import codecs

transcript_file = os.path.join('english_jo_fola_clean.tsv')

print(transcript_file)

transcriptions = open(transcript_file, encoding='utf8')

print(transcriptions)

with codecs.open(transcript_file, 'r', encoding='utf-8',
                 errors='ignore') as fdata:
    transcriptions = fdata.read().strip().split("\n")
    transcriptions = [t.split("\t")[-1].replace(" ", "") for t in transcriptions]
    transcriptions = ''.join(transcriptions)


chars = []
chars.extend(transcriptions)
chars = list(set(chars))
print(chars)
print(len(chars))


t = [
    "H",
    "ি",
    "0",
    "p",
    "F",
    ";",
    "৩",
    "৫",
    "ক",
    "i",
    "z",
    "ঢ",
    "1",
    "ো",
    "শ",
    ".",
    "b",
    "\"",
    "v",
    "ঢ়",
    "ঁ",
    ",",
    "g",
    ":",
    "র",
    "ধ",
    "9",
    "y",
    "_",
    "ও",
    "ঈ",
    "চ",
    "œ",
    "8",
    "ঊ",
    "d",
    "৭",
    "-",
    "n",
    "৮",
    "।",
    "ঔ",
    "ষ",
    "ঙ",
    "ৎ",
    "ৗ",
    "০",
    "ৰ",
    "ঋ",
    "?",
    "ল",
    "ে",
    "৪",
    "ই",
    "f",
    "দ",
    "৬",
    "x",
    "“",
    "ূ",
    "'",
    "ছ",
    "r",
    "২",
    "য",
    "ঘ",
    "ু",
    "ম",
    "উ",
    "…",
    "2",
    "ত",
    "/",
    "ী",
    "ঠ",
    "3",
    "s",
    "ব",
    "্",
    "ন",
    "প",
    "ড",
    "ভ",
    "১",
    "ট",
    "t",
    "—",
    "ৃ",
    "৯",
    "4",
    "–",
    "অ",
    "স",
    "e",
    "ঐ",
    "আ",
    "h",
    "k",
    "ৈ",
    "L",
    "O",
    "গ",
    "ণ",
    "খ",
    "a",
    "w",
    "m",
    "l",
    "জ",
    "ং",
    "ৌ",
    "ঝ",
    "u",
    "এ",
    "ঞ",
    "K",
    "o",
    "হ",
    "া",
    "’",
    "T",
    "‘",
    "থ",
    "য়",
    "j",
    "5",
    "ড়",
    "D",
    "B",
    "”",
    "ফ",
    "ঃ",
    "c",
    "%",
    "়",
    "!"
]

