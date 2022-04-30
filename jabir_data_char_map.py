chars = [
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
hasan_char = [
    "_",
    "ঀ",
    "ঁঁ",
    "ং",
    "ঃ",
    "অ",
    "আ",
    "ই",
    "ঈ",
    "উ",
    "ঊ",
    "ঋ",
    "ঌ",
    "এ",
    "ঐ",
    "ও",
    "ঔ",
    "ক",
    "খ",
    "গ",
    "ঘ",
    "ঙ",
    "চ",
    "ছ",
    "জ",
    "ঝ",
    "ঞ",
    "ট",
    "ঠ",
    "ড",
    "ঢ",
    "ণ",
    "ত",
    "থ",
    "দ",
    "ধ",
    "ন",
    "প",
    "ফ",
    "ব",
    "ভ",
    "ম",
    "য",
    "র",
    "ল",
    "শ",
    "ষ",
    "স",
    "হ",
    "়",
    "ঽ",
    "া",
    "ি",
    "ী",
    "ু",
    "ূ",
    "ৃ",
    "ৄ",
    "ে",
    "ৈ",
    "ো",
    "ৌ",
    "্",
    "ৎ",
    "ৗ",
    "ড়",
    "ঢ়",
    "য়",
    "ৠ",
    "০",
    "১",
    "২",
    "৩",
    "৪",
    "৫",
    "৬",
    "৭",
    "৮",
    "৯",
    "ৱ",
    "৲",
    "৴"
]
chars.extend(hasan_char)

chars = list(set(chars))

print(len(chars))
print(chars)

t = ['অ', ',', 'u', '—', '়', 'j', 'ঙ', '0', 'গ', 'D', 'o', 'ট', 'ঈ', 'ড', 'ৎ', 'এ', '5', '!', 'ৠ', 'ঞ', 'r', ';', '৩',
     'ঀ', 'ন', 'জ', 'k', 'দ', '৲', 'l', 'F', '৭', 'n', 'f', 'প', 'x', 'ঢ', '২', 'B', 'ঁ', 'b', 'ম', 'T', 'z', '৫', 'ষ',
     'ড়', 'L', '_', 'ু', '’', '৪', 'ধ', 'চ', 't', 'ভ', 'ছ', 'h', 'ফ', 'ঋ', '1', 'c', 'আ', 'H', 'O', '-', '4', "'", 'ঃ',
     '৴', 'ঽ', 'ঠ', '‘', '.', 'য', 'm', 'ং', 'ৈ', '৯', '।', 'ঁঁ', 'd', 'ৌ', 'ো', '“', 'হ', '”', 'e', 'œ', 'ী', '…', 'w',
     'ঘ', '্', 'y', 'ি', ':', 'ঊ', 'ৰ', '9', 'া', 'p', '8', '2', '3', 'স', 'থ', 'ৄ', 'ঐ', '"', 'ৗ', '/', 'ত', 'ঝ', '১',
     '০', 'ঌ', 'র', 'a', '?', 'ও', 'শ', 'ই', 'v', 'g', 'ূ', 'K', 's', '%', 'ণ', 'ঔ', 'i', 'খ', 'ব', '–', 'ৱ', 'ঢ়', '৬',
     '৮', 'য়', 'ৃ', 'ল', 'ক', 'উ', 'ে']
