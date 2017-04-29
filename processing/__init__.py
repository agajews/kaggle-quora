from .augment import noisify, transitivify
from .clean import fancy_tokenize, lower, replace_nums, stem, strip_punct
from .core import load_data

all_processors = {
    'lower': lower,
    'replace_nums': replace_nums,
    'fancy_tokenize': fancy_tokenize,
    'stem': stem,
    'strip_punct': strip_punct
}

processor_order = [
    'lower',
    'strip_punct',
    'fancy_tokenize',
    'replace_nums',
    'stem'
]

all_augmentors = {
    'transitivify': transitivify,
    'noisify': noisify
}

augmentor_order = [
    'transitivify',
    'noisify'
]
