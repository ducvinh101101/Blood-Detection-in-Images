from glob import glob
from progressbar import progressbar
from os import system

fs = glob('*/*')

for f in progressbar(fs):
    system('mogrify -geometry x320 "' + f + '"')
