import glob
import random

files = glob.glob('./data/train/*')
lists = []
for f in files:
    lists += glob.glob(f+'/*')
random.shuffle(lists)

with open('./data/train.csv', 'w') as f:
    for l in lists[:20000]:
        f.write('{},{}\n'.format(l, l.split('/')[3].split('c')[1]))

with open('./data/val.csv', 'w') as f:
    for l in lists[20000:]:
        f.write('{},{}\n'.format(l, l.split('/')[3].split('c')[1]))
