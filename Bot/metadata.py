from os import listdir
from os.path import isfile, join

path = '/Users/MacBook/Documents/LSTM Data/'

negative_files = [f'{path}/Negative/' + f for f in listdir(f'{path}/Negative/')
                  if isfile(join(f'{path}/Negative/', f))]
non_negative_files = [f'{path}/Non-negative/' + f for f in listdir(f'{path}/Non-negative/')
                      if isfile(join(f'{path}/Non-negative/', f))]

numWords = []
for n in negative_files:
    with open(n, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Negative files finished')

for nn in non_negative_files:
    with open(nn, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Non-negative files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords) / len(numWords))

'''
The total number of files is 313217
The total number of words in the files is 22134493
The average number of words in the files is 70.66823639840749
'''