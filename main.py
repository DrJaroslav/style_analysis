import analyzer_for_mypoeticside as afm
import os

links = ['https://mypoeticside.com/poets/william-shakespeare-poems',
         'https://mypoeticside.com/poets/william-blake-poems',
         'https://mypoeticside.com/poets/rudyard-kipling-poems']

train_files = [r'train_poems\shakespeare_poems.txt',
               r'train_poems\blake_poems.txt',
               r'train_poems\kipling_poems.txt']

test_files = [r'test_poems\shakespeare_poems.txt',
              r'test_poems\blake_poems.txt',
              r'test_poems\kipling_poems.txt']

names = ['Shakespeare', 'Blake', 'Kipling']

poem_file = "poem.txt"

#Pašalinti failus, jeigu jie jau buvo sukurti per ankstesnį programos paleidimą

for train_file, test_file in zip(train_files, test_files):
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)


#Ištraukti kiekvieno autoriaus vienodą nuorodų skaičių

minimum = afm.count_minimum(links)

for link, train_file, test_file in zip(links, train_files, test_files):
    afm.extract_poems(link, train_file, test_file, minimum, 5)
    print(f'{train_file}: completed.')
    print(f'{test_file}: completed.')


#Apskaičiuoti kiek žodžių yra tekstyne

total_counter = 0

for train_file in train_files:
    train_counter = afm.count_words(train_file)
    print(f'{train_file}: {train_counter:,} words.')

    total_counter += train_counter

print(f'Total: {total_counter:,} words.')

total_counter = 0

for test_file in test_files:
    test_counter = afm.count_words(test_file)
    print(f'{test_file}: {test_counter:,} words.')

    total_counter += test_counter

print(f'Total: {total_counter:,} words.')

afm.predict_author(train_files, names, poem_file)
