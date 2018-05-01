import nltk
from nltk.corpus import wordnet
foreground_categories = {'human':('humans','human','faces','face','person','mask','abaya'), 'place':('building','house','city','place')}
import sys
import json
from itertools import product

THRESHOLD = 0.8

if __name__ == '__main__':
    if(len(sys.argv)<3):
        exit(1)
    classifier_out = sys.argv[1]
    output = sys.argv[2]

    finp = open(classifier_out, 'r')
    raw_data = json.loads(finp.read())
    data = []
    max_weight = 0.0
    for key in raw_data:
        for category in key.split(','):
            category = category.strip()
            data.append((category, float(raw_data[key])))
        if(float(raw_data[key]) > max_weight):
            max_weight = float(raw_data[key])
    best = [x[0] for x in data if x[1] == max_weight]
    print("Best tags are %s"%(best,))
    try:
        nltk.data.find('corpora/wordnet')
    except:
        nltk.download('wordnet')
    tag_synsets = []
    for tag in best:
        tag_synsets.extend(wordnet.synsets(tag))
    tag_synsets = set(tag_synsets)
    category_max = []
    for key in foreground_categories:
        category_synset = []
        for tag in foreground_categories[key]:
            category_synset.extend(wordnet.synsets(tag))
        category_synset = set(category_synset)
        similarity = [(wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(tag_synsets, category_synset)]
        max_item = max(similarity)
        max_list = [x for x in (similarity) if max_item[0] == x[0]]
        category_max.append((max_item[0], len(max_list), key))
    category_max.sort(reverse = True)
    print("Best match is with %s with confidence %s"%(category_max[0][2], category_max[0][0]))
    fout = open(output, "w")
    if(category_max[0][0] < THRESHOLD):
        print("But this is below threshold. Falling back to general GAN.")
        fout.write(json.dumps({'use':'fallback'}))
    else:
        print("Using this GAN.")
        fout.write(json.dumps({'use':category_max[0][2]}))
    fout.close()
    exit(0)
