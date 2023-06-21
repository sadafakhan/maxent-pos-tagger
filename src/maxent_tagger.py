"""Sadaf Khan, LING570, HW10, 12/06/2021. Creates feature vectors for the training and test data. Runs mallet thereafter
to create a MaxEnt model using said feature vectors."""

import os
from collections import defaultdict
import sys

# have the format w1/t1 w2/t2 ... wn/tn
train_file = sys.argv[1]
test_file = sys.argv[2]

# any words in train_file and test_file that appear LESS THAN rare_thres in train_file are rare words
rare_thres = int(sys.argv[3])

# all the current word (w_i) features regardless of frequency should be kept
# for all OTHER types of features, if it appears LESS THAN feat_thres, it should be removed from feature vectors
feat_thres = int(sys.argv[4])

# stores the output files from the tagger
output_dir = sys.argv[5]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
training_sentences = []

# creating train_voc, the vocabulary that includes all the words appearing in train file
training_data = open(os.path.join(os.path.dirname(__file__), train_file), 'r').read().split("\n")[:-1]
train_voc_dict = defaultdict(int)
for line in training_data:
    tokenized = line.split()
    tokenized.insert(0, 'BOS/BOS')
    tokenized.insert(0, 'BOS/BOS')
    tokenized.append('EOS/EOS')
    tokenized.append('EOS/EOS')
    sentence = []
    for tagged in tokenized:
        pair = tagged.split("/")
        word = pair[0]
        tag = pair[1]
        # Mallet needs commas to be spelled out as such
        if word == ",":
            word = "comma"
            tag = "comma"
        sentence.append((word, tag))
        train_voc_dict[word] += 1
    training_sentences.append(sentence)

train_voc_sorted = {k: v for k, v in sorted(train_voc_dict.items(), key=lambda x: x[1], reverse=True)}
train_voc_path = os.path.join(output_dir, 'train_voc')
with open(train_voc_path, 'w') as g:
    for entry in train_voc_sorted:
        g.write(entry + "\t" + str(train_voc_sorted[entry]) + "\n")

# use the word frequency in train_voc and rare_thres to determine whether a word should be treated as a rare word
rare_voc = {}
for word in train_voc_dict:
    if train_voc_dict[word] < rare_thres:
        rare_voc[word] = train_voc_dict[word]

# form feature vectors for the words in train_file
training_vectors = []
for sentence in training_sentences:
    sentence_vectors = []
    for i in range(2, len(sentence)-2):
        word_vector = {}

        curWord = sentence[i][0]
        word_vector["curWord"] = curWord

        T = sentence[i][1]
        word_vector["T"] = T

        prevT = sentence[i-1][1]
        word_vector["prevT"] = prevT

        prevTwoTags = sentence[i-2][1] + "+" + sentence[i-1][1]
        word_vector["prevTwoTags"] = prevTwoTags

        prevW = sentence[i-1][0]
        word_vector["prevW"] = prevW

        prev2W = sentence[i-2][0]
        word_vector["prev2W"] = prev2W

        nextW = sentence[i+1][0]
        word_vector["nextW"] = nextW

        next2W = sentence[i+2][0]
        word_vector["next2W"] = next2W

        # handle rare words
        if curWord in rare_voc:
            # check for containing features
            for letter in curWord:
                if letter.isdigit():
                    word_vector["containNum"] = '1'
                if letter.isupper():
                    word_vector["containUC"] = '1'
                if letter == "-":
                    word_vector["containHyp"] = '1'
            if "containNum" not in word_vector:
                word_vector["containNum"] = '0'
            if "containUC" not in word_vector:
                word_vector["containUC"] = '0'
            if "containHyp" not in word_vector:
                word_vector["containHyp"] = '0'

            # prefix and suffix
            prefixes = []
            suffixes = []
            if len(curWord) > 4:
                b = 1
                e = 5
            else:
                b = 0
                e = len(curWord)
            for a in range(b, e):
                if len(curWord) > 4:
                    pref = curWord[0:a]
                else:
                    pref = curWord[0:a+1]
                suf = curWord[-a:]
                prefixes.append(pref)
                suffixes.append(suf)

            for k in range(len(prefixes)):
                pref_key = "pref" + str(k+1)
                suf_key = "suf" + str(k+1)
                word_vector[pref_key] = prefixes[k]
                word_vector[suf_key] = suffixes[k]
        sentence_vectors.append(word_vector)
    training_vectors.append(sentence_vectors)

flatten_prefix = {"pref1", "pref2", "pref3", "pref4"}
flatten_suffix = {"suf1", "suf2", "suf3", "suf4"}
init_feats = defaultdict(int)

# store the features & frequencies in the training data in init_feats
for sentence_vector in training_vectors:
    for word_vector in sentence_vector:
        for feat in word_vector:
            if feat == 'T':
                continue
            if feat in flatten_prefix:
                title = "pref"
            elif feat in flatten_suffix:
                title = "suf"
            else:
                title = feat
            entry = title + "=" + word_vector[feat]
            init_feats[entry] += 1

init_feats_sorted = {k: v for k, v in sorted(init_feats.items(), key=lambda x: x[1], reverse=True)}
init_feats_path = os.path.join(output_dir, 'init_feats')
with open(init_feats_path, 'w') as g:
    for entry in init_feats_sorted:
        g.write(entry + " " + str(init_feats_sorted[entry]) + "\n")

# Create kept_feats by using feat_thres to filter out low frequency features in init feats.
remove_feats = defaultdict(int)
keeping_feats = defaultdict(int)
kept_feats_path = os.path.join(output_dir, 'kept_feats')
with open(kept_feats_path, 'w') as g:
    for entry in init_feats_sorted:
        # wi features are NOT subject to filtering with feat_thres
        pair = entry.split("=")
        if pair[0] == "curWord" or "T":
            g.write(entry + "\t" + str(init_feats_sorted[entry]) + "\n")
        else:
            if init_feats_sorted[entry] >= feat_thres:
                g.write(entry + "\t" + str(init_feats_sorted[entry]) + "\n")
                keeping_feats[entry]
            else:
                remove_feats[entry]


# Go through the feature vectors for train_file and remove all the features that are not in kept_feats
for sentence_vector in training_vectors:
    for word_vector in sentence_vector:
        for feature in (list(word_vector.keys())):
            if feature in flatten_prefix:
                title = "pref"
            elif feature in flatten_suffix:
                title = "suf"
            else:
                title = feature
            removable = title + "=" + word_vector[feature]
            if removable in remove_feats:
                del word_vector[feature]

try_features = ["prevT", "prevTwoTags", "prevW", "prev2W",  "nextW", "next2W",
                "containNum", "containHyp", "containUC",
                "pref1", "pref2", "pref3", "pref4",
                "suf1", "suf2", "suf3", "suf4"]

containing_features = ["containNum", "containHyp", "containUC"]

# Create feature vectors file for train_file
final_train_vectors_path = os.path.join(output_dir, 'final_train.vectors.txt')
with open(final_train_vectors_path, 'w') as g:
    for i in range(0, len(training_vectors)):
        for k in range(0, len(training_vectors[i])):
            word_vector = training_vectors[i][k]
            number = str(i+1) + "-" + str(k) + "-"
            g.write(number)
            g.write(word_vector['curWord'] + " ")
            g.write(word_vector['T'] + " ")
            g.write("curW" + "=" + word_vector['curWord'] + " 1 ")
            for feature in try_features:
                if feature in word_vector:
                    if feature in flatten_prefix:
                        title = "pref"
                    elif feature in flatten_suffix:
                        title = "suf"
                    elif feature in containing_features:
                        if int(word_vector[feature]) == 0:
                            continue
                        else:
                            title = feature
                    else:
                        title = feature
                    if feature in containing_features:
                        g.write(title + " 1 ")
                    else:
                        g.write(title + "=" + word_vector[feature] + " 1 ")
            g.write("\n")


# Create feature vectors for test_file, and use only the features in kept feats.
testing_sentences = []
final_test_vectors_path = os.path.join(output_dir, 'final_test.vectors.txt')
test_data = open(os.path.join(os.path.dirname(__file__), test_file), 'r').read().split("\n")[:-1]

# format the data as before
for line in test_data:
    tokenized = line.split()
    tokenized.insert(0, 'BOS/BOS')
    tokenized.insert(0, 'BOS/BOS')
    tokenized.append('EOS/EOS')
    tokenized.append('EOS/EOS')
    sentence = []
    for tagged in tokenized:
        pair = tagged.split("/")
        word = pair[0]
        tag = pair[1]
        # Mallet needs commas to be spelled out as such
        if word == ",":
            word = "comma"
            tag = "comma"
        sentence.append((word, tag))
    testing_sentences.append(sentence)


# form feature vectors for the words in test_file
testing_vectors = []
for sentence in testing_sentences:
    sentence_vectors = []
    for i in range(2, len(sentence)-2):
        word_vector = {}

        curWord = sentence[i][0]
        word_vector["curWord"] = curWord

        T = sentence[i][1]
        word_vector["T"] = T

        prevT = sentence[i-1][1]
        word_vector["prevT"] = prevT

        prevTwoTags = sentence[i-2][1] + "+" + sentence[i-1][1]
        word_vector["prevTwoTags"] = prevTwoTags

        prevW = sentence[i-1][0]
        word_vector["prevW"] = prevW

        prev2W = sentence[i-2][0]
        word_vector["prev2W"] = prev2W

        nextW = sentence[i+1][0]
        word_vector["nextW"] = nextW

        next2W = sentence[i+2][0]
        word_vector["next2W"] = next2W

        # handle rare words based on training vector rare words
        if curWord in rare_voc:
            # check for containing features
            for letter in curWord:
                if letter.isdigit():
                    word_vector["containNum"] = '1'
                if letter.isupper():
                    word_vector["containUC"] = '1'
                if letter == "-":
                    word_vector["containHyp"] = '1'
            if "containNum" not in word_vector:
                word_vector["containNum"] = '0'
            if "containUC" not in word_vector:
                word_vector["containUC"] = '0'
            if "containHyp" not in word_vector:
                word_vector["containHyp"] = '0'

            # prefix and suffix
            prefixes = []
            suffixes = []
            if len(curWord) > 4:
                b = 1
                e = 5
            else:
                b = 0
                e = len(curWord)
            for a in range(b, e):
                if len(curWord) > 4:
                    pref = curWord[0:a]
                else:
                    pref = curWord[0:a+1]
                suf = curWord[-a:]
                prefixes.append(pref)
                suffixes.append(suf)

            for k in range(len(prefixes)):
                pref_key = "pref" + str(k+1)
                suf_key = "suf" + str(k+1)
                word_vector[pref_key] = prefixes[k]
                word_vector[suf_key] = suffixes[k]
        sentence_vectors.append(word_vector)
    testing_vectors.append(sentence_vectors)

# use only the features in kept feats
for sentence_vector in testing_vectors:
    for word_vector in sentence_vector:
        for feature in (list(word_vector.keys())):
            if feature in flatten_prefix:
                title = "pref"
            elif feature in flatten_suffix:
                title = "suf"
            else:
                title = feature
            removable = title + "=" + word_vector[feature]
            if title == "curWord" or "T":
                continue
            else:
                if removable not in keeping_feats:
                    del word_vector[feature]


with open(final_test_vectors_path, 'w') as g:
    for i in range(0, len(testing_vectors)):
        for k in range(0, len(testing_vectors[i])):
            word_vector = testing_vectors[i][k]
            number = str(i+1) + "-" + str(k) + "-"
            g.write(number)
            g.write(word_vector['curWord'] + " ")
            g.write(word_vector['T'] + " ")
            g.write("curW" + "=" + word_vector['curWord'] + " 1 ")
            for feature in try_features:
                if feature in word_vector:
                    if feature in flatten_prefix:
                        title = "pref"
                    elif feature in flatten_suffix:
                        title = "suf"
                    elif feature in containing_features:
                        if int(word_vector[feature]) == 0:
                            continue
                        else:
                            title = feature
                    else:
                        title = feature
                    if feature in containing_features:
                        g.write(title + " 1 ")
                    else:
                        g.write(title + "=" + word_vector[feature] + " 1 ")
            g.write("\n")