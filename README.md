# maxent-pos-tagger
```maxent-pos-tagger``` implements a maximum entropy tagger. It creates feature vectors for the training and test data, and then runs mallet to create a MaxEnt model using said feature vectors. You must have mallet installed in order to run this program. 

Args: 
* ```train_file```: has the format (e.g., test.word pos): w1/t1 w2/t2 ... wn/tn
* ```test_file```: has the format (e.g., test.word pos): w1/t1 w2/t2 ... wn/tn
* ```rare_thres```: an integer representing the threshold of occurence for which a word is considered "rare". E.g., if a word occurs less than rare_thres times, it is considered "rare." 
* ```feat_thres```: an integer representing the minimum threshold of occurence for a feature to be retained. E.g., if a feature occurs less than feat_thres times, it is discarded from the feature vectors. 


Returns: 
* ```output_dir```: a directory that stores the output files from the tagger. The output files include all that are listed below. 
* ```train_voc```: the frequency-sorted vocabulary that includes all the words appearing in train_file. 
* ```init_feats```: the frequency-sorted features that occur in train_file. 
* ```kept_feats```: a subset of init_feats which includes only the features that are kept after applying feat_thres.  
* ```final train.vectors.txt```: the feat vectors for the train file in the Mallet text format. Only features in kept_feats should be kept in this file. 
* ```final test.vectors.txt```: the feat vectors for the test_file in the Mallet text format. 
* ```final train.vectors```: the binary format of the vectors in final train.vectors.txt.
* ```me_model```: the MaxEnt model (in binary format) which is produced by the MaxEnt trainer.
* ```me_model.stdout```: the standard out produced by the MaxEnt trainer. 
* ```me_model.stderr```: the standard error produced by the MaxEnt trainer. 
* ```sys_out```: the system output file when running the MaxEnt classifier with command such as: ```mallet classify-file --input final_test.vectors.txt --classifier me_model --output sys_out```

To run: 
```
src/maxent_tagger.sh input/train_file input/test_file rare_thres feat_thres output
```

The output folder contains the outputs when running wsj_sec0.word_pos as train_file and test.word_pos as test_file. The directory name includes the rare_thres and feat_thres respectively, in that order. 

HW10 OF LING570 (12/06/2021)