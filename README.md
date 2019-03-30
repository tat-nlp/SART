# SART - Similarity, Analogies, and Relatedness for Tatar language: datasets for semantic/syntactic evaluation


This repository holds three evaluation datasets for the Tatar language:

- <a href="https://github.com/tat-nlp/SART/blob/master/datasets/tt_similarity.csv">Similarity dataset</a> - 202 pairs of words along with averaged human scores of *similarity* degree between the words (in 0-to-10 scale). For example, "<span style="font-family: Courier New;">йорт, бина, 7.69</span>".
- <a href="https://github.com/tat-nlp/SART/blob/master/datasets/tt_relatedness.csv">Relatedness dataset</a> - 252 pairs of words along with averaged human scores of *relatedness* degree between the words. For example, "<span style="font-family: Courier New;">урам, балалар, 5.38</span>".
- <a href="https://github.com/tat-nlp/SART/blob/master/datasets/tt_analogies.txt">Analogies dataset</a> - set of analytical questions of the form A:B::C:D, meaning A to B as C to D, and D is to be predicted. For example, "<span style="font-family: Courier New;">Әнкара Төркия Париж Франция</span>". Contains 34 categories, and in total 30 144 questions. 

Similarity and Relatedness datasets were built based on [WordSim353](http://alfonseca.org/eng/research/wordsim353.html) dataset, and Analogies dataset is an extension of several existing datasets, including [Google analogies test set](https://aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art)).

## Purpose  

These datasets can be used for semantic and syntactic evaluation of lingustic models. One of the popular usages is word embeddings evaluation, and we provide a script for doing that.


## Format

All datasets reside in `datasets` folder. Similarity and Relatedness come in a `.csv` format, and Analogies as a `.txt` file. Analogies dataset containes questions grouped in categories; whenever a new category begins, there is a line with a category name, beginning with ":" symbol, for example, "<span style="font-family: Courier New;">: capital-country</span>".


## Evaluation

In addition to the datasets we are providing the code for evaluating word embeddings. To run it, you should have Python3, and [numpy](http://www.numpy.org/), [pandas](https://pandas.pydata.org/), [scipy](https://www.scipy.org/) libraries installed.

If you want to run it on embeddings from our paper, download them from the links below:

- Skip-gram [Click to download](https://yadi.sk/i/DTyzMfYD6EKmFQ)
- FastText [Click to download](https://yadi.sk/i/kApWPiw8kuMimA)
- GloVe [Click to download](https://yadi.sk/i/SRfNf0KHrIFrCg)

Or, if you use your own embeddings, provide them in the standard word2vec format, i.e. a`.txt` file with dimensions in the first row.

Run `evaluate.py` script, providing path to embeddings file as an argument:

### Example

```bash
python evaluate.py tt_skipgram_emb.txt
```

## Annotation Instructions

Scores for Similarity and Relatedness datasets were obtained by  averaging scores of 13 annotators, all native Tatar speakers. You can find annotation instructions (in Russian) we showed them in `annotation_instructions` folder.


## Citing

Find more details about datasets construction and evaluations in the dedicated paper:

**SART - Similarity, Analogies, and Relatedness for Tatar language: New Benchmark Datasets for Word Embeddings Evaluation**


If you use these datasets, please cite as follows:

```
@article{khusainova2019sart,
  title={Similarity, Analogies, and Relatedness for Tatar language: New Benchmark Datasets for Word Embeddings Evaluation},
  author={Albina Khusainova, Adil Khan, and Adín Ramírez Rivera},
  journal={arXiv preprint arXiv:nnnn.nnnnn},
  year={2019}
}
```

## License

The provided datasets are licensed under the Creative Commons Attribution 4.0 International. A full copy of the license can be found in `LICENSE.txt`.
