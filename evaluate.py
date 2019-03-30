import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from utils import read_embeddings, normalize_embeddings, read_lines


# Evaluate on Similarity/Relatedness dataset,
# human scores versus cosine similarity correlation test, Spearman's correlation score
def human_vs_cos_sim_correlation(human_score_path, embeddings, word2ids):
    human_scores = pd.read_csv(human_score_path)
    # lowercase string values
    human_scores = human_scores.applymap(lambda s: s.lower() if type(s) == str else s)
    # get rid of oov
    human_scores = human_scores[(human_scores.word1.isin(word2ids)) & (human_scores.word2.isin(word2ids))]
    # get embeddings for the first and second words in pairs
    first_word_ids = [word2ids[k] for k in human_scores.iloc[:, 0]]
    second_word_ids = [word2ids[k] for k in human_scores.iloc[:, 1]]
    first_word_embs = embeddings.take(first_word_ids, axis=0)
    second_word_embs = embeddings.take(second_word_ids, axis=0)
    # calculate cosine similarity
    h_product = np.multiply(first_word_embs, second_word_embs)
    cos_similarity = np.sum(h_product, axis=1)
    rho, p = spearmanr(np.column_stack((human_scores.iloc[:, 2], cos_similarity)))
    print("Spearman's rho, p-value:", '{:.2f}'.format(rho), '{:.2e}'.format(p))
    return rho, p


# Evaluate on Analogies dataset.
# Cosine similarity is used to find the closest word to d = (b - a) + c, where [a,b,c] is a question, and d is the
# answer. Calculate top1 and top10 accuracy (%) for each category and average accuracies over
# semantic/syntactic/all categories. Assume semantic categories go first and syntactic category names start with 'gram'
def answer_analogy_questions(analogies_path, embeddings, words2ids, top_k):
    all_questions_init = read_lines(analogies_path)
    # lowercase everything
    all_questions_low = [[j.lower() for j in i] for i in all_questions_init]
    # get rid of oov
    all_questions = [q for q in all_questions_low if q[0] == ":" or
                     (q[0] in words2ids and q[1] in words2ids and q[2] in words2ids and q[3] in words2ids)]
    results = []
    group = []
    print('group_name', '1nn%', '10nn%')
    # answer questions, combining them in groups
    for line in all_questions:
        if line[0] == ':':
            if group:  # if group is not empty, evaluate and print results
                results[-1].extend(
                    answer_questions_in_group(group, embeddings, words2ids, top_k))
                print(results[-1][0], '%.2f' % results[-1][1], '%.2f' % results[-1][2])
                group = []
            group_name = line[1]
            results.append([group_name])
        else:
            group.append(line)
    # handle last group's results
    results[-1].extend(answer_questions_in_group(group, embeddings, words2ids, top_k))
    print(results[-1][0], '%.2f' % results[-1][1], '%.2f' % results[-1][2])
    # print overall results
    n_syntactic = sum(1 for r in results if r[0].startswith('gram'))
    summarize_analogies_results(results, n_syntactic)


# Answer analogy questions in one group
def answer_questions_in_group(questions, embeddings, words2ids, top_k):
    targets = np.ndarray(shape=(len(questions), embeddings.shape[1]), dtype=np.float32)
    for i, q in enumerate(questions):  # [a,b,c]. d = (b - a) + c
        # target embeddings - closest points to question answers
        targets[i, :] = (embeddings[words2ids[q[1]], :] - embeddings[words2ids[q[0]], :]) \
                        + embeddings[words2ids[q[2]], :]
    distances = np.dot(targets, embeddings.T)
    # number of nearest neighbors we are interested in, +3 to account for question words, which we will ignore then
    num_best = top_k + 3
    # partition instead of sorting as it is way faster
    partitioned = np.argpartition(-distances, num_best, axis=1)[:, : num_best]
    # number of correct answers as a 1st nearest neighbor / in a 10-nearest neighbors range
    num_1nn = 0
    num_10nn = 0
    # answer each question - consider first top_k neighbors ignoring question words
    for i, q in enumerate(questions):
        # convert question words to ids
        q_ids = [words2ids[w] for w in q]
        # sort partition based on distances
        p_i = partitioned[i, :]
        d_i = distances[i, :]
        nearest = p_i[np.argsort(-d_i[p_i])]
        # filter out question words and crop up to top_k
        nearest_filtered = [w for w in nearest if w not in q_ids[:3]][:top_k]

        # check for true answer
        if q_ids[3] in nearest_filtered:
            num_10nn += 1
            if q_ids[3] == nearest_filtered[0]:
                num_1nn += 1

    n_quest = len(questions)
    percent_1nn = num_1nn * 100 / n_quest
    percent_10nn = num_10nn * 100 / n_quest
    return percent_1nn, percent_10nn


# Summarize and print results of Analogies evaluation
def summarize_analogies_results(results, len_syn):
    len_sem = len(results) - len_syn
    avg_1 = sum([r[1] for r in results]) / len(results)
    avg_10 = sum([r[2] for r in results]) / len(results)
    avg_sem_1 = sum([r[1] for r in results[:len_sem]]) / len_sem
    avg_sem_10 = sum([r[2] for r in results[:len_sem]]) / len_sem
    avg_syn_1 = sum([r[1] for r in results[len_sem:]]) / len_syn
    avg_syn_10 = sum([r[2] for r in results[len_sem:]]) / len_syn
    print("Semantic avg 1nn, 10nn accuracy:", '%.2f' % avg_sem_1, '%.2f' % avg_sem_10)
    print("Syntactic avg 1nn, 10nn accuracy:", '%.2f' % avg_syn_1, '%.2f' % avg_syn_10)
    print("Overall avg 1nn, 10nn accuracy:", '%.2f' % avg_1, '%.2f' % avg_10)


def main():
    if len(sys.argv) > 1:
        emb_path = sys.argv[1]
        if not os.path.exists(emb_path):
            print('Error. Embeddings file is not found')
            return
    else:
        print('Error. Specify path to embeddings file')
        return
    embeddings, words2ids = read_embeddings(emb_path)
    embeddings = normalize_embeddings(embeddings)
    print('SIMILARITY test:')
    human_vs_cos_sim_correlation('datasets/tt_similarity.csv', embeddings, words2ids)
    print('RELATEDNESS test:')
    human_vs_cos_sim_correlation('datasets/tt_relatedness.csv', embeddings, words2ids)
    print('ANALOGIES test:')
    top_k = 10
    answer_analogy_questions('datasets/tt_analogies.txt', embeddings, words2ids, top_k)


if __name__ == '__main__':
    main()
