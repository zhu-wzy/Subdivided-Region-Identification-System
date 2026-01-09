from numpy import random


# 生成序列的概率转移矩阵  无限接近输入
def myf_get_pm(outputseq, alphabet):
    T = [c - 1 for c in outputseq]

    # create matrix of zeros
    M = [[0] * len(alphabet)] * len(alphabet)

    for (i, j) in zip(T, T[1:]):
        M[i][j] += 1

    # 查看频次矩阵
    print('\n频数')
    [print(row) for row in M]

    # now convert to probabilities:
    for row in M:
        n = sum(row)
        if n > 0:
            row[:] = [round(f / sum(row), 4) for f in row]

    return M


# 生成的序列
def myf_get_idx(alphabet, transitions):
    # Create probability matrix filled with zeroes
    # Matrix consists of nested libraries
    prob_matrix = {}

    for i in alphabet:
        prob_matrix[i] = {}

        for j in alphabet:
            prob_matrix[i][j] = 0

    # fill matrix with numbers based on transitions list
    T = [c - 1 for c in transitions]

    for (i, j) in zip(T, T[1:]):
        prob_matrix[alphabet[i]][alphabet[j]] += 1

    # 查看数据
    print('类间转移频数:')
    for i in prob_matrix.keys():
        print(list(prob_matrix[i].values()))

    # convert to probabilities
    for row in prob_matrix:

        total = sum([prob_matrix[row][column] for column in prob_matrix[row]])

        if total > 0:
            for column in prob_matrix[row]:
                prob_matrix[row][column] /= total

    # generate first random sequence letter
    outputseq = [random.choice(alphabet, None)]

    # generate rest of string based on probability matrix
    for i in range(10000):
        probabilities = [prob_matrix[outputseq[-1]][j] for j in alphabet]
        outputseq += [random.choice(alphabet, None, False, probabilities)]

    return outputseq


# ###############################################################################################
if __name__ == '__main__':
    alphabet = [1, 2, 3]
    transitions = [3,  1,  1,  2,  3,  1,  3,  2,  1,  3]

    outputseq = myf_get_idx(alphabet, transitions)
    print('\noutputseq:\n', outputseq)

    prob_matrix = myf_get_pm(outputseq, alphabet)
    print('\nprob_matrix:')

    [print(row) for row in prob_matrix]
