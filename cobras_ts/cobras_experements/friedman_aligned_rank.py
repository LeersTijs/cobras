"""
https://www.geeksforgeeks.org/friedman-test/

It is a non-parametric test alternative to hte one way ANOVA with repeated measures.

Friedman test: used to test for differences between groups
            when the dependent variable is ordinal.

            It is particularly useful when the sample size is very small.

Elements of the test:
    - One group that is measured on three or more blocks of measures overtime/experimental conditions
    - One dependent variable which can be ordinal, interval or ratio

Assumptions:
    - The group is a random sample from the population
    - Samples are not normally distributed

=> We can stop after step 6 (it gives an assign ranks for the columns.)

"""
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as st


def friedman_aligned_ranks_test(*args):
    """
        Performs a Friedman aligned ranks ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.
        The difference with a friedman test is that it uses the median of each group to construct the ranking, which is useful when the number of samples is low.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        Chi2-value : float
            The computed Chi2-value of the test.
        p-value : float
            The associated p-value from the Chi2-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
         J.L. Hodges, E.L. Lehmann, Ranks methods for combination of independent experiments in analysis of variance, Annals of Mathematical Statistics 33 (1962) 482–497.
    """
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    aligned_observations = []
    for i in range(n):
        loc = sp.mean([col[i] for col in args])
        aligned_observations.extend([col[i] - loc for col in args])

    aligned_observations_sort = sorted(aligned_observations)

    aligned_ranks = []
    for i in range(n):
        row = []
        for j in range(k):
            v = aligned_observations[i * k + j]
            row.append(aligned_observations_sort.index(v) + 1 + (aligned_observations_sort.count(v) - 1) / 2.)
        aligned_ranks.append(row)

    rankings_avg = [sp.mean([case[j] for case in aligned_ranks]) for j in range(k)]
    rankings_cmp = [r / sp.sqrt(k * (n * k + 1) / 6.) for r in rankings_avg]

    r_i = [np.sum(case) for case in aligned_ranks]
    r_j = [np.sum([case[j] for case in aligned_ranks]) for j in range(k)]
    T = (k - 1) * (sp.sum(v ** 2 for v in r_j) - (k * n ** 2 / 4.) * (k * n + 1) ** 2) / float(
        ((k * n * (k * n + 1) * (2 * k * n + 1)) / 6.) - (1. / float(k)) * sp.sum(v ** 2 for v in r_i))

    p_value = 1 - st.chi2.cdf(T, k - 1)

    return T, p_value, rankings_avg, rankings_cmp


def friedman_test(*args):
    """
        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2. for v in row])

    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (k + 1) / (6. * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
            (sp.sum(r ** 2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def main():
    path = "../../test_file.txt"
    table = pd.read_csv(path)
    # table.drop("dataset", inplace=True, axis=1)
    table.drop("super-instance", inplace=True, axis=1)
    print(table)
    l = -table.values
    iman, p, rankings_avg, rankings_cmp = friedman_test(*l)

    print("using every column")
    print(rankings_avg)
    print()
    print(rankings_cmp)

    print("----------")
    result = np.array([0, 0, 0], dtype=np.float64)
    for heur in ["size", "max-dist", "avg-dist", "med-dist", "var-dist"]:
        l = -table[heur].values
        l = list(map(lambda x: [x], l))
        print(l)
        iman, p, rankings_avg, rankings_cmp = friedman_test(*l)
        print(rankings_cmp)
        result += np.array(rankings_cmp)
    print("result:")
    print(result)


if __name__ == "__main__":
    # main()
    df = pd.DataFrame(columns=["size", "max dist", "avg dist", "med dist", "var dist"])
    # df.add([2, 3.4, 4.5, 3, 7], axis="rows")
    df.loc[len(df.index)] = [2, 3.4, 4.5, 3, 7]
    print(df)
