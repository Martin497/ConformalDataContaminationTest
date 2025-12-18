#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:14:57 2025

This module contains functionality for computing p-values for the contamination
hypothesis H_0 : pi <= pi_th.
"""

import numpy as np
from math import floor
from scipy.stats import chi2, nhypergeom, irwinhall, binom


class ConformalContaminationTest(object):
    """
    Wrapper class for computing conformal contamination test p-values.
    """
    def __init__(self, type_="storey"):
        """
        Inputs:
        -------
            type_ : str
                Options are "storey", "quantile", "fisher", "linear.
        """
        super(ConformalContaminationTest, self).__init__()
        type_options = ["storey", "quantile", "fisher", "linear"]
        assert type_ in type_options, "The chosen conformal contamination test p-value is not supported."
        self.type_ = type_
        if type_ == "storey":
            self.Tfun = self.Storey_test_statistic
            self.pfun = self.Storey_pvalue
        elif type_ == "quantile":
            self.Tfun = self.quantile_test_statistic
            self.pfun = self.quantile_pvalue
        elif type_ == "fisher":
            self.Tfun = self.Fisher_test_statistic
            self.pfun = self.Fisher_pvalue
        elif type_ == "linear":
            self.Tfun = self.linear_test_statistic
            self.pfun = self.linear_pvalue

    def compute_conformal_pvalues(self, calibration_scores, test_scores):
        """
        Computing conformal p-values.

        Inputs:
        -------
            calibration_scores : ndarray, size=(n,)
                The conformal scores on the calibration data.
            test_scores : ndarray, size=(m,)
                The conformal scores on the test data.

        Output:
        -------
            conformal_pvalues : ndarray, size=(m,)
                The conformal p-values.
        """
        n, m = len(calibration_scores), len(test_scores)
        conformal_pvalues = np.zeros(m, dtype=np.float32)
        for j in range(m):
            conformal_pvalues[j] = (1 + np.sum(calibration_scores <= test_scores[j]))/(n+1)
        return conformal_pvalues

    def quantile_test_statistic(self, conformal_pvalues, **kwargs):
        """
        Computing the quantile non-conformity statistic.
        """
        n, m, i0 = kwargs["n"], len(conformal_pvalues), kwargs["i0"]
        return round(np.sort(conformal_pvalues)[m-i0-1] * (n+1))

    def Storey_test_statistic(self, conformal_pvalues, **kwargs):
        """
        Computing the Storey non-conformity statistic.
        """
        lambda_ = kwargs["lambda_"]
        return round(np.sum(conformal_pvalues > lambda_))

    def linear_test_statistic(self, conformal_pvalues):
        """
        Computing the summation non-conformity statistic.
        """
        return np.sum(conformal_pvalues)

    # def linear_test_statistic_(self, conformal_pvalues, **kwargs):
    #     """
    #     Computing the summation non-conformity statistic.
    #     """
    #     n, m = kwargs["n"], len(conformal_pvalues)
    #     return np.sum(conformal_pvalues)-m*1/(n+1)

    def Fisher_test_statistic(self, conformal_pvalues, **kwargs):
        """
        Computing the Fisher non-conformity statistic.
        """
        n, m = kwargs["n"], len(conformal_pvalues)
        return -2*m*np.log(1/(n+1)) + 2*np.sum(np.log(conformal_pvalues))

    def quantile_pvalue(self, test_statistic, **kwargs):
        """
        Parallelised implementation computing the quantile p-value
        for the null hypothesis pi <= pi_th when using the quantile
        non-conformity statistic.

        Inputs:
        -------
            test_statistic : float in {1, 2, dots, n+1}
                The quantile test statistic.

        kwargs:
        -------
            pi_th : float in [0, 1)
                The estimate of pi restricted to pi in (0, pi_th).
            n : int
                The size of the null sample.
            m : int
                The size of the test sample.
            i_0 : int in {0, 1, dots, m-1}
                The quantile index hyperparameter.

        Output:
        -------
            pval : float in [0, 1]
                The p-value.
        """
        pi_th, n, m, i0 = kwargs["pi_th"], kwargs["n"], kwargs["m"], kwargs["i0"]
        assert test_statistic >= 0, "Non-negative threshold defining the rejection region."
        if pi_th > 0:
            k_arr = np.arange(0, m+1, 1)
            k_arr_restricted = k_arr[k_arr > i0]
            part1 = binom.pmf(k_arr, m, 1-pi_th)
            part2 = np.ones(m+1, dtype=np.float64)
            part2[k_arr > i0] = nhypergeom.cdf(test_statistic-1, n+k_arr_restricted, n, k_arr_restricted-i0)
            pval = np.sum(part1*part2)
        elif pi_th == 0:
            pval = nhypergeom.cdf(test_statistic-1, n+m, n, m-i0)
        return pval

    def Storey_pvalue(self, test_statistic, **kwargs):
        """
        Parallelised implementation computing the quantile p-value
        for the null hypothesis pi <= pi_th when using the Storey
        non-conformity statistic.
    
        Inputs:
        -------
            test_statistic : int in {0, 1, dots, m}
                The Storey test statistic.

        kwargs:
        -------
            pi_th : float in [0, 1)
                The estimate of pi restricted to pi in (0, pi_th).
            n : int
                The size of the null sample.
            m : int
                The size of the test sample.
            lambda_ : float in (0, 1)
                Storey's hyperparameter.
    
        Output:
        -------
            pval : float in [0, 1]
                The p-value.
        """
        pi_th, n, m, lambda_ = kwargs["pi_th"], kwargs["n"], kwargs["m"], kwargs["lambda_"]
        assert test_statistic >= 0, "Non-negative threshold defining the rejection region."
        if pi_th > 0:
            k_arr = np.arange(0, m+1, 1)
            k_arr_restricted = k_arr[k_arr > test_statistic]
            part1 = binom.pmf(k_arr, m, 1-pi_th)
            part2 = np.ones(m+1, dtype=np.float64)
            part2[k_arr > test_statistic] = nhypergeom.cdf(floor(lambda_*(n+1))-1, n+k_arr_restricted, n, k_arr_restricted-test_statistic)
            pval = np.sum(part1*part2)
        elif pi_th == 0:
            pval = nhypergeom.cdf(floor(lambda_*(n+1))-1, n+m, n, m-test_statistic)
        return pval

    def linear_pvalue(self, test_statistic, **kwargs):
        """
        Parallelised implementation computing the quantile p-value
        for the null hypothesis pi <= pi_th when using the summation
        non-conformity statistic.
    
        Inputs:
        -------
            test_statistic : float
                The linear test statistic.

        kwargs:
        -------
            pi_th : float in [0, 1)
                The estimate of pi restricted to pi in (0, pi_th).
            n : int
                The size of the null sample.
            m : int
                The size of the test sample.
    
        Output:
        -------
            pval : float in [0, 1]
                The p-value.
        """
        pi_th, n, m = kwargs["pi_th"], kwargs["n"], kwargs["m"]
        test_statistic = max(test_statistic, 0)
        # assert test_statistic >= 0, "Non-negative threshold defining the rejection region."
        if pi_th > 0:
            k_arr = np.arange(1, m+1, 1)
            gamma_k_arr = k_arr/n
            xi_k_arr = np.sqrt(1+gamma_k_arr)
            part1 = binom.pmf(k_arr, m, 1-pi_th)
            part2 = irwinhall.cdf((test_statistic+k_arr*(xi_k_arr-1)/2)/xi_k_arr, k_arr)
            pval = pi_th**m + np.sum(part1*part2)
        elif pi_th == 0:
            pval = irwinhall.cdf((test_statistic+m*(np.sqrt(1+m/n)-1)/2)/np.sqrt(1+m/n), m)
        return pval

    def Fisher_pvalue(self, test_statistic, **kwargs):
        """
        Parallelised implementation computing the quantile p-value
        for the null hypothesis pi <= pi_th when using the Fisher
        non-conformity statistic.

        Inputs:
        -------
            test_statistic : float in [0, infty)
                The shifted Fisher test statistic.

        kwargs:
        -------
            pi_th : float in [0, 1)
                The threshold on pi.
            n : int
                The size of the null sample.
            m : int
                The size of the test sample.

        Output:
        -------
            pval : float in [0, 1]
                The p-value.
        """
        pi_th, n, m = kwargs["pi_th"], kwargs["n"], kwargs["m"]
        test_statistic = max(test_statistic, 0)
        # assert test_statistic >= 0, "Non-negative threshold defining the rejection region."
        if pi_th > 0:
            k_arr = np.arange(1, m+1, 1)
            gamma_k_arr = k_arr/n
            xi_k_arr = np.sqrt(1+gamma_k_arr)
            part1 = binom.pmf(k_arr, m, 1-pi_th)
            test_statistic_part = -test_statistic - 2*k_arr*np.log(1/(n+1))
            part2 = 1 - chi2.cdf((test_statistic_part+2*k_arr*(xi_k_arr-1))/xi_k_arr, 2*k_arr)
            pval = pi_th**m + np.sum(part1*part2)
        elif pi_th == 0:
            xi = np.sqrt(1+m/n)
            test_statistic_part = -test_statistic - 2*m*np.log(1/(n+1))
            pval = 1 - chi2.cdf((test_statistic_part+2*m*(xi-1))/xi, 2*m)
        return pval

    def all_conformal_contamination_tests(self, calibration_scores, test_scores, **kwargs):
        """
        Wrapper method for computing all the proposed the conformal
        contamination test p-values.
        """
        conformal_pvalues = self.compute_conformal_pvalues(calibration_scores, test_scores)
        Tstorey = self.Storey_test_statistic(conformal_pvalues, **kwargs)
        Tquantile = self.quantile_test_statistic(conformal_pvalues, **kwargs)
        Tfisher = self.Fisher_test_statistic(conformal_pvalues, **kwargs)
        Tlinear = self.linear_test_statistic(conformal_pvalues)
        pstorey = self.Storey_pvalue(Tstorey, **kwargs)
        pquantile = self.quantile_pvalue(Tquantile, **kwargs)
        pfisher = self.Fisher_pvalue(Tfisher, **kwargs)
        plinear = self.linear_pvalue(Tlinear, **kwargs)
        return pstorey, pquantile, plinear, pfisher

    def all_conformal_contamination_tests_from_conformal_pvalues(self, conformal_pvalues, **kwargs):
        """
        Wrapper method for computing all the proposed the conformal
        contamination test p-values.
        """
        Tstorey = self.Storey_test_statistic(conformal_pvalues, **kwargs)
        Tquantile = self.quantile_test_statistic(conformal_pvalues, **kwargs)
        Tfisher = self.Fisher_test_statistic(conformal_pvalues, **kwargs)
        Tlinear = self.linear_test_statistic(conformal_pvalues)
        pstorey = self.Storey_pvalue(Tstorey, **kwargs)
        pquantile = self.quantile_pvalue(Tquantile, **kwargs)
        pfisher = self.Fisher_pvalue(Tfisher, **kwargs)
        plinear = self.linear_pvalue(Tlinear, **kwargs)
        return pstorey, pquantile, plinear, pfisher

    def __call__(self, calibration_scores, test_scores, **kwargs):
        """
        Wrapper method to compute a conformal contamination test p-value.
        """
        conformal_pvalues = self.compute_conformal_pvalues(calibration_scores, test_scores)
        test_statistic = self.Tfun(conformal_pvalues, **kwargs)
        pvalue = self.pfun(test_statistic, **kwargs)
        return pvalue
