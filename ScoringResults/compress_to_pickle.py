#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 10:37:34 2025
"""

import numpy as np
import pickle


if __name__ == "__main__":
    in_data = np.load("temp.npz")
    pi_true = in_data["pi_true"].astype(np.float16)
    pvals_all = np.round((in_data["pvals_all"]*41)).astype(np.uint16)
    pi_th_arr = in_data["pi_th_arr"].astype(np.float16)
    indicator_all = in_data["indicator_all"].astype(np.bool)
    pcon_total = in_data["pcon_total"].astype(np.float32)
    save_dict = {"pi_true": pi_true, "pvals_all": pvals_all, "pi_th_arr": pi_th_arr,
                  "indicator_all": indicator_all, "pcon_total": pcon_total}
    np.savez("temp_compressed.npz", **save_dict)

    with open('temp_compressed.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)