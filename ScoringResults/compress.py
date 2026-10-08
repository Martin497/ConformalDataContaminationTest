#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:42:15 2025
"""

import numpy as np
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    folder = "MNIST_partial_fn"
    file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    num_images = len(file_names)

    for file_name in file_names:
        if file_name[-4:] == ".npz":
            in_data = np.load(f"{folder}/{file_name}")

            # for key_ in in_data.keys():
            #     print(key_)
            #     print(np.prod(in_data[key_].shape))

            pi_true = in_data["pi_true"].astype(np.float16)
            pvals_all = np.round((in_data["pvals_all"]*41)).astype(np.uint16)
            pi_th_arr = in_data["pi_th_arr"].astype(np.float16)
            indicator_all = in_data["indicator_all"].astype(np.bool)
            pcon_total = in_data["pcon_total"].astype(np.float32)
            save_dict = {"pi_true": pi_true, "pvals_all": pvals_all, "pi_th_arr": pi_th_arr,
                          "indicator_all": indicator_all, "pcon_total": pcon_total}
            np.savez(f"{folder}/{file_name}", **save_dict)
    
    # loadname = "temp"
    # in_data = np.load("temp.npz")
    # pi_true = in_data["pi_true"]
    # pvals_all = in_data["pvals_all"]
    # pi_th_arr = in_data["pi_th_arr"]
    # indicator_all = in_data["indicator_all"]
    # pcon_total = in_data["pcon_total"]