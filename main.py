import os
import argparse
import time
import csv
import sys
import json
import random
import numpy as np
import pprint
import yaml

import torch
import torch.multiprocessing as mp

# import ray
# from ray import tune

import cnn_processing, cnn_model, cnn_trainer


def main():
    parser = argparse.ArgumentParser(description='dos_prediciton_using_cnn')

    parser.add_argument(
        "--config_path",
        default="config.yml",
        type=str,
        help="Location of config file (default: config.json)",
    )
    parser.add_argument(
        "--run_mode",
        default='cnn',
        type=str,
        help="run modes: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis",
    )
    parser.add_argument(
        "--job_name",
        default=None,
        type=str,
        help="name of your job and output files/folders",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="has to been limited as cnn now",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for data split, 0=random",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path of the model .pth file",
    )
    parser.add_argument(
        "--save_model",
        default=None,
        type=str,
        help="Save model",
    )
    parser.add_argument(
        "--load_model",
        default=None,
        type=str,
        help="Load model",
    )
    parser.add_argument(
        "--write_output",
        default=None,
        type=str,
        help="Write outputs to csv",
    )
    parser.add_argument(
        "--parallel",
        default=None,
        type=str,
        help="Use parallel mode (ddp) if available",
    )
    parser.add_argument(
        "--reprocess",
        default=None,
        type=str,
        help="Reprocess data since last run",
    )
    ###Processing arguments
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="Location of data containing structures (json or any other valid format) and accompanying files",)
    parser.add_argument("--format", default=None, type=str, help="format of input data")
    ###Training arguments
    parser.add_argument("--train_ratio", default=None, type=float, help="train ratio")
    parser.add_argument(
        "--val_ratio", default=None, type=float, help="validation ratio"
    )
    parser.add_argument("--test_ratio", default=None, type=float, help="test ratio")
    parser.add_argument(
        "--verbosity", default=None, type=int, help="prints errors every x epochs"
    )
    parser.add_argument(
        "--target_index",
        default=None,
        type=int,
        help="which column to use as target property in the target file",
    )
    ###Model arguments
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="batch size")
    parser.add_argument("--lr", default=None, type=float, help="learning rate")

    ##Get arguments from command line
    args = parser.parse_args(sys.argv[1:])   #解释命令行参数并将其存储在变量args中，args是一个namespace

    ##Open provided config file
    assert os.path.exists(args.config_path), ("Config file not found in " + args.config_path)
    with open(args.config_path, "r", encoding='utf-8') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    ##Update config values from command line
    if args.run_mode != None:
        config["Job"]["run_mode"] = args.run_mode
    run_mode = config["Job"].get("run_mode")


    config["Job"] = config["Job"].get(run_mode) #将 "Job" 部分的值更新为与 run_mode 匹配的子部分
    if config["Job"] == None:
        print("Invalid run mode")
        sys.exit()

    if args.job_name != None:
        config["Job"]["job_name"] = args.job_name  #将配置文件中 "Job" 部分的 "job_name" 键的值设置为命令行参数 args.job_name 的值
    if args.model != None:
        config["Job"]["model"] = args.model
    if args.seed != None:
        config["Job"]["seed"] = args.seed
    if args.model_path != None:
        config["Job"]["model_path"] = args.model_path
    if args.load_model != None:
        config["Job"]["load_model"] = args.load_model
    if args.save_model != None:
        config["Job"]["save_model"] = args.save_model
    if args.write_output != None:
        config["Job"]["write_output"] = args.write_output
    if args.parallel != None:
        config["Job"]["parallel"] = args.parallel
    if args.reprocess != None:
        config["Job"]["reprocess"] = args.reprocess

    if args.data_path != None:
        config["Processing"]["data_path"] = args.data_path
    if args.format != None:
        config["Processing"]["data_format"] = args.format

    if args.train_ratio != None:
        config["Training"]["train_ratio"] = args.train_ratio
    if args.val_ratio != None:
        config["Training"]["val_ratio"] = args.val_ratio
    if args.test_ratio != None:
        config["Training"]["test_ratio"] = args.test_ratio
    if args.verbosity != None:
        config["Training"]["verbosity"] = args.verbosity
    if args.target_index != None:
        config["Training"]["target_index"] = args.target_index

    for key in config["Models"]:
        if args.epochs != None:
            config["Models"][key]["epochs"] = args.epochs
        if args.batch_size != None:
            config["Models"][key]["batch_size"] = args.batch_size
        if args.lr != None:
            config["Models"][key]["lr"] = args.lr

    if run_mode != "Hyperparameter":
        process_start_time = time.time()
        dataset = cnn_processing.get_dataset(config["Processing"]["data_path"], config["Training"]["target_index"], False, None)
        print("paramaters setting and data finding is finished")
        print("--- %s seconds for processing ---" % (time.time() - process_start_time))
    ################################################################################
    #  Training begins
    ################################################################################
    if run_mode == "cnn":
        print("Starting regular training.................................................................")
        world_size = torch.cuda.device_count()
        '''world_size = 0时, rank为cpu'''
        if world_size == 0:
            cnn_trainer.train_regular(
                "cpu",
                world_size,
                config["Processing"]["data_path"],
                config["Job"],
                config["Training"],
                config["Models"],)
        else:
            print('world size > 0 is not supportive now')

    elif run_mode == 'predict':
        print("Starting prediction from trained model")
        cnn_trainer.predict(dataset, config["Training"]["loss"], config["Job"])


if __name__ == '__main__':
    main()

