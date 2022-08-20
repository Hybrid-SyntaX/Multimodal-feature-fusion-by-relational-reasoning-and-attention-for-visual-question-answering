import csv
import json
import os
def method_1():
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    sep=';'
    with open('../logs_dev/results.tsv','r',encoding='utf-8') as results_csv:
        with open('../logs_dev/results_csv_to_json.json', 'w',encoding='utf-8') as results_json:
            #fieldnames = results_csv.readlines()[0].split(sep)
            reader = csv.DictReader(results_csv,delimiter='\t',quoting=csv.QUOTE_NONE)
            #reader.next()

            for row in reader:
                json.dump(row, results_json)
                results_json.write('\n')

import pandas as pd
csv_file='../logs/results.tsv'
json_file='../logs/results_csv_to_json.json'
csv_file = pd.DataFrame(pd.read_csv(csv_file, sep = "\t", header = 0, index_col = False))
csv_file.to_json(json_file, orient = "records", date_format = "epoch", double_precision = 10, force_ascii = True,
                 date_unit = "ms", default_handler = None)
