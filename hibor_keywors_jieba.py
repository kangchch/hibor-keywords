
import json
import pdb

import jieba as jb
import jieba.analyse as jba
import pandas as pd
import argparse as ap

def process_hibor(hibor_data, stop_words, verbose, results_file):
    def read_csv_data():
        abstruct = {}
        df = pd.read_csv(hibor_data, keep_default_na=False, encoding='utf-8')
        for index, row in df.iterrows():
            abstruct[row.get('DId')] = row.get('DocAbstruct', 'aaaa')
        return abstruct

    def to_csv_data(res_lst, results_file):
        columns = ['DId', 'DocAbstruct', 'Keywords']
        df = pd.DataFrame(res_lst)
        df.to_csv(results_file, columns=columns, header=True, index=False, mode= 'w', encoding='utf8')

    def jba_tfidf(items):
        results = {}
        res_lst = []
        for w in items:
            res = jba.extract_tags(items[w] if items[w] else 'test', topK=10, withWeight=True)
            results = {item[0]:item[1] for item in res}
            res_lst.append(results)
            # pdb.set_trace()
        return res_lst

    def jba_textrank(items):
        results = {}
        res_lst = []
        n = 0
        for did in items:
            n+=1
            if n%1000 == 0:
                print(n)
            res = jba.textrank(items[did] if items[did] else 'test', topK=8, withWeight=True)
            # pdb.set_trace()
            # results = {item[0]:item[1] for item in res}
            keywords = ','.join([item[0] for item in res])
            results_keys = {'DId': did, 'DocAbstruct': items[did], 'Keywords': keywords}
            res_lst.append(results_keys)
        return res_lst

    df = read_csv_data()

    jba.set_stop_words(stop_words)
    ''' TF-IDF '''
    if verbose == 'tfidf':
        tfidf_res = jba_tfidf(df)
        to_csv_data(tfidf_res, results_file)
    ''' TextRank'''
    if verbose=='textrank':
        textrank_res = jba_textrank(df)
        to_csv_data(textrank_res, results_file)

def main():

    parser = ap.ArgumentParser()
    parser.add_argument('-hc', '--hibor_csv', default='hibor_utf8.csv')
    parser.add_argument('-sw', '--stop_words', default='stop_words.txt')
    parser.add_argument('-v', '--verbose', default='textrank')
    parser.add_argument('-o', '--results_file', default='results_file.csv')
    args = parser.parse_args()

    process_hibor(args.hibor_csv, args.stop_words, args.verbose, args.results_file)

if __name__== "__main__":

    main()
