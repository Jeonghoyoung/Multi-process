import pandas as pd
import argparse
import util.file_util as ft
import translate_crawling as tc


def args():
    parser = argparse.ArgumentParser(usage='usage', description='Usage of parameters ')
    parser.add_argument('--src_lang', required=False, default='ko')
    parser.add_argument('--tgt_lang', required=False, default='en')
    parser.add_argument('--max_delay_time', required=False, default= 5)

    parser.add_argument('--input', required=False, default='../data/enko/20220930/btxt_2dir_en-XX-ko-YY_Dialog__insightvessel-202209-test.ko')
    parser.add_argument('--output', required=False,default='../data/enko/test.en')
    return parser.parse_args()


def main():
    config = args()
    input_path = config.input
    output_path = config.output
    src_lang = config.src_lang
    tgt_lang = config.tgt_lang
    delay_time = config.max_delay_time

    file_name = input_path.split('/')[-1].split('.')[0]
    print(file_name)
    src_list = ft.get_all_lines(f'{input_path}')

    result_google = tc.google_trans(src_list, src_lang, tgt_lang)
    ft.write_list_file(result_google, f'{output_path}')


if __name__ == '__main__':
    main()