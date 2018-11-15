# -*- coding: UTF-8 -*-

import jieba
import codecs

def encode_sentence(filename):
    user_dict = 'D:/BaiduNetdiskDownload/dataset_dir/user_dict.txt'
    jieba.load_userdict(user_dict)
    row = 0
    out_file = codecs.open('D:/BaiduNetdiskDownload/dataset_dir/out/out_cut_word.txt', "w", encoding='UTF-8')
    for k, v in enumerate(open(filename, encoding='UTF-8')):
        lines = v.strip().split('\t')
        question = lines[0].strip()
        answer = lines[1].strip()

        cut_question = jieba.cut(question, cut_all=False)
        cut_answer = jieba.cut(answer, cut_all=False)

        if row < 10:
            proced_question_list = []
            for proc_q in cut_question:
                proced_question_list.append(proc_q)
            proced_question_str = ' '.join(proced_question_list)

            proced_answer_list = []
            for proc_a in cut_answer:
                proced_answer_list.append(proc_a)
            proced_answer_str = ' '.join(proced_answer_list)
            #print(proced_question_str, ':::', proced_answer_str)
            proced_line = '\t'.join((proced_question_str, proced_answer_str))

            out_file.write(proced_line+"\n")
            # print(proced_question, proced_answer)
            #print(v.strip())
            row += 1

if __name__ == '__main__':
    filename = 'D:/BaiduNetdiskDownload/dataset_dir/tieba.dialogues'
    encode_sentence(filename)


