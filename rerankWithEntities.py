from collections import OrderedDict
from operator import itemgetter

# read from 2 ranklists (DRMM Word and DRMM Entity)
word_ranklist = open("DRMM-LCH-IDF-rob04-title.ranklist", "r")
entity_ranklist = open("DRMM-Entity-rob04-title.ranklist", "r")

# join individual ranklists
combined_ranklist = {}
for line in word_ranklist:
    query_no, q0, doc_id, rank_no, score, label = line.split('\t')
    if query_no not in combined_ranklist:
        combined_ranklist[query_no] = {}
    combined_ranklist[query_no][doc_id] = float(score)

for line in entity_ranklist:
    query_no, q0, doc_id, rank_no, score, label = line.split('\t')
    if doc_id not in combined_ranklist[query_no]:
        print("Doc ID wasn't found for query ... ", query_no, doc_id)
        final_score = 0
        combined_ranklist[query_no][doc_id] = final_score
    else:
        final_score = (float(combined_ranklist[query_no][doc_id]) + float(score))/2.0
        combined_ranklist[query_no][doc_id] = final_score

word_ranklist.close()
entity_ranklist.close()

for query in combined_ranklist:
    reranked = OrderedDict(sorted(combined_ranklist[query].items(), key=itemgetter(1), reverse=True))
    combined_ranklist[query] = reranked

# produce final ranklist for each query
save_ranklist_file = open("Combined_DRMM_title.ranklist", "w")

for itQ in combined_ranklist:
    irank = 0
    for itRanklist in combined_ranklist[itQ]:
        line = itQ +  "\tQ0\t" + itRanklist + "\t" + str(irank) + "\t" + str(combined_ranklist[itQ][itRanklist]) + "\tNN4IR-LCH-IDF-Combined\n"
        save_ranklist_file.write(line)
        irank += 1

save_ranklist_file.close()
