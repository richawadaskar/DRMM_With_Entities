from collections import OrderedDict
from operator import itemgetter


word_weights = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
entity_weights = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

def combine_ranklists(word_weight, entity_weight, index_for_naming):
    # read from 2 ranklists (DRMM Word and DRMM Entity)
    word_ranklist = open("DRMM-LCH-IDF-rob04-title.ranklist", "r")
    entity_ranklist = open("DRMM-Entity-rob04-title.ranklist", "r")

    # join individual ranklists
    combined_ranklist = {}
    for line in word_ranklist:
        query_no, q0, doc_id, rank_no, score, label = line.split('\t')
        if query_no not in combined_ranklist:
            combined_ranklist[query_no] = {}
        combined_ranklist[query_no][doc_id] = word_weight * float(score)

    for line in entity_ranklist:
        query_no, q0, doc_id, rank_no, score, label = line.split('\t')
        if doc_id not in combined_ranklist[query_no]:
            print("Doc ID wasn't found for query ... ", query_no, doc_id)
            final_score = 0
            # currently only considering documents ranked by both word and entity embeddings
            combined_ranklist[query_no][doc_id] = final_score
        else:
            final_score = (float(combined_ranklist[query_no][doc_id]) + float(score))/2.0
            combined_ranklist[query_no][doc_id] = entity_weight * final_score

    word_ranklist.close()
    entity_ranklist.close()

    for query in combined_ranklist:
        reranked = OrderedDict(sorted(combined_ranklist[query].items(), key=itemgetter(1), reverse=True))
        combined_ranklist[query] = reranked

    # produce final ranklist for each query
    combined_filename = "Combined_DRMM_title_" + str(index_for_naming) + ".ranklist"
    save_ranklist_file = open(combined_filename, "w")

    for itQ in combined_ranklist:
        irank = 0
        for itRanklist in combined_ranklist[itQ]:
            line = itQ +  "\tQ0\t" + itRanklist + "\t" + str(irank) + "\t" + str(combined_ranklist[itQ][itRanklist]) + "\tNN4IR-LCH-IDF-Combined\n"
            save_ranklist_file.write(line)
            irank += 1

    save_ranklist_file.close()


for i in range(0, len(word_weights)):
    combine_ranklists(word_weights[i], entity_weights[i], i)
