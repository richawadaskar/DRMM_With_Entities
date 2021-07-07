from collections import OrderedDict
from operator import itemgetter


word_weights = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
entity_weights = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

word_weights2 = [0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60]
entity_weights2 = [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]

word_weights3 = [0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70]
entity_weights3 = [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]


def combine_ranklists(word_weight, entity_weight, index_for_naming, weight_vector):
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
    combined_filename = "ranklists" + weight_vector + "/Combined_DRMM_title_" + str(index_for_naming) + ".ranklist"
    save_ranklist_file = open(combined_filename, "w")

    for itQ in combined_ranklist:
        irank = 0
        for itRanklist in combined_ranklist[itQ]:
            line = itQ +  "\tQ0\t" + itRanklist + "\t" + str(irank) + "\t" + str(combined_ranklist[itQ][itRanklist]) + "\tNN4IR-LCH-IDF-Combined\n"
            save_ranklist_file.write(line)
            irank += 1

    save_ranklist_file.close()


for i in range(0, len(word_weights)):
    combine_ranklists(word_weights[i], entity_weights[i], i, "1")
    combine_ranklists(word_weights2[i], entity_weights2[i], i, "2")
    combine_ranklists(word_weights3[i], entity_weights3[i], i, "3")
