# Add your import statements here



# Add any utility functions here
def ground_truth_non_relevance_metrics(query_rels,query_ids):
    query_count = len(query_ids)
    query_ids = sorted(query_ids)
    ground_truth_list = [ [] for i in range(query_count)]
    for query_info in query_rels:
        query_val = int(query_info["query_num"])
        if query_val in query_ids:
            query_i = query_ids.index(query_val)
            ground_truth_list[query_i].append(int(query_info["id"]))

    return ground_truth_list