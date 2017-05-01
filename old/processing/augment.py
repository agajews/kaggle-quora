import random


def add_question(qid, match, dictionary):
    if qid in dictionary:
        dictionary[qid].add(match)
    else:
        dictionary[qid] = {match}


def find_matches(qid, positive, matches):
    for match in positive[qid]:
        if match not in matches:
            matches.add(match)
            find_matches(match, positive, matches)


def transitivify(data, **kwargs):
    print('Original data has {} pairs'.format(len(data)))

    questions = {}
    positive = {}
    negative = {}
    for question in data:
        qid1 = question[0]
        qid2 = question[1]
        if qid1 not in questions:
            questions[qid1] = question[2]
        if qid2 not in questions:
            questions[qid2] = question[3]
        if question[4] == 1:
            add_question(qid1, qid2, positive)
            add_question(qid2, qid1, positive)
        else:
            add_question(qid1, qid2, negative)
            add_question(qid2, qid1, negative)

    # print(negative)
    positive_pairs = set()
    for qid in positive.keys():
        matches = set()
        find_matches(qid, positive, matches)
        for match in matches:
            positive_pairs.add((min(qid, match), max(qid, match)))
    print('Generated {} positive pairs'.format(len(positive_pairs)))

    negative_pairs = set()
    for qid in negative.keys():
        if len(negative_pairs) > 3 * len(positive_pairs):
            print('Breaking early')
            break
        matches = set()
        find_matches(qid, negative, matches)
        for match in random.sample(matches, min(len(matches), 5)):
            negative_pairs.add((min(qid, match), max(qid, match)))
    print('Generated {} negative pairs'.format(len(negative_pairs)))

    data = []
    for (qid1, qid2) in positive_pairs:
        data.append((qid1, qid2, questions[qid1], questions[qid2], 1))
    for (qid1, qid2) in negative_pairs:
        data.append((qid1, qid2, questions[qid1], questions[qid2], 0))
    random.shuffle(data)

    return data


def noisify(data, modify_p=0.0, **kwargs):
    tokens = set()

    def noisify_q(question):
        if len(question) and random.random() < modify_p:
            choice = random.choice(list(range(len(question))))
            if question[choice] in tokens:
                question.pop(choice)
            tokens.update(question)
        return question

    new_data = []
    for (qid1, qid2, q1, q2, duplicate) in data:
        new_data.append(
            (qid1, qid2, noisify_q(q1), noisify_q(q2), duplicate))

    return new_data
