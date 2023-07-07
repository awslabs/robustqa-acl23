def flatten_answers(answers):
    final_answers = []
    for ans in answers:
        if type(ans) is list:
            final_answers.extend(flatten_answers(ans))
        else:
            final_answers.append(ans)
    return final_answers