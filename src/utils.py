
# Are these integer to sentiment mappings ok? or should we use negative numbers for
# negative emotions?
INT_TO_SENTIMENT_DICT = {
    0 : "Extremely Negative",
    1 : "Negative",
    2 : "Neutral",
    3 : "Positive",
    4 : "Extremely Positive"
}

SENTIMENT_TO_INT_DICT = {v: k for k, v in INT_TO_SENTIMENT_DICT.items()}

COVID_TO_STANFORD_DICT = {
    0: 0,
    1: 0,
    2: 2,
    3: 4,
    4: 4
}

def int_to_sentiment(val):
    if val not in INT_TO_SENTIMENT_DICT:
        raise ValueError("Invalid sentiment integer: {}".format(val))
    return INT_TO_SENTIMENT_DICT[val]

def sentiment_to_int(val): 
    if val not in SENTIMENT_TO_INT_DICT:
        raise ValueError("Invalid sentiment string: {}".format(val))
    return SENTIMENT_TO_INT_DICT[val]

def reduce_sentiment(val):
    if val == "Extremely Negative":
        return "Negative"
    elif val == "Extremely Positive":
        return "Positive"
    return val

def covid_to_stanford(val):
    if val not in COVID_TO_STANFORD_DICT:
        raise ValueError("Invalid sentiment string: {}".format(val))
    return COVID_TO_STANFORD_DICT[val]