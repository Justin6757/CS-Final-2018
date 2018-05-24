from lstm_network import create

neural_net = create()


def get_sentiment(sentence):
    positive_score = neural_net.predict(sentence)[0]
    negative_score = neural_net.predict(sentence)[1]
    print(f'Positive: {positive_score}\nNegative: {negative_score}')
    return positive_score + negative_score


def test():
    neural_net.test_model()

# while True:
#     get_sentiment(input())

# test()
