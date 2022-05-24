def main(min, max, n):
    import Regression_DecisionTree
    import Regression_GradientBoosting
    import Regression_NeuralNetwork

    try:
        Regression_DecisionTree.random_forrest(min, max, n)
    except:
        print('Error with Regression_DecisionTree')
    try:
        Regression_GradientBoosting.gradient_boosting(min, max, n)
    except:
        print('Error with Regression_GradientBoosting')
    try:
        Regression_NeuralNetwork.neural_network(min, max, n)
    except:
        print('Error with Regression_NeuralNetwork')


main(-1_000_000, 1_000_000, 300_000)