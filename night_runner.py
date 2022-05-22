def main():
    import Regression_DecisionTree
    import Regression_GradientBoosting
    import Regression_NeuralNetwork

    try:
        Regression_DecisionTree.random_forrest()
    except:
        print('Error with Regression_DecisionTree')
    try:
        Regression_GradientBoosting.gradient_boosting()
    except:
        print('Error with Regression_GradientBoosting')
    try:
        Regression_NeuralNetwork.neural_network()
    except:
        print('Error with Regression_NeuralNetwork')


main()