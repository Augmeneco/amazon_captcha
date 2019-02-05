from fann2 import libfann as fann

# 0.01038538757711649
# 0.014601808972656727

connection_rate = 1
learning_rate = 0.3
layers = [1225, 300, 26]
ENG_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

desired_error = 0.001
max_iterations = 5000
iterations_between_reports = 2

ann = fann.neural_net()
ann.create_sparse_array(1, layers)
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(fann.SIGMOID_SYMMETRIC_STEPWISE)
ann.set_activation_function_hidden(fann.SIGMOID_SYMMETRIC_STEPWISE)
ann.randomize_weights(-1.0, 1.0)

ann.train_on_file('./train.dat', max_iterations, iterations_between_reports, desired_error)

#td = fann.training_data()
#td.read_train_from_file('./train.dat')
#for _ in range(max_iterations):
#    ann.reset_MSE()
#    for inpt, otpt in td.get_input(), td.get_output():
#        ann.train(inpt, otpt)
#    if ann.get_MSE() <= desired_error:
#        break

ann.save('captcha.net')
