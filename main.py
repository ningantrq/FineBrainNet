import util
from experiment import MaintaskTrain
from experiment import MaintaskTest
from subtask_experiment import SubtaskTrain
from subtask_experiment import SubtaskTest
from test_experiment import Test


if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()
    sub_argv=util.sub_option.parse()

    # run and analyze experiment
    #if not argv.no_train: MaintaskTrain(argv)
    #if not argv.no_test: MaintaskTest(argv)
    #if not argv.no_train: SubtaskTrain(sub_argv)


    if not argv.no_test: SubtaskTest(sub_argv)
    exit(0)
