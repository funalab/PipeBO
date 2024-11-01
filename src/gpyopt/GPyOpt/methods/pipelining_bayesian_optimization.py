import GPyOpt
import numpy as np
import time
from ..methods import BayesianOptimization
from ..core.errors import InvalidConfigError
from ..optimization.acquisition_optimizer import ContextManager
from ..util.duplicate_manager import DuplicateManager

import sys
from json_error import JsonError


class PipeliningBayesianOptimization(BayesianOptimization):
    def __init__(self, f, domain = None, constraints = None, cost_withGradients = None, model_type = 'GP', X = None, Y = None,
    	initial_design_numdata = 5, initial_design_type='random', acquisition_type ='EI', normalize_Y = True,
        exact_feval = False, acquisition_optimizer_type = 'lbfgs', model_update_interval=1, evaluator_type = 'sequential',
        batch_size = 1, num_cores = 1, verbosity=False, verbosity_model = False, maximize=False, de_duplication=False,
        process_setting = None, intermediate_update = True, other_pipeline_LP = True, **kwargs):

        if batch_size == 1 and evaluator_type == 'local_penalization_pipe' :
            print('\033[35m' + 'Warning if batch_size = 1 with pipelining, you should use evaluator pipelining')
            print('evaluator_type was changed to pipelining' + '\033[0m')
            evaluator_type = 'pipelining'

        if batch_size != 1 and evaluator_type == 'pipelining' :
            print('\033[35m' + 'Warning if batch_size = 1 with pipelining, you should use evaluator local_penalization_pipe')
            print('evaluator_type was changed to local_penalization_pipe' + '\033[0m')
            evaluator_type = 'local_penalization_pipe'

        try:
            if process_setting != None:
                self.process_setting = process_setting
            else:
                raise JsonError('\033[31mError: Parameter could not be read. Please check the json file\033[0m')
        except Exception as e:
            print(e)
            sys.exit(1)

        self.intermediate_update = intermediate_update
        self.other_pipline_LP = other_pipeline_LP

        super().__init__(f, domain, constraints, cost_withGradients, model_type, X, Y,
    	initial_design_numdata, initial_design_type, acquisition_type, normalize_Y,
        exact_feval, acquisition_optimizer_type, model_update_interval, evaluator_type,
        batch_size, num_cores, verbosity, verbosity_model, maximize, de_duplication, **kwargs)


    def run_optimization(self, max_iter = 0, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, save_models_parameters= True, report_file = None, evaluations_file = None, models_file=None):
        print('Error the class is for piepelining bayesian optimization\nYou should use run_pipelining_optimization insted of run_optimization')
        sys.exit()

    def run_pipelining_optimization(self, max_iter = 0, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, save_models_parameters= True,
                                    report_file = None, evaluations_file = None, models_file=None,  function_best = None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param report_file: file to which the results of the optimization are saved (default, None).
        :param evaluations_file: file to which the evalations are saved (default, None).
        :param models_file: file to which the model parameters are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.save_models_parameters = save_models_parameters
        self.report_file = report_file
        self.evaluations_file = evaluations_file
        self.models_file = models_file
        self.model_parameters_iterations = None
        self.context = context

        # --- Check if we can save the model parameters in each iteration
        if self.save_models_parameters == True:
            if not (isinstance(self.model, GPyOpt.models.GPModel) or isinstance(self.model, GPyOpt.models.GPModel_MCMC)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False

        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        pipeline_conditions = self.X.reshape([len(self.process_setting), self.batch_size, len(self.domain)])

        self.X = pipeline_conditions[0, ]
        self.Y = self.Y[:self.batch_size]

        save_pre_update = np.empty([0,len(self.domain)])

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            # --- Update model
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError:
                break

            if (self.num_acquisitions >= self.max_iter):
                    # or (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)):
                break

            save_pre_update = np.append(save_pre_update, pipeline_conditions[1, ], axis=0)
            pipeline_conditions = self._compute_fix_next_evaluations_pipelining(conditions = pipeline_conditions, process_setting = self.process_setting,
                                                                                 intermediate_update=self.intermediate_update, other_pipeline_LP=self.other_pipline_LP)
            self.suggested_sample = pipeline_conditions[1, ]

            # Delete conditions added to the model
            pipeline_conditions = np.delete(pipeline_conditions, obj = 0, axis = 0)

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()


            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

        # --- Stop messages and execution time
        self._compute_results()

        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)

    def _compute_fix_next_evaluations_pipelining(self, conditions, process_setting, intermediate_update, other_pipeline_LP, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """

        ## --- Update the context if any
        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)

        ### --- Activate de_duplication
        if self.de_duplication:
            duplicate_manager = DuplicateManager(space=self.space, zipped_X=self.X, pending_zipped_X=pending_zipped_X, ignored_zipped_X=ignored_zipped_X)
        else:
            duplicate_manager = None

        return self.evaluator.compute_batch_pipelining(conditions, process_setting, duplicate_manager=duplicate_manager,
                                                       intermediate_update=intermediate_update, other_pipeline_LP=other_pipeline_LP, context_manager= self.acquisition.optimizer.context_manager)

        ### We zip the value in case there are categorical variables
        # return self.space.zip_inputs(self.evaluator.compute_batch_pipelining(conditions, process_setting, duplicate_manager=duplicate_manager, context_manager= self.acquisition.optimizer.context_manager))