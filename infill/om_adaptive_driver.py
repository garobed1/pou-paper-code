"""
Driver for adaptive sampling in the SMT framework
"""

import traceback
import inspect

import numpy as np

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError
from openmdao.drivers.doe_generators import DOEGenerator, ListGenerator

from openmdao.utils.mpi import MPI

from openmdao.recorders.sqlite_recorder import SqliteRecorder
from openmdao.recorders.case_reader import CaseReader

from infill.getxnew import getxnew, adaptivesampling
from infill.refinecriteria import ASCriteria

#from surrogate.om_smt_interface import SMTSurrogate
from smt.surrogate_models import SurrogateModel as SMT_SM

from optimization.defaults import DefaultOptOptions

from utils.om_utils import get_om_dict
from utils.sutils import divide_cases

#TODO: NEED TO SEPARATE OUT adaptivesampling FUNCTION INTO 3 PARTS

class AdaptiveDriver(Driver):

    def __init__(self, criteria=None, **kwargs):
        """
        Construct A DOEDriver.
        """

        if criteria and not isinstance(criteria, ASCriteria):
            if inspect.isclass(criteria):
                raise TypeError("AdaptiveDriver requires an instance of ASCriteria, "
                                "but a class object was found: %s"
                                % criteria.__name__)
            else:
                raise TypeError("AdaptiveDriver requires an instance of ASCriteria, "
                                "but an instance of %s was found."
                                % type(criteria).__name__)

        super().__init__(**kwargs)

        # What we support

        # What we don't support
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        if criteria is not None:
            self.options['init_crit'] = criteria

        self._name = ''
        self._problem_comm = None
        self._color = None
        self._cr = None # case reader
        self._ins_all = None
        self._ins_search_list = None
        
        self.first_run = True

        

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        #TODO: Individualized criteria limits for stopping criteria?
        # This also contains number of points per batch, won't be too relevant
        self.options.declare('init_criteria', types=(ASCriteria), default=ASCriteria(),
                             desc='The refinement/infill criteria. If default, no cases are generated.')
        self.options.declare('smt_model', types=list, default=[SMT_SM()], #(SMT_SM)
                             desc='The SMT surrogate to use')
        
        # 0: om name to smt index variable mapping
        self.options.declare('smt_meta', types=tuple, default=(),
                             desc='SMT metadata, required across components and driver that rely on SMT')
        
        self.options.declare('ref_outs', types=(str, list), default=None,
                             desc='outputs to refine about, most methods only take one. default picks first output')
        
        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute cases in parallel.')
        self.options.declare('procs_per_model', types=int, default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')
        self.options.declare('max_runs', types=int, default=1, lower=1,
                             desc='Maximum number of points to add, if stopping criteria not met.')
        self.options.declare('crit_tol', types=float, default=0.0, lower=0.0,
                             desc='Stopping criteria')                     
        self.options.declare('opt_options', types=dict, default=DefaultOptOptions,
                             desc='Options for optimizations over infill criteria functions')                     

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        self._problem_comm = comm

        if MPI:
            procs_per_model = self.options['procs_per_model']

            full_size = comm.size
            size = full_size // procs_per_model
            if full_size != size * procs_per_model:
                raise RuntimeError("The total number of processors is not evenly divisible by the "
                                   "specified number of processors per model.\n Provide a "
                                   "number of processors that is a multiple of %d, or "
                                   "specify a number of processors per model that divides "
                                   "into %d." % (procs_per_model, full_size))
            color = self._color = comm.rank % size
            model_comm = comm.Split(color)
        else:
            model_comm = comm

        return model_comm

    def _set_name(self):
        """
        Set the name of this adaptive sampling driver and its case generator.

        Returns
        -------
        str
            The name of this AS driver and its case generator.
        """
        criteria = self.criteria

        rcrit_type = type(criteria).__name__.replace('Criteria', '')
        if rcrit_type == 'ASCriteria':
            self._name = 'ASDriver'  # Empty generator
        else:
            self._name = 'ASDriver_' + rcrit_type

        return self._name
    
    def _get_name(self):
        """
        Get the name of this adaptive sampling driver and its criteria.

        Returns
        -------
        str
            The name of this adaptive sampling driver and its criteria.
        """
        return self._name


    
        
#TODO: Get X New Step
    def run(self):
        """
        Generate new points and run the model for each set of new input values.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        self.iter_count = 0
        max_iter = self.options['max_iter']
        crit_tol = self.options['crit_tol']
        opt_options = self.options['opt_options']
        surrogates = self.options['SMT_SM']
        smt_map = self.options['smt_meta'][0]
        xlimits = self.options['smt_meta'][1]

        # get list of inputs to refine over (usually either dvs or ucs)
        # TODO: criteria needs knowledge of this too
        self._ins_all = list(self._designvars)
        self._search_list = self.options['search_list']

        # get list of relevant outputs temporary solution
        # TODO: distinguish between surrogate outs and non-surrogate
        all_outs = self._problem.model.list_outputs()
        ref_outs = self.options['ref_outs']

        # get criteria (it is updated with surrogate)
        criteria = self.criteria
        if self.first_run:
            criteria = self.options['init_criteria']

        # set driver name with current criteria
        self._set_name()

        # get variable bounds (needs SMT model specifically, rename file to reflect this?)
        self.criteria.smt_model

        # run adaptive sampling procedure for requested number of points, and retrieve new criteria
        while self.iter_count < max_iter or crit_tol < criteria.get_energy():
            
            # determine sample locations, this can only be done serially
            nnew = min(criteria.nnew, max_iter - self.iter_count)
            xnews = getxnew(criteria, 
                xlimits, 
                nnew, 
                options=opt_options)


            # parallelization if possible
            rcases = divide_cases(xnews.shape[0], self._problem_comm.size)

            cases = []

            # convert xnews to cases and run
            for i in rcases[self._problem_comm.rank]:
                case = get_om_dict(xnews[i], smt_map)
                self._run_case(case)
                
                out_dict = {}
                for item in all_outs:
                    out_dict[item] = self._problem.get_val(item)

                # compute grads if the surrogates support it
                grad_dict = {}
                if surrogates[0].supports["training_derivatives"]:
                    # returned in a flat dict
                    grad = self._run_grad(case)

                cases.append((case, out_dict)) # need all outputs for the surrogates if needed

            cases = self._problem_comm.Allgather(cases)
            self.iter_count += xnews.shape[0]

            for i in range(len(rcases)):
                for j in range(len(rcases[i])):
                    

            # update surrogate(s)
            for surr in surrogates:
                # get existing points
                xt = surr.training_points[None][0][0]
                yt = surr.training_points[None][0][1]
                nt = xt.shape[0]
                dim = xt.shape[1]
                if surr.supports["training_derivatives"]:
                    gt = np.zeros([nt,dim])
                    for j in range(dim):
                        gt[:,j] = surr.training_points[None][j+1][1]

                # add new points to the model
                xt = np.append(xt, xnews, axis=0)
                yt = np.append(yt, func(xnew), axis=0)
                gt = np.append(gt, np.zeros([xnews.shape[0], dim]), axis=0)

                # retrain

            #NOTE: Criteria may depend on just one output, or pareto solution for several
            # Need to implement multi output criteria/option to choose which output to adapt for

            # update criteria

        self.first_run = False

        return False

#TODO: Compute new points
    def _run_case(self, case):
        """
        Run case, save exception info and mark the metadata if the case fails.

        Parameters
        ----------
        case : list
            list of name, value tuples for the variables.
        """
        metadata = {}

        for dv_name, dv_val in case:
            try:
                msg = None
                if isinstance(dv_val, np.ndarray):
                    self.set_design_var(dv_name, dv_val.flatten())
                else:
                    self.set_design_var(dv_name, dv_val)
            except ValueError as err:
                msg = "Error assigning %s = %s: " % (dv_name, dv_val) + str(err)
            finally:
                if msg:
                    raise(ValueError(msg))

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            try:
                self._problem().model.run_solve_nonlinear()
                metadata['success'] = 1
                metadata['msg'] = ''
            except AnalysisError:
                metadata['success'] = 0
                metadata['msg'] = traceback.format_exc()
            except Exception:
                metadata['success'] = 0
                metadata['msg'] = traceback.format_exc()
                print(metadata['msg'])

            # save reference to metadata for use in record_iteration
            self._metadata = metadata


    def _run_grad(self, case):
        """
        Evaluate and return the total gradients of the model

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.

        Returns
        -------
        ndarray
            Gradient of objective with respect to input array.
        """
        try:
            grad = self._compute_totals(of=self._problem.model.list_outputs(), 
                                        wrt=self._ins_all,
                                        return_format='array')
            self._grad_cache = grad

            # First time through, check for zero row/col.
            if self._check_jac:
                raise_error = self.options['singular_jac_behavior'] == 'error'
                self._total_jac.check_total_jac(raise_error=raise_error,
                                                tol=self.options['singular_jac_tol'])
                self._check_jac = False

        except Exception as msg:
            self._exc_info = msg
            return np.array([[]])

        # print("Gradients calculated for objective")
        # print('   xnew', x_new)
        # print('   grad', grad[0, :])

        return grad

#TODO: Add points to model/retrain


    def _setup_recording(self):
        """
        Set up case recording.
        """

        # Internal recorder for using new data
        rname = f'{self._get_name}_adapt_rec'
        self_rec = SqliteRecorder(rname)
        self_rec.options['record_params'] = True
        self_rec.options['record_metadata'] = True
        self_rec.options['record_derivatives'] = True
        self.add_recorder(self_rec)

        # Internal reader
        self._cr = CaseReader(rname)

        if MPI:
            procs_per_model = self.options['procs_per_model']

            for recorder in self._rec_mgr:
                recorder._parallel = True

                # if SqliteRecorder, write cases only on procs up to the number
                # of parallel DOEs (i.e. on the root procs for the cases)
                if isinstance(recorder, SqliteRecorder):
                    if procs_per_model == 1:
                        recorder._record_on_proc = True
                    else:
                        size = self._problem_comm.size // procs_per_model
                        if self._problem_comm.rank < size:
                            recorder._record_on_proc = True
                        else:
                            recorder._record_on_proc = False

        super()._setup_recording()

    def _get_recorder_metadata(self, case_name):
        """
        Return metadata from the latest iteration for use in the recorder.

        Parameters
        ----------
        case_name : str
            Name of current case.

        Returns
        -------
        dict
            Metadata dictionary for the recorder.
        """
        self._metadata['name'] = case_name
        return self._metadata