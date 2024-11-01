import numpy as np
import scipy
from .batch_local_penalization import LocalPenalization
from ...util.general import samples_multidimensional_uniform
import sys

class LocalPenalization_Pipelining(LocalPenalization):
    """
    Class for the method on 'Bayesian optimization with pipelining'
    """

    def compute_batch_pipelining(self, pipeline_conditions, process_setting, duplicate_manager=None, context_manager=None,
                                 intermediate_update = True, other_pipeline_LP = True):
        """
        Computes the elements of the batch sequentially by penalizing the acquisition.
        """
        from ...acquisitions import AcquisitionLP
        assert isinstance(self.acquisition, AcquisitionLP)

        self.acquisition.update_batches(None,None,None)

        # ---------- Approximate the constants of the the method
        L = estimate_L(self.acquisition.model.model,self.acquisition.space.get_bounds())
        Min = self.acquisition.model.model.Y.min()

        return_X = np.copy(pipeline_conditions)
        return_X = np.append(return_X, np.empty(((1, ) + pipeline_conditions.shape[1:])),axis=0)
        X_batch = np.array([])

        for i in range(1,len(process_setting)):
            if intermediate_update:
                new_sample = self.acquisition.optimize_fix(condtion = pipeline_conditions[i, 0, :sum(process_setting[:len(process_setting) - i])])[0]
                X_batch = np.vstack((X_batch,new_sample)) if X_batch.size else new_sample
                return_X[i, 0] = new_sample
            else:
                X_batch = np.vstack((X_batch,return_X[i, 0])) if X_batch.size else return_X[i, 0]
            k=1

            while k<self.batch_size:
                if intermediate_update:
                    self.acquisition.update_batches(X_batch,L,Min)
                    new_sample = self.acquisition.optimize_fix(condtion = pipeline_conditions[i, k, :sum(process_setting[:len(process_setting) - i])])[0]
                    X_batch = np.vstack((X_batch,new_sample))
                    return_X[i , k] = new_sample
                else:
                    X_batch = np.vstack((X_batch,return_X[i, k]))
                k +=1
            k =0
            if not other_pipeline_LP:
                X_batch = np.array([])
                self.acquisition.update_batches(None,None,None)
            else:
                self.acquisition.update_batches(X_batch,L,Min)

        new_sample = self.acquisition.optimize()[0]
        X_batch = np.vstack((X_batch,new_sample)) if X_batch.size else new_sample
        return_X[-1, 0] = new_sample
        k=1

        while k<self.batch_size:
            self.acquisition.update_batches(X_batch,L,Min)
            new_sample = self.acquisition.optimize()[0]
            X_batch = np.vstack((X_batch,new_sample))
            return_X[-1, k] = new_sample
            k +=1

        # --- Back to the non-penalized acquisition
        self.acquisition.update_batches(None,None,None)
        return return_X

def estimate_L(model,bounds,storehistory=True):
    """
    Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
    """
    def df(x,model,x0):
        x = np.atleast_2d(x)
        dmdx,_ = model.predictive_gradients(x)
        if dmdx.ndim>2:
            dmdx = dmdx.reshape(dmdx.shape[:2])
        res = np.sqrt((dmdx*dmdx).sum(1)) # simply take the norm of the expectation of the gradient
        return -res

    samples = samples_multidimensional_uniform(bounds,500)
    samples = np.vstack([samples,model.X])
    pred_samples = df(samples,model,0)
    x0 = samples[np.argmin(pred_samples)]
    res = scipy.optimize.minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (model,x0), options = {'maxiter': 200})
    minusL = float(res.fun)
    L = -minusL
    if L<1e-7: L=10  ## to avoid problems in cases in which the model is flat.
    return L