"""Dynamic Parcel Aggregation with Clustering (dypac)."""

# Authors: Pierre Bellec, Amal Boukhdir
# License: BSD 3 clause
import glob
import itertools
import warnings

from scipy.sparse import vstack
import numpy as np

from sklearn.cluster import k_means
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression

from nilearn import EXPAND_PATH_WILDCARDS
from joblib import Memory
from nilearn._utils.niimg_conversions import _resolve_globbing
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from nilearn.decomposition.base import BaseDecomposition

import bascpp as bpp

import gc
class dypac(BaseDecomposition):
    """
    Perform Stable Dynamic Cluster Analysis.

    Parameters
    ----------
    n_clusters: int
        Number of clusters to extract per time window

    n_states: int
        Number of expected dynamic states

    n_replications: int
        Number of replications of cluster analysis in each fMRI run

    n_batch: int
        Number of batches to run through consensus clustering.
        If n_batch<=1, consensus clustering will be applied
        to all replications in one pass. Processing with batch will
        reduce dramatically the compute time, but will change slightly
        the results.

    n_init: int
        Number of initializations for k-means

    subsample_size: int
        Number of time points in a subsample

    max_iter: int
        Max number of iterations for k-means

    threshold_sim: float (0 <= . <= 1)
        Minimal acceptable average dice in a state

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    mask_strategy: {'background', 'epi' or 'template'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, 'epi' if they
        are raw EPI images, or you could use 'template' which will
        extract the gray matter part of your data by resampling the MNI152
        brain mask for your data's field of view.
        Depending on this value, the mask will be computed from
        masking.compute_background_mask, masking.compute_epi_mask or
        masking.compute_gray_matter_mask. Default is 'epi'.

    mask_args: dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    memory: instance of joblib.Memory or str
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    verbose: integer, optional
        Indicate the level of verbosity. By default, print progress.

    Attributes
    ----------
    `mask_img_` : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.
    """

    def __init__(
        self,
        n_clusters=10,
        n_states=3,
        n_replications=40,
        n_batch=1,
        n_init=30,
        n_init_aggregation=100,
        subsample_size=30,
        max_iter=30,
        threshold_sim=0.3,
        random_state=None,
        mask=None,
        smoothing_fwhm=None,
        standardize=True,
        detrend=True,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
        mask_strategy="epi",
        mask_args=None,
        memory=Memory(cachedir=None),
        memory_level=0,
        verbose=1,
    ):
        """Set up default attributes for the class."""
        # All those settings are taken from nilearn BaseDecomposition
        self.random_state = random_state
        self.mask = mask

        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.memory = memory
        self.memory_level = max(0, memory_level + 1)
        self.verbose = verbose

        # Those settings are specific to parcel aggregation
        self.n_clusters = n_clusters
        self.n_states = n_states
        self.n_batch = n_batch
        self.n_replications = n_replications
        self.n_init = n_init
        self.n_init_aggregation = n_init_aggregation
        self.subsample_size = subsample_size
        self.max_iter = max_iter
        self.threshold_sim = threshold_sim

    def _check_components_(self):
        """Check for presence of estimated components."""
        if not hasattr(self, "components_"):
            raise ValueError(
                "Object has no components_ attribute. "
                "This is probably because fit has not "
                "been called."
            )

    def fit(self, imgs, confounds=None):
        """
        Compute the mask and the dynamic parcels across datasets.

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which the mask is calculated. If this is a list,
            the affine is considered the same for all.

        confounds: list of CSV file paths or 2D matrices
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details. Should match with the list
            of imgs given.

         Returns
         -------
         self: object
            Returns the instance itself. Contains attributes listed
            at the object level.
        """
        # Base fit for decomposition estimators : compute the embedded masker
        if isinstance(imgs, str):
            if EXPAND_PATH_WILDCARDS and glob.has_magic(imgs):
                imgs = _resolve_globbing(imgs)

        if isinstance(imgs, str) or not hasattr(imgs, "__iter__"):
            # these classes are meant for list of 4D images
            # (multi-subject), we want it to work also on a single
            # subject, so we hack it.
            imgs = [imgs]

        if len(imgs) == 0:
            # Common error that arises from a null glob. Capture
            # it early and raise a helpful message
            raise ValueError(
                "Need one or more Niimg-like objects as input, "
                "an empty list was given."
            )
        self.masker_ = check_embedded_nifti_masker(self)

        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        # if no confounds have been specified, match length of imgs
        if confounds is None:
            confounds = list(itertools.repeat(confounds, len(imgs)))

        # Control random number generation
        self.random_state = check_random_state(self.random_state)

        # Check that number of batches is reasonable
        if self.n_batch > len(imgs):
            warnings.warn("{0} batches were requested, but only {1} datasets available. Using one dataset per batch instead.".format(self.n_batch, len(imgs)))
            self.n_batch = len(imgs)

        # mask_and_reduce step
        if (self.n_batch > 1):
            stab_maps, dwell_time = self._mask_and_reduce_batch(imgs, confounds)
        else:
            stab_maps, dwell_time = self._mask_and_reduce(imgs, confounds)

        # Return components
        self.components_ = stab_maps
        self.dwell_time_ = dwell_time
        return self


    def _mask_and_reduce_batch(self, imgs, confounds=None):
        """Iterate dypac on batches of files."""
        for bb in range(self.n_batch):
            slice_batch = slice(bb, len(imgs), self.n_batch)
            if self.verbose:
                print("[{0}] Processing batch {1}".format(self.__class__.__name__, bb))
            stab_maps, dwell_time = self._mask_and_reduce(imgs[slice_batch], confounds[slice_batch])
            if bb == 0:
                stab_maps_all = stab_maps
                dwell_time_all = dwell_time
            else:
                stab_maps_all = vstack([stab_maps_all, stab_maps])
                dwell_time_all = np.concatenate([dwell_time_all, dwell_time])

        # Consensus clustering step
        _, states_all, _ = k_means(
            stab_maps_all,
            n_clusters=self.n_states,
            init="k-means++",
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=self.n_init_aggregation,
        )

        # average stability maps and dwell times across consensus states
        stab_maps_cons, dwell_time_cons = bpp.stab_maps(stab_maps_all, states_all,
            self.n_replications, self.n_states, dwell_time_all)

        return stab_maps_cons, dwell_time_cons

    def _mask_and_reduce(self, imgs, confounds=None):
        """Uses cluster aggregation over sliding windows to estimate
        stable dynamic parcels from a list of 4D fMRI datasets.

        Returns
        -------
        stab_maps: ndarray
            stability maps of each state.

        dwell_time: ndarray
            dwell time of each state.
        """

        for ind, img, confound in zip(range(len(imgs)), imgs, confounds):
            this_data = self.masker_.transform(img, confound)
            # Now get rid of the img as fast as possible, to free a
            # reference count on it, and possibly free the corresponding
            # data
            del img
            onehot = bpp.replicate_clusters(
                this_data.transpose(),
                subsample_size=self.subsample_size,
                n_clusters=self.n_clusters,
                n_replications=self.n_replications,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                desc="Replicating clusters in data #{0}".format(ind),
                verbose=self.verbose,
            )
            if ind == 0:
                onehot_all = onehot
            else:
                onehot_all = vstack([onehot_all, onehot])
        
        del this_data
        gc.collect()
        # find the states
        states = bpp.find_states(
            onehot_all,
            n_states=self.n_states,
            max_iter=self.max_iter,
            threshold_sim=self.threshold_sim,
            random_state=self.random_state,
            n_init=self.n_init_aggregation,
            verbose=self.verbose,
        )

        # Generate the stability maps
        stab_maps, dwell_time = bpp.stab_maps(
            onehot_all, states, self.n_replications, self.n_states
        )

        return stab_maps, dwell_time

    def transform_sparse(self, img, confound=None):
        """Transform a 4D dataset in a component space."""
        self._check_components_()
        this_data = self.masker_.transform(img, confound)
        del img
        reg = LinearRegression().fit(
            self.components_.transpose(), this_data.transpose()
        )
        return reg.coef_

    def inverse_transform_sparse(self, weights):
        """Transform component weights as a 4D dataset."""
        self._check_components_()
        self.masker_.inverse_transform(weights * self.components_)
