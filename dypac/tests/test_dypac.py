import numpy as np

from nibabel import Nifti1Image

from nilearn.image import new_img_like

from dypac import Dypac
from dypac.tests.test_bascpp import simu_tseries


def test_fit():
    # Generate a single state with obvious structure
    n_time = 100
    tseries, _ = simu_tseries(n_time=n_time, n_roi=125, n_clusters=3, alpha=2)
    img = Nifti1Image(np.reshape(tseries, [5, 5, 5, n_time]), np.eye(4))

    # Generate a mask for the image
    mask = new_img_like(img, np.ones([5, 5, 5]), np.eye(4))

    # fit a dypac model
    n_replications = 10
    n_clusters = 3
    n_states = 3
    model = Dypac(
        n_clusters=n_clusters,
        n_states=3,
        min_size=0,
        n_replications=n_replications,
        mask=mask,
        grey_matter=None,
    )
    model.fit(img)
    print(model.components_.shape)

    # Check that the fitted model has the right number of components
    assert model.components_.shape[0] == n_states
